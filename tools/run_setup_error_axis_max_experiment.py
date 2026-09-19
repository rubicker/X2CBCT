from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from run_setup_error_focus_experiment import (
    DEFAULT_CACHE_ROOTS,
    DEFAULT_FORMAL_RESULTS,
    DEFAULT_ROI_REVIEWED,
    FocusPair,
    _candidate_splits,
    _comparison_rows,
    _denormalize_cbct,
    _export_focus_viewer,
    _find_case_payload,
    _find_sample_index,
    _read_formal_results,
    _run_registration,
    _run_inference_for_pairs,
    _run_prediction_registration,
    _write_comparison_outputs,
    _write_nifti,
)
from x2cbct_deepsparse import DeepSparseCachedDataset, build_deepsparse_model
from x2cbct_difnet.metrics import mean_metrics, metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.setup_error_registration import load_roi_review
from x2cbct_difnet.setup_error_resampling import (
    evaluation_grid,
    resample_to_evaluation_grid,
    rescale_roi_row,
)


AXIS_CHOICES = ("si-max", "yaw-max", "both")
BACKEND_CHOICES = ("difnet", "deepsparse")
DEEPSPARSE_TENSOR_KEYS = (
    "projs",
    "points_ct",
    "points_gt",
    "points_proj",
    "points_lr_proj",
    "view_mask",
    "points_valid_mask",
    "points_lr_valid_mask",
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is only supported on CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _to_deepsparse_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key in DEEPSPARSE_TENSOR_KEYS:
        value = batch[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.bool() if key.endswith("mask") else tensor.float()
        tensors[key] = tensor.to(device, non_blocking=device.type == "cuda")
    return tensors


def _parse_pair_spec(value: str) -> FocusPair:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in (3, 5) or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "pair must use PATIENT:REFERENCE:MOVING or "
            "PATIENT:REFERENCE:MOVING:REFERENCE_AUG:MOVING_AUG, "
            f"got {value!r}"
        )
    return FocusPair(*parts)


def _pair_to_dict(pair: FocusPair) -> dict[str, str | None]:
    return {
        "patient_id": pair.patient_id,
        "reference_case_id": pair.reference_case_id,
        "moving_case_id": pair.moving_case_id,
        "reference_augmentation_id": pair.reference_augmentation_id,
        "moving_augmentation_id": pair.moving_augmentation_id,
        "pair_key": pair.pair_key,
    }


def _select_axis_max_pair(
    formal_results: dict[tuple[str, str, str], dict[str, str]],
    axis: str,
) -> FocusPair:
    if axis == "si-max":
        signed_field = "setup_error_SI_mm"
        absolute_field = "setup_error_abs_SI_mm"
    elif axis == "yaw-max":
        signed_field = "setup_error_yaw_deg"
        absolute_field = "setup_error_abs_yaw_deg"
    else:
        raise ValueError(f"unsupported axis: {axis}")

    candidates = [
        (key, row)
        for key, row in formal_results.items()
        if str(row.get("status", "")).lower() == "ok"
    ]
    if not candidates:
        raise ValueError("formal registration results contain no status=ok rows")

    def magnitude(row: dict[str, str]) -> float:
        raw = row.get(absolute_field)
        return float(raw) if raw not in (None, "") else abs(float(row[signed_field]))

    key, _ = max(candidates, key=lambda item: magnitude(item[1]))
    return FocusPair(str(key[0]), str(key[1]), str(key[2]))


def _selected_axes(value: str) -> list[str]:
    if value == "both":
        return ["si-max", "yaw-max"]
    if value in {"si-max", "yaw-max"}:
        return [value]
    raise ValueError(f"unsupported experiment selection: {value}")


def _select_pairs(
    args: argparse.Namespace,
    formal_results: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, FocusPair]:
    pairs: dict[str, FocusPair] = {}
    for axis in _selected_axes(args.experiment):
        override = args.si_pair if axis == "si-max" else args.yaw_pair
        pairs[axis] = override or _select_axis_max_pair(formal_results, axis)
    return pairs


def _checkpoint_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return []
    candidates = [root]
    child = root / "checkpoints"
    if child.is_dir():
        candidates.append(child)
    return candidates


def _list_checkpoints(root: Path) -> list[dict[str, Any]]:
    resolved = root.resolve()
    if resolved.is_file():
        files = [resolved]
    elif resolved.is_dir():
        files = []
        for directory in _checkpoint_dirs(resolved):
            files.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
            )
    else:
        raise FileNotFoundError(f"checkpoint path does not exist: {resolved}")

    unique = sorted({path.resolve() for path in files}, key=lambda path: (path.parent.name, path.name))
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for path in unique
    ]


def _resolve_cache_roots(values: list[Path] | None) -> tuple[list[Path], list[str]]:
    requested = values if values is not None else list(DEFAULT_CACHE_ROOTS)
    resolved: list[Path] = []
    missing: list[str] = []
    for value in requested:
        path = _project_path(value)
        if not path.is_dir():
            missing.append(str(path))
            continue
        manifest = path / "cache_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"cache root has no cache_manifest.json: {path}")
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise FileNotFoundError(
            "none of the requested cache roots exist; pass one or more --cache-root values"
        )
    if values is not None and missing:
        raise FileNotFoundError(f"explicit cache roots do not exist: {missing}")
    return resolved, missing


def _order_cache_roots_for_split(
    cache_roots: list[Path],
    required_split: str,
) -> list[Path]:
    if required_split == "any":
        return list(cache_roots)
    preferred = [
        root
        for root in cache_roots
        if required_split in _candidate_splits(root)
    ]
    remaining = [root for root in cache_roots if root not in preferred]
    return preferred + remaining


def _checkpoint_run_config(checkpoint: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        checkpoint.parent / "config_resolved.json",
        checkpoint.parent.parent / "config_resolved.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint run config must be a JSON object: {path}")
        return value, path
    return None, None


def _configured_view_count(config: dict[str, Any] | None) -> int | None:
    if config is None:
        return None
    if config.get("view_count") is not None:
        return int(config["view_count"])
    dataset = config.get("dataset")
    if isinstance(dataset, dict) and dataset.get("view_count") is not None:
        return int(dataset["view_count"])
    return None


def _configured_backend(config: dict[str, Any] | None) -> str | None:
    if config is None:
        return None
    model = config.get("model")
    if isinstance(model, dict) and model.get("official_config"):
        return "deepsparse"
    if config.get("model_version") is not None or config.get("combine") is not None:
        return "difnet"
    return None


def _preflight_backend_parameters(
    args: argparse.Namespace,
    run_config: dict[str, Any] | None,
) -> dict[str, Any]:
    configured_backend = _configured_backend(run_config)
    if configured_backend is not None and configured_backend != args.backend:
        raise ValueError(
            f"checkpoint run config indicates backend={configured_backend}, but "
            f"--backend={args.backend}"
        )
    if args.backend == "difnet":
        configured_mid_channels = (
            int(run_config["mid_channels"])
            if isinstance(run_config, dict) and run_config.get("mid_channels") is not None
            else None
        )
        if (
            configured_mid_channels is not None
            and configured_mid_channels != int(args.mid_channels)
        ):
            raise ValueError(
                f"checkpoint run config uses mid_channels={configured_mid_channels}, "
                f"but --mid-channels={args.mid_channels}"
            )
        return {
            "backend": "difnet",
            "mid_channels": args.mid_channels,
            "configured_mid_channels": configured_mid_channels,
            "configured_backend": configured_backend,
        }

    if args.deepsparse_config is not None:
        deepsparse_config = _project_path(args.deepsparse_config)
        if not deepsparse_config.is_file():
            raise FileNotFoundError(
                f"DeepSparse override config does not exist: {deepsparse_config}"
            )
    else:
        deepsparse_config = None

    model_config = run_config.get("model", {}) if isinstance(run_config, dict) else {}
    training_config = (
        run_config.get("training", {}) if isinstance(run_config, dict) else {}
    )
    dataset_config = run_config.get("dataset", {}) if isinstance(run_config, dict) else {}
    runtime_config = run_config.get("runtime", {}) if isinstance(run_config, dict) else {}
    official_config = _project_path(
        args.official_config
        if args.official_config is not None
        else model_config.get(
            "official_config",
            "external/deepsparse_official/configs/finetune_s1.yaml",
        )
    )
    if not official_config.is_file():
        raise FileNotFoundError(
            f"DeepSparse official config does not exist: {official_config}"
        )
    num_points = int(
        args.deepsparse_num_points
        if args.deepsparse_num_points is not None
        else training_config.get("num_points", 10_000)
    )
    lr_res = int(
        args.deepsparse_lr_res
        if args.deepsparse_lr_res is not None
        else model_config.get("lr_res", 32)
    )
    projection_multiplier = float(
        args.projection_multiplier
        if args.projection_multiplier is not None
        else dataset_config.get("projection_multiplier", 1.0)
    )
    seed = int(args.seed if args.seed is not None else runtime_config.get("seed", 20260721))
    mixed_precision = (
        bool(args.mixed_precision)
        if args.mixed_precision is not None
        else bool(runtime_config.get("mixed_precision", True))
    )
    if num_points < 1:
        raise ValueError("--deepsparse-num-points must be >= 1")
    if lr_res < 2:
        raise ValueError("--deepsparse-lr-res must be >= 2")
    if not np.isfinite(projection_multiplier) or projection_multiplier <= 0.0:
        raise ValueError("--projection-multiplier must be a finite positive number")
    return {
        "backend": "deepsparse",
        "configured_backend": configured_backend,
        "deepsparse_config": str(deepsparse_config) if deepsparse_config else None,
        "official_config": str(official_config),
        "num_points": num_points,
        "lr_res": lr_res,
        "projection_multiplier": projection_multiplier,
        "seed": seed,
        "mixed_precision": mixed_precision,
    }


def _find_inference_location(
    cache_roots: list[Path],
    *,
    view_key: str,
    patient_id: str,
    case_id: str,
    augmentation_id: str | None = None,
    required_split: str = "any",
) -> dict[str, Any]:
    searched: list[str] = []
    for cache_root in cache_roots:
        for split in _candidate_splits(cache_root):
            if required_split != "any" and split != required_split:
                continue
            searched.append(f"{cache_root}::{split}::{view_key}")
            index = _find_sample_index(
                cache_root,
                split,
                view_key,
                patient_id,
                case_id,
                augmentation_id,
            )
            if index is not None:
                return {
                    "cache_root": str(cache_root),
                    "split": split,
                    "dataset_index": index,
                    "view_key": view_key,
                }
    raise FileNotFoundError(
        f"no inference sample for patient={patient_id}, case={case_id}, "
        f"augmentation={augmentation_id}, "
        f"view_key={view_key}; searched={searched}"
    )


def _processed_size_xyz(case_manifest: dict[str, Any]) -> tuple[int, int, int]:
    values = case_manifest.get("processed_size_xyz")
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("case manifest has no valid processed_size_xyz")
    return tuple(int(value) for value in values)


def _preflight(
    args: argparse.Namespace,
    pairs: dict[str, FocusPair],
    formal_results: dict[tuple[str, str, str], dict[str, str]],
    cache_roots: list[Path],
    missing_default_caches: list[str],
) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"checkpoint is empty: {checkpoint}")
    run_config, run_config_path = _checkpoint_run_config(checkpoint)
    configured_view_count = _configured_view_count(run_config)
    if configured_view_count is not None and configured_view_count != args.view_count:
        raise ValueError(
            f"checkpoint run config uses view_count={configured_view_count}, but "
            f"--view-count={args.view_count}: {run_config_path}"
        )
    backend_parameters = _preflight_backend_parameters(args, run_config)

    roi_by_case = load_roi_review(args.roi_reviewed)
    view_key = view_key_from_count(args.view_count)
    planned: list[dict[str, Any]] = []
    warnings: list[str] = []
    if missing_default_caches:
        warnings.append(
            "Missing built-in cache roots were ignored: "
            + ", ".join(missing_default_caches)
        )
    if run_config is None:
        warnings.append(
            "No adjacent config_resolved.json was found; checkpoint view count "
            "will be validated only when the model state is loaded."
        )
    for axis, pair in pairs.items():
        key = (pair.patient_id, pair.reference_case_id, pair.moving_case_id)
        formal = formal_results.get(key)
        if formal is None:
            raise KeyError(f"formal registration result is missing for {key}")
        if str(formal.get("status", "")).lower() != "ok":
            raise ValueError(f"formal registration is not status=ok for {key}: {formal.get('status')}")

        for case_id in (pair.reference_case_id, pair.moving_case_id):
            roi = roi_by_case.get((pair.patient_id, case_id))
            if roi is None:
                raise KeyError(f"reviewed ROI is missing for patient={pair.patient_id}, case={case_id}")
            if str(roi.get("status", "")).lower() != "accepted":
                raise ValueError(
                    f"ROI is not accepted for patient={pair.patient_id}, case={case_id}: "
                    f"{roi.get('status')}"
                )

        reference_payload, reference_manifest, _ = _find_case_payload(
            cache_roots,
            pair.patient_id,
            pair.reference_case_id,
            pair.reference_augmentation_id,
        )
        moving_payload, moving_manifest, _ = _find_case_payload(
            cache_roots,
            pair.patient_id,
            pair.moving_case_id,
            pair.moving_augmentation_id,
        )
        inference = _find_inference_location(
            cache_roots,
            view_key=view_key,
            patient_id=pair.patient_id,
            case_id=pair.moving_case_id,
            augmentation_id=pair.moving_augmentation_id,
            required_split=args.require_inference_split,
        )
        if inference["split"] != "test":
            warnings.append(
                f"{axis} moving case {pair.moving_case_key} is read from "
                f"split={inference['split']}, not a held-out test split."
            )

        reference_size = _processed_size_xyz(reference_manifest)
        moving_size = _processed_size_xyz(moving_manifest)
        expected_size = (int(args.eval_grid_size),) * 3
        if reference_size != moving_size:
            raise ValueError(
                f"reference/moving processed sizes differ for {pair.pair_key}: "
                f"{reference_size} != {moving_size}"
            )
        reference_eval_grid = evaluation_grid(
            reference_manifest,
            int(args.eval_grid_size),
        )
        moving_eval_grid = evaluation_grid(
            moving_manifest,
            int(args.eval_grid_size),
        )
        if moving_size != expected_size:
            warnings.append(
                f"{axis} registration inputs will be resampled from "
                f"{moving_size} to {expected_size} while preserving physical extent."
            )

        planned.append(
            {
                "axis": axis,
                "selection_source": (
                    "override"
                    if (axis == "si-max" and args.si_pair is not None)
                    or (axis == "yaw-max" and args.yaw_pair is not None)
                    else "formal_global_max"
                ),
                **_pair_to_dict(pair),
                "true_setup_error_SI_mm": float(formal["setup_error_SI_mm"]),
                "true_setup_error_yaw_deg": float(formal["setup_error_yaw_deg"]),
                "reference_payload": str(reference_payload),
                "moving_payload": str(moving_payload),
                "source_processed_size_xyz": list(moving_size),
                "source_processed_spacing_xyz": moving_manifest.get(
                    "processed_spacing_xyz"
                ),
                "registration_grid_policy": (
                    "endpoint_aligned_preserve_physical_extent"
                ),
                "registration_grid_size_xyz": list(moving_eval_grid.size_xyz),
                "registration_grid_spacing_xyz": list(
                    moving_eval_grid.spacing_xyz
                ),
                "registration_grid_origin_xyz": list(moving_eval_grid.origin_xyz),
                "registration_grid_direction": list(moving_eval_grid.direction),
                "reference_registration_grid_spacing_xyz": list(
                    reference_eval_grid.spacing_xyz
                ),
                "inference": inference,
                "output_dir": str((args.output_dir / axis.replace("-", "_")).resolve()),
            }
        )

    return {
        "schema": "x2cbct_setup_error_axis_max_plan.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "model_label": args.model_label,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_run_config": str(run_config_path) if run_config_path else None,
        "checkpoint_configured_view_count": configured_view_count,
        "backend_parameters": backend_parameters,
        "view_count": args.view_count,
        "eval_grid_size": args.eval_grid_size,
        "grid_control_tolerance": {
            "SI_mm": args.grid_control_si_tolerance_mm,
            "yaw_deg": args.grid_control_yaw_tolerance_deg,
            "enforced": bool(args.enforce_grid_control),
        },
        "chunk_size": args.chunk_size,
        "device": args.device,
        "cache_roots": [str(path) for path in cache_roots],
        "missing_default_cache_roots_ignored": missing_default_caches,
        "warnings": warnings,
        "experiments": planned,
    }


def _geometry_summary(case_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_xyz": [
            int(value) for value in case_manifest["processed_size_xyz"]
        ],
        "spacing_xyz": [
            float(value) for value in case_manifest["processed_spacing_xyz"]
        ],
        "origin_xyz": [
            float(value) for value in case_manifest["processed_origin_xyz"]
        ],
        "direction": [
            float(value) for value in case_manifest["processed_direction"]
        ],
        "physical_fov_mm_xyz": [
            float(value)
            * float(int(size) - 1)
            for value, size in zip(
                case_manifest["processed_spacing_xyz"],
                case_manifest["processed_size_xyz"],
            )
        ],
    }


def _prepare_evaluation_grid_registration_inputs(
    args: argparse.Namespace,
    pairs: list[FocusPair],
    predictions: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    cache_roots = [path.resolve() for path in args.cache_root]
    source_roi_by_case = load_roi_review(args.roi_reviewed)
    pred_by_pair = {str(row["pair_key"]): row for row in predictions}
    roi_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    true_reg_pairs: list[dict[str, Any]] = []
    pred_reg_pairs: list[dict[str, Any]] = []
    volume_info: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    volume_root = args.output_dir / "setup_error" / "registration_inputs"
    target_shape = (int(args.eval_grid_size),) * 3

    for pair in pairs:
        ref_payload, ref_manifest, ref_cache_manifest = _find_case_payload(
            cache_roots,
            pair.patient_id,
            pair.reference_case_id,
            pair.reference_augmentation_id,
        )
        mov_payload, mov_manifest, mov_cache_manifest = _find_case_payload(
            cache_roots,
            pair.patient_id,
            pair.moving_case_id,
            pair.moving_augmentation_id,
        )
        pred_row = pred_by_pair[pair.pair_key]
        pred_norm = np.load(pred_row["generated_cbct_path"]).astype(np.float32)
        if tuple(pred_norm.shape) != target_shape:
            raise ValueError(
                f"prediction shape for {pair.pair_key} is {tuple(pred_norm.shape)}, "
                f"expected {target_shape}"
            )

        ref_norm = np.load(ref_payload).astype(np.float32)
        mov_norm = np.load(mov_payload).astype(np.float32)
        ref_eval_norm, ref_eval_manifest = resample_to_evaluation_grid(
            ref_norm,
            ref_manifest,
            int(args.eval_grid_size),
            default_value=0.0,
        )
        mov_eval_norm, mov_eval_manifest = resample_to_evaluation_grid(
            mov_norm,
            mov_manifest,
            int(args.eval_grid_size),
            default_value=0.0,
        )

        pair_dir = volume_root / pair.pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)
        ref_norm_path = pair_dir / "C_reference_cbct_norm.npy"
        mov_norm_path = pair_dir / "B_treatment_cbct_norm.npy"
        pred_norm_path = pair_dir / "B_hat_reconstruction_norm.npy"
        np.save(ref_norm_path, ref_eval_norm)
        np.save(mov_norm_path, mov_eval_norm)
        np.save(pred_norm_path, pred_norm)

        ref_hu = _denormalize_cbct(
            ref_eval_norm,
            ref_cache_manifest["cbct_normalization"],
        )
        mov_hu = _denormalize_cbct(
            mov_eval_norm,
            mov_cache_manifest["cbct_normalization"],
        )
        pred_hu = _denormalize_cbct(pred_norm, pred_row["normalization"])
        ref_nii = _write_nifti(
            ref_hu,
            ref_eval_manifest,
            pair_dir / "C_reference_cbct_hu.nii.gz",
        )
        mov_nii = _write_nifti(
            mov_hu,
            mov_eval_manifest,
            pair_dir / "B_treatment_cbct_hu.nii.gz",
        )
        pred_nii = _write_nifti(
            pred_hu,
            mov_eval_manifest,
            pair_dir / "B_hat_reconstruction_hu.nii.gz",
        )

        ref_source_roi = source_roi_by_case.get(
            (pair.patient_id, pair.reference_case_id)
        )
        mov_source_roi = source_roi_by_case.get(
            (pair.patient_id, pair.moving_case_id)
        )
        if ref_source_roi is None or mov_source_roi is None:
            raise KeyError(f"reviewed ROI is missing for {pair.pair_key}")
        ref_source_size = _processed_size_xyz(ref_manifest)
        mov_source_size = _processed_size_xyz(mov_manifest)
        ref_roi = rescale_roi_row(
            ref_source_roi,
            source_size_xyz=ref_source_size,
            target_size_xyz=target_shape,
        )
        mov_roi = rescale_roi_row(
            mov_source_roi,
            source_size_xyz=mov_source_size,
            target_size_xyz=target_shape,
        )
        ref_roi["cbct_path"] = str(ref_nii)
        mov_roi["cbct_path"] = str(mov_nii)
        pred_roi = dict(mov_roi)
        pred_roi["case_id"] = pair.pred_case_id
        pred_roi["cbct_path"] = str(pred_nii)
        roi_by_case[(pair.patient_id, pair.reference_case_id)] = ref_roi
        roi_by_case[(pair.patient_id, pair.moving_case_id)] = mov_roi
        roi_by_case[(pair.patient_id, pair.pred_case_id)] = pred_roi

        true_reg_pairs.append(
            {
                "patient_id": pair.patient_id,
                "reference_case_id": pair.reference_case_id,
                "moving_case_id": pair.moving_case_id,
                "reference_cbct_path": str(ref_nii),
                "moving_cbct_path": str(mov_nii),
            }
        )
        pred_reg_pairs.append(
            {
                "patient_id": pair.patient_id,
                "reference_case_id": pair.reference_case_id,
                "moving_case_id": pair.pred_case_id,
                "reference_cbct_path": str(ref_nii),
                "moving_cbct_path": str(pred_nii),
            }
        )
        volume_info[pair.pair_key] = {
            "reference_norm_path": str(ref_norm_path),
            "moving_norm_path": str(mov_norm_path),
            "prediction_norm_path": str(pred_norm_path),
            "reference_hu_nifti": str(ref_nii),
            "moving_hu_nifti": str(mov_nii),
            "prediction_hu_nifti": str(pred_nii),
            "processed_spacing_xyz": list(
                mov_eval_manifest["processed_spacing_xyz"]
            ),
            "registration_grid_size_xyz": list(target_shape),
            "source_grid_size_xyz": list(mov_source_size),
        }
        audit_rows.append(
            {
                "pair_key": pair.pair_key,
                "policy": "endpoint_aligned_preserve_physical_extent",
                "interpolator": "linear",
                "reference_source_geometry": _geometry_summary(ref_manifest),
                "reference_registration_geometry": _geometry_summary(
                    ref_eval_manifest
                ),
                "moving_source_geometry": _geometry_summary(mov_manifest),
                "moving_registration_geometry": _geometry_summary(
                    mov_eval_manifest
                ),
                "reference_roi_source": {
                    key: ref_source_roi[key]
                    for key in (
                        "x_min",
                        "x_max",
                        "y_min",
                        "y_max",
                        "z_min",
                        "z_max",
                    )
                },
                "reference_roi_registration": {
                    key: ref_roi[key]
                    for key in (
                        "x_min",
                        "x_max",
                        "y_min",
                        "y_max",
                        "z_min",
                        "z_max",
                    )
                },
                "moving_roi_source": {
                    key: mov_source_roi[key]
                    for key in (
                        "x_min",
                        "x_max",
                        "y_min",
                        "y_max",
                        "z_min",
                        "z_max",
                    )
                },
                "moving_roi_registration": {
                    key: mov_roi[key]
                    for key in (
                        "x_min",
                        "x_max",
                        "y_min",
                        "y_max",
                        "z_min",
                        "z_max",
                    )
                },
            }
        )

    _write_json(
        args.output_dir / "setup_error" / "registration_grid_audit.json",
        {
            "schema": "x2cbct_setup_error_registration_grid.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "eval_grid_size": int(args.eval_grid_size),
            "rows": audit_rows,
        },
    )
    return true_reg_pairs, pred_reg_pairs, roi_by_case, volume_info


def _registration_result_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(row["patient_id"]),
            str(row["reference_case_id"]),
            str(row["moving_case_id"]),
        ): row
        for row in rows
    }


def _grid_control_row(
    pair: FocusPair,
    *,
    source_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_si = float(source_result["setup_error_SI_mm"])
    source_yaw = float(source_result["setup_error_yaw_deg"])
    evaluation_si = float(evaluation_result["setup_error_SI_mm"])
    evaluation_yaw = float(evaluation_result["setup_error_yaw_deg"])
    delta_si = evaluation_si - source_si
    delta_yaw = evaluation_yaw - source_yaw
    si_pass = abs(delta_si) <= float(args.grid_control_si_tolerance_mm)
    yaw_pass = abs(delta_yaw) <= float(args.grid_control_yaw_tolerance_deg)
    return {
        "pair_key": pair.pair_key,
        "source_grid_status": source_result["status"],
        "evaluation_grid_status": evaluation_result["status"],
        "source_grid_size": _processed_size_xyz(
            _find_case_payload(
                [path.resolve() for path in args.cache_root],
                pair.patient_id,
                pair.moving_case_id,
                pair.moving_augmentation_id,
            )[1]
        )[0],
        "evaluation_grid_size": int(args.eval_grid_size),
        "source_grid_setup_error_SI_mm": source_si,
        "evaluation_grid_setup_error_SI_mm": evaluation_si,
        "evaluation_minus_source_SI_mm": delta_si,
        "abs_evaluation_minus_source_SI_mm": abs(delta_si),
        "SI_tolerance_mm": float(args.grid_control_si_tolerance_mm),
        "SI_pass": si_pass,
        "source_grid_setup_error_yaw_deg": source_yaw,
        "evaluation_grid_setup_error_yaw_deg": evaluation_yaw,
        "evaluation_minus_source_yaw_deg": delta_yaw,
        "abs_evaluation_minus_source_yaw_deg": abs(delta_yaw),
        "yaw_tolerance_deg": float(args.grid_control_yaw_tolerance_deg),
        "yaw_pass": yaw_pass,
        "overall_pass": bool(si_pass and yaw_pass),
    }


def _write_grid_control_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    control_dir = output_dir / "setup_error" / "grid_resolution_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with (control_dir / "grid_resolution_control.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        control_dir / "grid_resolution_control.json",
        {
            "schema": "x2cbct_setup_error_grid_resolution_control.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows,
        },
    )


def _attach_grid_control(
    comparison_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control_by_pair = {str(row["pair_key"]): row for row in controls}
    enriched = []
    for row in comparison_rows:
        control = control_by_pair[str(row["pair_key"])]
        enriched.append(
            {
                **row,
                "true_registration_grid_size": control[
                    "evaluation_grid_size"
                ],
                "source_formal_grid_size": control["source_grid_size"],
                "source_formal_setup_error_SI_mm": control[
                    "source_grid_setup_error_SI_mm"
                ],
                "source_formal_setup_error_yaw_deg": control[
                    "source_grid_setup_error_yaw_deg"
                ],
                "grid_control_delta_SI_mm": control[
                    "evaluation_minus_source_SI_mm"
                ],
                "grid_control_abs_delta_SI_mm": control[
                    "abs_evaluation_minus_source_SI_mm"
                ],
                "grid_control_delta_yaw_deg": control[
                    "evaluation_minus_source_yaw_deg"
                ],
                "grid_control_abs_delta_yaw_deg": control[
                    "abs_evaluation_minus_source_yaw_deg"
                ],
                "grid_control_pass": control["overall_pass"],
            }
        )
    return enriched


def _find_deepsparse_dataset_for_case(
    cache_roots: list[Path],
    *,
    view_count: int,
    patient_id: str,
    case_id: str,
    augmentation_id: str | None,
    num_points: int,
    lr_res: int,
    seed: int,
    eval_grid_size: int,
    projection_multiplier: float,
) -> tuple[DeepSparseCachedDataset, int]:
    view_key = view_key_from_count(view_count)
    for cache_root in cache_roots:
        for split in _candidate_splits(cache_root):
            index = _find_sample_index(
                cache_root,
                split,
                view_key,
                patient_id,
                case_id,
                augmentation_id,
            )
            if index is None:
                continue
            dataset = DeepSparseCachedDataset(
                cache_root,
                split=split,
                view_count=view_count,
                num_points=num_points,
                lr_res=lr_res,
                seed=seed,
                deterministic=True,
                eval_grid_size=eval_grid_size,
                projection_multiplier=projection_multiplier,
            )
            return dataset, index
    raise FileNotFoundError(
        f"no DeepSparse inference sample for patient={patient_id}, case={case_id}, "
        f"augmentation={augmentation_id}, "
        f"view_key={view_key}"
    )


def _deepsparse_options(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    file_config = (
        _read_yaml(_project_path(args.deepsparse_config))
        if args.deepsparse_config is not None
        else {}
    )
    config = _deep_merge(checkpoint_config, file_config)
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    runtime_config = config.get("runtime", {})

    checkpoint_view_count = dataset_config.get("view_count")
    if checkpoint_view_count is not None and int(checkpoint_view_count) != args.view_count:
        raise ValueError(
            "DeepSparse checkpoint view_count does not match requested view_count: "
            f"{checkpoint_view_count} != {args.view_count}"
        )

    seed = int(args.seed if args.seed is not None else runtime_config.get("seed", 20260721))
    num_points = int(
        args.deepsparse_num_points
        if args.deepsparse_num_points is not None
        else training_config.get("num_points", 10_000)
    )
    lr_res = int(
        args.deepsparse_lr_res
        if args.deepsparse_lr_res is not None
        else model_config.get("lr_res", 32)
    )
    projection_multiplier = float(
        args.projection_multiplier
        if args.projection_multiplier is not None
        else dataset_config.get("projection_multiplier", 1.0)
    )
    mixed_precision = (
        bool(args.mixed_precision)
        if args.mixed_precision is not None
        else bool(runtime_config.get("mixed_precision", True))
    )
    if device.type != "cuda":
        mixed_precision = False
    official_config = _project_path(
        args.official_config
        if args.official_config is not None
        else model_config.get(
            "official_config",
            "external/deepsparse_official/configs/finetune_s1.yaml",
        )
    )
    return {
        "checkpoint_config": config,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "seed": seed,
        "num_points": num_points,
        "lr_res": lr_res,
        "projection_multiplier": projection_multiplier,
        "mixed_precision": mixed_precision,
        "official_config": official_config,
    }


def _run_deepsparse_inference_for_pairs(
    args: argparse.Namespace,
    pairs: list[FocusPair],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = _resolve_device(args.device)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    options = _deepsparse_options(args, checkpoint, device)
    _seed_everything(int(options["seed"]))

    model = build_deepsparse_model(
        PROJECT_ROOT,
        config_path=options["official_config"],
    ).to(device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    cache_roots = [path.resolve() for path in args.cache_root]
    view_key = view_key_from_count(args.view_count)
    predictions: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    pred_dir = args.output_dir / "predictions"
    generated_dir = pred_dir / "generated_cbct"
    target_dir = pred_dir / "target_cbct"

    with torch.no_grad():
        for pair in pairs:
            dataset, index = _find_deepsparse_dataset_for_case(
                cache_roots,
                view_count=args.view_count,
                patient_id=pair.patient_id,
                case_id=pair.moving_case_id,
                augmentation_id=pair.moving_augmentation_id,
                num_points=int(options["num_points"]),
                lr_res=int(options["lr_res"]),
                seed=int(options["seed"]),
                eval_grid_size=args.eval_grid_size,
                projection_multiplier=float(options["projection_multiplier"]),
            )
            item = dataset[index]
            batch = {
                key: np.expand_dims(item[key], axis=0)
                for key in DEEPSPARSE_TENSOR_KEYS
            }
            tensors = _to_deepsparse_device(batch, device)
            with _autocast(device, bool(options["mixed_precision"])):
                result = model(tensors, is_eval=True, eval_npoint=int(args.chunk_size))
            shape = tuple(int(value) for value in item["image"].shape)
            generated = (
                result["points_pred"][0, 0]
                .float()
                .cpu()
                .numpy()
                .reshape(shape)
            )
            generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
            target = np.asarray(item["image"], dtype=np.float32)
            normalization = dataset.cache_manifest["cbct_normalization"]
            metrics = metric_dict(generated, target, normalization=normalization)
            trace = item["trace"]
            case_key = str(trace["case_key"])
            generated_path = generated_dir / f"{case_key}_{view_key}.npy"
            target_path = target_dir / f"{case_key}_{view_key}.npy"
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(generated_path, generated)
            np.save(target_path, target)

            resolved_model = {
                "backend": "deepsparse",
                "official_config": str(options["official_config"]),
                "checkpoint_epoch": options["checkpoint_epoch"],
                "lr_res": options["lr_res"],
                "num_points": options["num_points"],
                "projection_multiplier": options["projection_multiplier"],
                "mixed_precision": options["mixed_precision"],
                "seed": options["seed"],
            }
            row = {
                "patient_id": trace["patient_id"],
                "case_id": trace["case_id"],
                "case_key": case_key,
                "reference_case_id": pair.reference_case_id,
                "reference_case_key": pair.reference_case_key,
                "pair_key": pair.pair_key,
                "view_count": args.view_count,
                "view_key": view_key,
                "inference_cache_root": trace["cache_root"],
                "inference_cache_version": trace["cache_version"],
                "inference_split": trace["split"],
                "source_cache_item_path": trace["sample_path"],
                "generated_cbct_path": str(generated_path),
                "target_cbct_path": str(target_path),
                "geometry_branch": trace.get("geometry_branch"),
                "processed_shape_zyx": trace.get("processed_shape_zyx"),
                "processed_spacing_xyz": trace.get("processed_spacing_xyz"),
                "processed_origin_xyz": trace.get("processed_origin_xyz"),
                "processed_direction": trace.get("processed_direction"),
                "normalization": normalization,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "model_version": "deepsparse",
                "model_config": resolved_model,
                "eval_grid_size": args.eval_grid_size,
                "metrics": metrics,
            }
            predictions.append(row)
            metrics_rows.append({"case_key": case_key, "pair_key": pair.pair_key, **metrics})
            print(
                f"[inference:deepsparse] {pair.pair_key} "
                f"cache={trace['cache_version']} split={trace['split']} "
                f"psnr={metrics['psnr']:.3f}",
                flush=True,
            )

    summary = mean_metrics(
        [
            {
                key: float(row[key])
                for key in ("mae_norm", "mae_hu", "psnr", "ssim")
            }
            for row in metrics_rows
        ]
    )
    summary.update(
        {
            "case_count": len(metrics_rows),
            "view_count": args.view_count,
            "eval_grid_size": args.eval_grid_size,
            "backend": "deepsparse",
        }
    )
    manifest = {
        "schema": "x2cbct_setup_error_axis_predictions.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "view_count": args.view_count,
        "view_key": view_key,
        "model": predictions[0]["model_config"] if predictions else {},
        "predictions": predictions,
    }
    _write_json(pred_dir / "prediction_manifest.json", manifest)
    _write_json(pred_dir / "metrics_summary.json", summary)
    with (pred_dir / "metrics_per_case.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_key",
                "pair_key",
                "mae_norm",
                "mae_hu",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    return predictions, metrics_rows


def _write_axis_readme(
    *,
    args: argparse.Namespace,
    axis: str,
    pair: FocusPair,
    row: dict[str, Any],
    viewer_html: Path | None,
) -> None:
    title = "S-I maximum pair" if axis == "si-max" else "yaw maximum pair"
    lines = [
        f"# Setup error experiment: {title}",
        "",
        "## Run identity",
        "",
        f"- model_label: `{args.model_label}`",
        f"- backend: `{args.backend}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- view_count: `{args.view_count}`",
        f"- eval_grid_size: `{args.eval_grid_size}`",
        "- registration_grid_policy: "
        "`endpoint_aligned_preserve_physical_extent`",
        f"- chunk_size: `{args.chunk_size}`",
        f"- pair: `{pair.patient_id}:{pair.reference_case_id}:{pair.moving_case_id}`",
        f"- viewer: `{viewer_html}`" if viewer_html is not None else "- viewer: skipped",
        "",
        "The primary true values below are re-registered on the same evaluation "
        "grid as the model output. The source formal-grid values are retained as "
        "a resolution-control measurement.",
        "",
        "## Setup-error comparison",
        "",
        "| true S-I | reconstructed S-I | abs error S-I | true yaw | reconstructed yaw | abs error yaw |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {float(row['true_setup_error_SI_mm']):.4f} | "
            f"{float(row['pred_setup_error_SI_mm']):.4f} | "
            f"{float(row['abs_delta_SI_mm']):.4f} | "
            f"{float(row['true_setup_error_yaw_deg']):.4f} | "
            f"{float(row['pred_setup_error_yaw_deg']):.4f} | "
            f"{float(row['abs_delta_yaw_deg']):.4f} |"
        ),
        "",
        "## Grid-resolution control",
        "",
        "| source grid | evaluation grid | source S-I | eval S-I | abs shift S-I | source yaw | eval yaw | abs shift yaw | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        (
            f"| {int(row['source_formal_grid_size'])} | "
            f"{int(row['true_registration_grid_size'])} | "
            f"{float(row['source_formal_setup_error_SI_mm']):.4f} | "
            f"{float(row['true_setup_error_SI_mm']):.4f} | "
            f"{float(row['grid_control_abs_delta_SI_mm']):.4f} | "
            f"{float(row['source_formal_setup_error_yaw_deg']):.4f} | "
            f"{float(row['true_setup_error_yaw_deg']):.4f} | "
            f"{float(row['grid_control_abs_delta_yaw_deg']):.4f} | "
            f"{row['grid_control_pass']} |"
        ),
        "",
        "Primary table metrics are the two absolute setup-error differences. "
        "Image-domain MAE is not part of this table.",
    ]
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_axis(
    args: argparse.Namespace,
    *,
    axis: str,
    pair: FocusPair,
    formal_results: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    print(
        f"[axis:start] axis={axis} model={args.model_label} "
        f"pair={pair.patient_id}:{pair.reference_case_id}:{pair.moving_case_id}",
        flush=True,
    )
    if args.backend == "difnet":
        predictions, _ = _run_inference_for_pairs(args, [pair])
    elif args.backend == "deepsparse":
        predictions, _ = _run_deepsparse_inference_for_pairs(args, [pair])
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    (
        true_reg_pairs,
        pred_reg_pairs,
        roi_by_case,
        volume_info,
    ) = _prepare_evaluation_grid_registration_inputs(
        args,
        [pair],
        predictions,
    )
    true_results = _run_registration(
        args,
        true_reg_pairs,
        roi_by_case,
        output_name="registration_true_eval_grid",
        log_label="registration:true-eval-grid",
    )
    true_result_map = _registration_result_map(true_results)
    true_result = true_result_map[
        (pair.patient_id, pair.reference_case_id, pair.moving_case_id)
    ]
    if str(true_result.get("status", "")).lower() != "ok":
        raise ValueError(
            f"evaluation-grid true registration failed for {pair.pair_key}: "
            f"{true_result.get('notes')}"
        )
    source_result = formal_results[
        (pair.patient_id, pair.reference_case_id, pair.moving_case_id)
    ]
    control = _grid_control_row(
        pair,
        source_result=source_result,
        evaluation_result=true_result,
        args=args,
    )
    _write_grid_control_outputs(args.output_dir, [control])
    if args.enforce_grid_control and not control["overall_pass"]:
        raise ValueError(
            f"grid-resolution control failed for {pair.pair_key}: "
            f"abs_SI={control['abs_evaluation_minus_source_SI_mm']:.4f}mm "
            f"(limit {control['SI_tolerance_mm']:.4f}), "
            f"abs_yaw={control['abs_evaluation_minus_source_yaw_deg']:.4f}deg "
            f"(limit {control['yaw_tolerance_deg']:.4f})"
        )

    pred_results = _run_prediction_registration(
        args,
        pred_reg_pairs,
        roi_by_case,
    )
    rows = _comparison_rows(
        [pair],
        predictions,
        pred_results,
        true_result_map,
    )
    rows = _attach_grid_control(rows, [control])
    _write_comparison_outputs(args.output_dir, rows)
    viewer_html = None
    if not args.skip_viewer:
        viewer_html = _export_focus_viewer(
            args.output_dir,
            [pair],
            volume_info,
            rows,
            slice_stride=args.viewer_slice_stride,
        )
    _write_axis_readme(
        args=args,
        axis=axis,
        pair=pair,
        row=rows[0],
        viewer_html=viewer_html,
    )
    summary = {
        "ok": True,
        "axis": axis,
        "model_label": args.model_label,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "pair": _pair_to_dict(pair),
        "comparison": rows[0],
        "grid_resolution_control": control,
        "true_registration_csv": str(
            args.output_dir
            / "setup_error"
            / "registration_true_eval_grid"
            / "registration_results.csv"
        ),
        "grid_resolution_control_csv": str(
            args.output_dir
            / "setup_error"
            / "grid_resolution_control"
            / "grid_resolution_control.csv"
        ),
        "comparison_csv": str(
            args.output_dir / "setup_error" / "setup_error_comparison.csv"
        ),
        "viewer_html": str(viewer_html) if viewer_html is not None else None,
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    print(
        f"[axis:done] axis={axis} "
        f"abs_SI={float(rows[0]['abs_delta_SI_mm']):.4f}mm "
        f"abs_yaw={float(rows[0]['abs_delta_yaw_deg']):.4f}deg "
        f"grid_control_pass={control['overall_pass']}",
        flush=True,
    )
    return summary


def _write_table_ready_outputs(
    output_dir: Path,
    summaries: list[dict[str, Any]],
) -> None:
    rows = []
    for summary in summaries:
        comparison = summary["comparison"]
        rows.append(
            {
                "model_label": summary["model_label"],
                "backend": summary["backend"],
                "checkpoint": summary["checkpoint"],
                "axis_experiment": summary["axis"],
                "patient_id": comparison["patient_id"],
                "reference_case_id": comparison["reference_case_id_C"],
                "moving_case_id": comparison["moving_case_id_B"],
                "registration_grid_size": comparison[
                    "true_registration_grid_size"
                ],
                "true_SI_mm": comparison["true_setup_error_SI_mm"],
                "reconstructed_SI_mm": comparison["pred_setup_error_SI_mm"],
                "abs_error_SI_mm": comparison["abs_delta_SI_mm"],
                "true_yaw_deg": comparison["true_setup_error_yaw_deg"],
                "reconstructed_yaw_deg": comparison["pred_setup_error_yaw_deg"],
                "abs_error_yaw_deg": comparison["abs_delta_yaw_deg"],
                "source_formal_grid_size": comparison[
                    "source_formal_grid_size"
                ],
                "source_formal_SI_mm": comparison[
                    "source_formal_setup_error_SI_mm"
                ],
                "source_formal_yaw_deg": comparison[
                    "source_formal_setup_error_yaw_deg"
                ],
                "grid_control_abs_delta_SI_mm": comparison[
                    "grid_control_abs_delta_SI_mm"
                ],
                "grid_control_abs_delta_yaw_deg": comparison[
                    "grid_control_abs_delta_yaw_deg"
                ],
                "grid_control_pass": comparison["grid_control_pass"],
                "pred_registration_status": comparison["pred_registration_status"],
                "pred_registration_similarity_score": comparison[
                    "pred_registration_similarity_score"
                ],
            }
        )
    with (output_dir / "table_ready_results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "table_ready_results.json",
        {
            "schema": "x2cbct_setup_error_axis_max_table.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows,
        },
    )


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory already exists and is non-empty: {path}. "
            "Choose a new model/epoch-specific output directory."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the maximum-S-I and/or maximum-yaw real-CBCT setup-error "
            "reconstruction experiment for one explicitly selected checkpoint."
        )
    )
    parser.add_argument("--list-checkpoints", type=Path)
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-label")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment", choices=AXIS_CHOICES, default="both")
    parser.add_argument(
        "--si-pair",
        type=_parse_pair_spec,
        help=(
            "Override the S-I pair with PATIENT:REFERENCE:MOVING, or append "
            ":REFERENCE_AUG:MOVING_AUG to select exact cached augmentations."
        ),
    )
    parser.add_argument(
        "--yaw-pair",
        type=_parse_pair_spec,
        help=(
            "Override the yaw pair with PATIENT:REFERENCE:MOVING, or append "
            ":REFERENCE_AUG:MOVING_AUG to select exact cached augmentations."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        help=(
            "X2CBCT noaug cache root. Repeat to search multiple roots. "
            "Explicit values replace the built-in cache-root list."
        ),
    )
    parser.add_argument("--roi-reviewed", type=Path, default=DEFAULT_ROI_REVIEWED)
    parser.add_argument(
        "--formal-registration-results",
        type=Path,
        default=DEFAULT_FORMAL_RESULTS,
    )
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=5)
    parser.add_argument(
        "--eval-grid-size",
        type=int,
        default=128,
        help=(
            "Native cubic reconstruction and registration grid. Source CBCTs "
            "are endpoint-aligned resampled to this size while preserving their "
            "physical field of view."
        ),
    )
    parser.add_argument(
        "--grid-control-si-tolerance-mm",
        type=float,
        default=0.5,
        help="Advisory maximum |C_eval-C_source| for S-I.",
    )
    parser.add_argument(
        "--grid-control-yaw-tolerance-deg",
        type=float,
        default=0.5,
        help="Advisory maximum |C_eval-C_source| for yaw.",
    )
    parser.add_argument(
        "--enforce-grid-control",
        action="store_true",
        help="Fail after writing control outputs if either grid-control tolerance is exceeded.",
    )
    parser.add_argument("--chunk-size", type=int, default=30_000)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--require-inference-split",
        choices=("any", "train", "test", "eval"),
        default="any",
        help="Fail preflight unless the moving reconstruction sample resolves to this split.",
    )
    parser.add_argument("--viewer-slice-stride", type=int, default=4)
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--deepsparse-config", type=Path)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--deepsparse-num-points", type=int)
    parser.add_argument("--deepsparse-lr-res", type=int)
    parser.add_argument("--projection-multiplier", type=float)
    parser.add_argument("--seed", type=int)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        action="store_true",
    )
    precision.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
    )
    parser.set_defaults(mixed_precision=None)
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace) -> None:
    if args.list_checkpoints is not None:
        return
    missing = [
        name
        for name, value in (
            ("--backend", args.backend),
            ("--checkpoint", args.checkpoint),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"required arguments are missing: {', '.join(missing)}")
    if int(args.eval_grid_size) < 2:
        raise ValueError("--eval-grid-size must be >= 2")
    if (
        not np.isfinite(args.grid_control_si_tolerance_mm)
        or float(args.grid_control_si_tolerance_mm) < 0.0
    ):
        raise ValueError(
            "--grid-control-si-tolerance-mm must be a finite non-negative number"
        )
    if (
        not np.isfinite(args.grid_control_yaw_tolerance_deg)
        or float(args.grid_control_yaw_tolerance_deg) < 0.0
    ):
        raise ValueError(
            "--grid-control-yaw-tolerance-deg must be a finite non-negative number"
        )
    if int(args.chunk_size) < 1:
        raise ValueError("--chunk-size must be >= 1")
    if int(args.viewer_slice_stride) < 1:
        raise ValueError("--viewer-slice-stride must be >= 1")


def main() -> int:
    args = _parse_args()
    _validate_cli(args)
    if args.list_checkpoints is not None:
        print(
            json.dumps(
                {
                    "root": str(args.list_checkpoints.resolve()),
                    "checkpoints": _list_checkpoints(args.list_checkpoints),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    args.checkpoint = _project_path(args.checkpoint)
    args.output_dir = _project_path(args.output_dir)
    args.roi_reviewed = _project_path(args.roi_reviewed)
    args.formal_registration_results = _project_path(
        args.formal_registration_results
    )
    args.model_label = args.model_label or args.checkpoint.stem
    cache_roots, missing_default_caches = _resolve_cache_roots(args.cache_root)
    cache_roots = _order_cache_roots_for_split(
        cache_roots,
        args.require_inference_split,
    )
    args.cache_root = cache_roots

    formal_results = _read_formal_results(args.formal_registration_results)
    pairs = _select_pairs(args, formal_results)
    plan = _preflight(
        args,
        pairs,
        formal_results,
        cache_roots,
        missing_default_caches,
    )
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0

    _ensure_empty_output(args.output_dir)
    for axis in pairs:
        _ensure_empty_output(args.output_dir / axis.replace("-", "_"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "resolved_run.json", plan)

    summaries = []
    root_output_dir = args.output_dir
    for axis, pair in pairs.items():
        axis_args = argparse.Namespace(**vars(args))
        axis_args.output_dir = root_output_dir / axis.replace("-", "_")
        axis_args.output_dir.mkdir(parents=True, exist_ok=False)
        summaries.append(
            _run_axis(
                axis_args,
                axis=axis,
                pair=pair,
                formal_results=formal_results,
            )
        )

    _write_table_ready_outputs(root_output_dir, summaries)
    final_summary = {
        "ok": True,
        "schema": "x2cbct_setup_error_axis_max_run.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_label": args.model_label,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "output_dir": str(root_output_dir),
        "experiment_count": len(summaries),
        "runs": summaries,
        "table_ready_csv": str(root_output_dir / "table_ready_results.csv"),
    }
    _write_json(root_output_dir / "run_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
