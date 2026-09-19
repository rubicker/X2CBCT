from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_difnet_v0b import _model_item, _resolve_device
from view_difnet_v0b_prediction import _export_volume, _write_json
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest
from x2cbct_difnet.cache import read_json, resolve_cache_path
from x2cbct_difnet.capsule_routing import get_capsule_squash_mode, set_capsule_squash_mode
from x2cbct_difnet.metrics import mean_metrics, metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.setup_error_registration import load_roi_review, register_pair, write_registration_outputs
from x2cbct_difnet.v2_hash_model import (
    build_difnet_model,
    model_config,
    resolve_checkpoint_capsule_kwargs,
    resolve_checkpoint_combine,
    resolve_checkpoint_hash_backend,
    resolve_checkpoint_model_version,
    resolve_checkpoint_routed_capsule_squash,
    validate_capsule_kwargs_for_combine,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "output" / "dif_net_baseline" / "v10_real_noaug" / "checkpoints" / "epoch_0300.pt"
DEFAULT_SHARED_BASELINE = PROJECT_ROOT / "runs" / "setup_error" / "model_experiments" / "_shared_baseline"
DEFAULT_CACHE_ROOTS = [
    PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256" / "self_projection1208" / "train_mixed_noaug_256",
    PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256" / "self_projection1208" / "train_patient_noaug_256",
    PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256" / "self_projection1208" / "test_mixed_noaug_256",
    PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256" / "self_projection1208" / "test_patient_noaug_256",
]
DEFAULT_ROI_REVIEWED = DEFAULT_SHARED_BASELINE / "cervical_roi_review" / "roi_reviewed.csv"
DEFAULT_FORMAL_RESULTS = DEFAULT_SHARED_BASELINE / "registration_si_yaw_fixed" / "registration_results.csv"
DEFAULT_PAIRS = [
    ("0", "0", "5"),
    ("0", "0", "7"),
    ("0", "0", "10"),
    ("6", "0", "3"),
    ("24", "2", "7"),
    ("24", "2", "8"),
    ("24", "2", "9"),
    ("24", "2", "13"),
    ("27", "0", "1"),
    ("27", "0", "12"),
]


@dataclass(frozen=True)
class FocusPair:
    patient_id: str
    reference_case_id: str
    moving_case_id: str
    reference_augmentation_id: str | None = None
    moving_augmentation_id: str | None = None

    @staticmethod
    def _case_key(patient_id: str, case_id: str, augmentation_id: str | None) -> str:
        base = f"patient{patient_id}_case{case_id}"
        if augmentation_id in (None, "identity"):
            return base
        return f"{base}_{augmentation_id}"

    @property
    def reference_case_key(self) -> str:
        return self._case_key(
            self.patient_id,
            self.reference_case_id,
            self.reference_augmentation_id,
        )

    @property
    def moving_case_key(self) -> str:
        return self._case_key(
            self.patient_id,
            self.moving_case_id,
            self.moving_augmentation_id,
        )

    @property
    def pair_key(self) -> str:
        moving = f"case{self.moving_case_id}"
        if self.moving_augmentation_id not in (None, "identity"):
            moving = f"{moving}_{self.moving_augmentation_id}"
        reference = f"case{self.reference_case_id}"
        if self.reference_augmentation_id not in (None, "identity"):
            reference = f"{reference}_{self.reference_augmentation_id}"
        return f"patient{self.patient_id}_{moving}_vs_{reference}"

    @property
    def pred_case_id(self) -> str:
        suffix = (
            f"_{self.moving_augmentation_id}"
            if self.moving_augmentation_id not in (None, "identity")
            else ""
        )
        return f"{self.moving_case_id}{suffix}_bhat"


def _load_pairs(path: Path | None) -> list[FocusPair]:
    if path is None:
        return [FocusPair(*item) for item in DEFAULT_PAIRS]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        FocusPair(
            str(row["patient_id"]),
            str(row["reference_case_id"]),
            str(row["moving_case_id"]),
            str(row["reference_augmentation_id"])
            if row.get("reference_augmentation_id")
            else None,
            str(row["moving_augmentation_id"])
            if row.get("moving_augmentation_id")
            else None,
        )
        for row in rows
    ]


def _read_formal_results(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (str(row["patient_id"]), str(row["reference_case_id"]), str(row["moving_case_id"])): row
            for row in csv.DictReader(handle)
        }


def _sample_augmentation_id(sample: dict[str, Any]) -> str:
    setup = sample.get("setup_augmentation")
    if isinstance(setup, dict):
        value = setup.get("augmentation_id")
        if value not in (None, ""):
            return str(value)
    value = sample.get("augmentation_id")
    if value not in (None, ""):
        return str(value)
    return "identity"


def _find_sample_index(
    cache_root: Path,
    split: str,
    view_key: str,
    patient_id: str,
    case_id: str,
    augmentation_id: str | None = None,
) -> int | None:
    manifest = load_cache_manifest(cache_root)
    for index, sample in enumerate(item for item in manifest["samples"] if item["split"] == split and item["view_key"] == view_key):
        if (
            str(sample.get("patient_id")) == str(patient_id)
            and str(sample.get("case_id")) == str(case_id)
            and (
                augmentation_id is None
                or _sample_augmentation_id(sample) == str(augmentation_id)
            )
        ):
            return index
    return None


def _candidate_splits(cache_root: Path) -> list[str]:
    manifest = load_cache_manifest(cache_root)
    return sorted({str(item["split"]) for item in manifest["samples"]})


def _find_dataset_for_case(
    cache_roots: list[Path],
    view_key: str,
    patient_id: str,
    case_id: str,
    eval_grid_size: int,
    *,
    augmentation_id: str | None = None,
    c2rv_grids: bool = False,
    c2rv_grid_scales: tuple[int, ...] = (16, 8, 4),
) -> tuple[CachedDifNetV0bDataset, int]:
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
            dataset = CachedDifNetV0bDataset(
                cache_root,
                split=split,
                view_key=view_key,
                deterministic=True,
                eval_grid_size=eval_grid_size,
                c2rv_grids=c2rv_grids,
                c2rv_grid_scales=c2rv_grid_scales,
            )
            return dataset, index
    raise FileNotFoundError(
        f"no inference sample for patient={patient_id}, case={case_id}, "
        f"augmentation={augmentation_id}, view_key={view_key}"
    )


def _find_case_payload(
    cache_roots: list[Path],
    patient_id: str,
    case_id: str,
    augmentation_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    for cache_root in cache_roots:
        manifest = load_cache_manifest(cache_root)
        for sample in manifest["samples"]:
            if str(sample.get("patient_id")) != str(patient_id) or str(sample.get("case_id")) != str(case_id):
                continue
            if (
                augmentation_id is not None
                and _sample_augmentation_id(sample) != str(augmentation_id)
            ):
                continue
            case_manifest = read_json(resolve_cache_path(cache_root, str(sample["case_manifest_path"])))
            payload = resolve_cache_path(cache_root, str(sample["cbct_payload_path"]))
            return payload, case_manifest, manifest
    raise FileNotFoundError(
        f"no CBCT payload for patient={patient_id}, case={case_id}, "
        f"augmentation={augmentation_id}"
    )


def _denormalize_cbct(array: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    if normalization.get("mode") != "fixed":
        raise ValueError(f"unsupported CBCT normalization for HU export: {normalization}")
    hu_min = float(normalization["hu_min"])
    hu_span = float(normalization["hu_span"])
    return (np.asarray(array, dtype=np.float32) * hu_span + hu_min).astype(np.float32)


def _write_nifti(array: np.ndarray, case_manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(tuple(float(v) for v in case_manifest["processed_spacing_xyz"]))
    image.SetOrigin(tuple(float(v) for v in case_manifest["processed_origin_xyz"]))
    image.SetDirection(tuple(float(v) for v in case_manifest["processed_direction"]))
    sitk.WriteImage(image, str(path))
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object is not JSON serializable: {type(value)}")


def _build_model(checkpoint: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_version = resolve_checkpoint_model_version(checkpoint, requested="auto")
    combine = resolve_checkpoint_combine(checkpoint, requested="auto")
    routed_capsule_squash = resolve_checkpoint_routed_capsule_squash(checkpoint, requested="auto")
    set_capsule_squash_mode(routed_capsule_squash)
    hashgrid_backend = resolve_checkpoint_hash_backend(checkpoint, requested="auto")
    capsule_kwargs = resolve_checkpoint_capsule_kwargs(
        checkpoint,
        {
            "capsule_hidden_dim": None,
            "capsule_layers": None,
            "capsule_heads": None,
            "capsule_mlp_ratio": None,
            "capsule_dropout": None,
            "capsule_valid_mask": None,
        },
    )
    validate_capsule_kwargs_for_combine(combine, **capsule_kwargs)
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=model_version,
        num_views=args.view_count,
        combine=combine,
        mid_channels=args.mid_channels,
        hashgrid_backend=hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
        **capsule_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    resolved = model_config(
        model,
        {
            "model_version": model_version,
            "num_views": args.view_count,
            "combine": combine,
            "mid_channels": args.mid_channels,
            "routed_capsule_squash": get_capsule_squash_mode(),
        },
    )
    return model, {
        "model_version": model_version,
        "combine": combine,
        "hashgrid_backend": hashgrid_backend,
        "routed_capsule_squash": get_capsule_squash_mode(),
        "capsule_kwargs": capsule_kwargs,
        "model_config": resolved,
    }


def _run_inference_for_pairs(args: argparse.Namespace, pairs: list[FocusPair]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, resolved_model = _build_model(checkpoint, args, device)
    cache_roots = [path.resolve() for path in args.cache_root]
    view_key = view_key_from_count(args.view_count)
    c2rv_grids = resolved_model["combine"] == "c2rv" or resolved_model["model_version"] == "c2rv-style"
    c2rv_config = resolved_model["model_config"].get("c2rv", {})
    c2rv_grid_scales = tuple(int(item) for item in c2rv_config.get("grid_scales", (16, 8, 4)))
    predictions: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    pred_dir = args.output_dir / "predictions"
    generated_dir = pred_dir / "generated_cbct"
    target_dir = pred_dir / "target_cbct"

    with torch.no_grad():
        for pair in pairs:
            dataset, index = _find_dataset_for_case(
                cache_roots,
                view_key,
                pair.patient_id,
                pair.moving_case_id,
                args.eval_grid_size,
                augmentation_id=pair.moving_augmentation_id,
                c2rv_grids=c2rv_grids,
                c2rv_grid_scales=c2rv_grid_scales,
            )
            item = dataset[index]
            batch = _model_item(item, device)
            pred = model(batch, is_eval=True, eval_npoint=args.chunk_size)
            generated = pred[0, 0].detach().cpu().numpy().reshape(item["image"].shape).astype(np.float32)
            generated = np.clip(generated, 0.0, 1.0)
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
                "model_version": resolved_model["model_version"],
                "model_config": resolved_model["model_config"],
                "eval_grid_size": args.eval_grid_size,
                "metrics": metrics,
            }
            predictions.append(row)
            metrics_rows.append({"case_key": case_key, "pair_key": pair.pair_key, **metrics})
            print(f"[inference] {pair.pair_key} cache={trace['cache_version']} split={trace['split']} psnr={metrics['psnr']:.3f}", flush=True)

    summary = mean_metrics([{key: float(row[key]) for key in ("mae_norm", "mae_hu", "psnr", "ssim")} for row in metrics_rows])
    summary.update({"case_count": len(metrics_rows), "view_count": args.view_count, "eval_grid_size": args.eval_grid_size})
    manifest = {
        "schema": "x2cbct_setup_error_focus_predictions.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "view_count": args.view_count,
        "view_key": view_key,
        "model": resolved_model,
        "predictions": predictions,
    }
    _write_json(manifest, pred_dir / "prediction_manifest.json")
    _write_json(summary, pred_dir / "metrics_summary.json")
    with (pred_dir / "metrics_per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_key", "pair_key", "mae_norm", "mae_hu", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    return predictions, metrics_rows


def _prepare_registration_inputs(
    args: argparse.Namespace,
    pairs: list[FocusPair],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    cache_roots = [path.resolve() for path in args.cache_root]
    roi_by_case = load_roi_review(args.roi_reviewed)
    pred_by_pair = {str(row["pair_key"]): row for row in predictions}
    volume_info: dict[str, dict[str, Any]] = {}
    reg_pairs: list[dict[str, Any]] = []
    volume_root = args.output_dir / "setup_error" / "registration_inputs"

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
        ref_norm = np.load(ref_payload).astype(np.float32)
        mov_norm = np.load(mov_payload).astype(np.float32)
        ref_hu = _denormalize_cbct(ref_norm, ref_cache_manifest["cbct_normalization"])
        mov_hu = _denormalize_cbct(mov_norm, mov_cache_manifest["cbct_normalization"])
        pred_hu = _denormalize_cbct(pred_norm, pred_row["normalization"])
        ref_nii = _write_nifti(ref_hu, ref_manifest, volume_root / pair.pair_key / "C_reference_cbct_hu.nii.gz")
        mov_nii = _write_nifti(mov_hu, mov_manifest, volume_root / pair.pair_key / "B_treatment_cbct_hu.nii.gz")
        pred_nii = _write_nifti(pred_hu, mov_manifest, volume_root / pair.pair_key / "B_hat_reconstruction_hu.nii.gz")
        pred_roi = dict(roi_by_case[(pair.patient_id, pair.moving_case_id)])
        pred_roi["case_id"] = pair.pred_case_id
        pred_roi["cbct_path"] = str(pred_nii)
        roi_by_case[(pair.patient_id, pair.pred_case_id)] = pred_roi
        reg_pairs.append(
            {
                "patient_id": pair.patient_id,
                "reference_case_id": pair.reference_case_id,
                "moving_case_id": pair.pred_case_id,
                "reference_cbct_path": str(ref_nii),
                "moving_cbct_path": str(pred_nii),
            }
        )
        volume_info[pair.pair_key] = {
            "reference_norm_path": str(ref_payload),
            "moving_norm_path": str(mov_payload),
            "prediction_norm_path": pred_row["generated_cbct_path"],
            "reference_hu_nifti": str(ref_nii),
            "moving_hu_nifti": str(mov_nii),
            "prediction_hu_nifti": str(pred_nii),
            "processed_spacing_xyz": pred_row.get("processed_spacing_xyz") or mov_manifest["processed_spacing_xyz"],
        }
    return reg_pairs, roi_by_case, volume_info


def _run_registration(
    args: argparse.Namespace,
    reg_pairs: list[dict[str, Any]],
    roi_by_case: dict[tuple[str, str], dict[str, Any]],
    *,
    output_name: str,
    log_label: str,
) -> list[dict[str, Any]]:
    results = []
    init_rows: list[dict[str, Any]] = []
    for pair in reg_pairs:
        result, init = register_pair(pair, roi_by_case)
        results.append(result)
        for row in init:
            payload = asdict(row)
            payload.update(
                {
                    "patient_id": pair["patient_id"],
                    "reference_case_id": pair["reference_case_id"],
                    "moving_case_id": pair["moving_case_id"],
                }
            )
            init_rows.append(payload)
        print(
            f"[{log_label}] patient={pair['patient_id']} "
            f"{pair['reference_case_id']}->{pair['moving_case_id']} "
            f"status={result.status} SI={result.setup_error_SI_mm:.3f} yaw={result.setup_error_yaw_deg:.3f}",
            flush=True,
        )
    write_registration_outputs(
        args.output_dir / "setup_error" / output_name,
        results,
        init_rows,
    )
    return [asdict(row) for row in results]


def _run_prediction_registration(
    args: argparse.Namespace,
    reg_pairs: list[dict[str, Any]],
    roi_by_case: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return _run_registration(
        args,
        reg_pairs,
        roi_by_case,
        output_name="registration_pred",
        log_label="registration:pred",
    )


def _comparison_rows(
    pairs: list[FocusPair],
    predictions: list[dict[str, Any]],
    pred_results: list[dict[str, Any]],
    formal_results: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    pred_by_case = {str(row["pair_key"]): row for row in predictions}
    pred_reg_by_key = {
        (str(row["patient_id"]), str(row["reference_case_id"]), str(row["moving_case_id"])): row
        for row in pred_results
    }
    rows = []
    for pair in pairs:
        formal = formal_results[(pair.patient_id, pair.reference_case_id, pair.moving_case_id)]
        pred = pred_reg_by_key[(pair.patient_id, pair.reference_case_id, pair.pred_case_id)]
        pred_case = pred_by_case[pair.pair_key]
        true_si = float(formal["setup_error_SI_mm"])
        true_yaw = float(formal["setup_error_yaw_deg"])
        pred_si = float(pred["setup_error_SI_mm"])
        pred_yaw = float(pred["setup_error_yaw_deg"])
        rows.append(
            {
                "pair_key": pair.pair_key,
                "patient_id": pair.patient_id,
                "reference_case_id_C": pair.reference_case_id,
                "moving_case_id_B": pair.moving_case_id,
                "reconstruction_case_id_B_hat": pair.pred_case_id,
                "true_setup_error_SI_mm": true_si,
                "true_setup_error_yaw_deg": true_yaw,
                "pred_setup_error_SI_mm": pred_si,
                "pred_setup_error_yaw_deg": pred_yaw,
                "delta_SI_mm": pred_si - true_si,
                "delta_yaw_deg": pred_yaw - true_yaw,
                "abs_delta_SI_mm": abs(pred_si - true_si),
                "abs_delta_yaw_deg": abs(pred_yaw - true_yaw),
                "SI_direction_match": (true_si == 0.0 and pred_si == 0.0) or (true_si * pred_si > 0.0),
                "yaw_direction_match": (true_yaw == 0.0 and pred_yaw == 0.0) or (true_yaw * pred_yaw > 0.0),
                "formal_registration_status": formal["status"],
                "pred_registration_status": pred["status"],
                "pred_registration_similarity_score": pred["similarity_score"],
                "inference_cache_version": pred_case["inference_cache_version"],
                "inference_split": pred_case["inference_split"],
                "generated_cbct_path": pred_case["generated_cbct_path"],
                "target_cbct_path": pred_case["target_cbct_path"],
            }
        )
    return rows


def _write_comparison_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    setup_dir = output_dir / "setup_error"
    setup_dir.mkdir(parents=True, exist_ok=True)
    csv_path = setup_dir / "setup_error_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "x2cbct_setup_error_focus_comparison.v1",
        "case_count": len(rows),
        "mean_abs_delta_SI_mm": float(np.mean([float(row["abs_delta_SI_mm"]) for row in rows])),
        "mean_abs_delta_yaw_deg": float(np.mean([float(row["abs_delta_yaw_deg"]) for row in rows])),
        "SI_direction_match_count": int(sum(1 for row in rows if row["SI_direction_match"])),
        "yaw_direction_match_count": int(sum(1 for row in rows if row["yaw_direction_match"])),
        "rows": rows,
    }
    _write_json(summary, setup_dir / "setup_error_comparison.json")


def _viewer_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Setup Error Focus Viewer</title>
  <style>
    :root { --bg:#f4f6f5; --panel:#fff; --ink:#17201c; --muted:#617069; --line:#d9dfdc; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Segoe UI, Arial, sans-serif; letter-spacing:0; background:var(--bg); color:var(--ink); }
    header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; justify-content:space-between; gap:14px; }
    h1 { margin:0; font-size:20px; }
    main { padding:12px; }
    .toolbar, .viewer, .metrics { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .toolbar { display:grid; grid-template-columns:90px minmax(280px,1fr) 90px minmax(140px,.35fr) 70px minmax(140px,.35fr); gap:10px; align-items:center; margin-bottom:12px; }
    select, input[type=range] { width:100%; height:32px; }
    label, .muted { color:var(--muted); font-size:13px; }
    .viewer { display:grid; grid-template-columns:86px repeat(3,minmax(0,1fr)); gap:10px; align-items:stretch; }
    .head { color:var(--muted); font-size:13px; font-weight:700; text-transform:uppercase; }
    .rowLabel { display:grid; align-items:center; color:var(--muted); font-size:14px; font-weight:700; }
    .frame { background:#08100d; border-radius:6px; aspect-ratio:1/1; display:grid; place-items:center; overflow:hidden; }
    img { width:100%; height:100%; object-fit:contain; image-rendering:pixelated; filter:brightness(var(--brightness,1)) contrast(var(--contrast,1)); transform:scale(var(--zoom,1)); }
    .metrics { margin-top:12px; display:grid; grid-template-columns:210px 1fr; gap:7px 12px; font-size:13px; }
    code { overflow-wrap:anywhere; }
    @media(max-width:980px){ header,.toolbar{display:block}.viewer{grid-template-columns:70px repeat(3,minmax(120px,1fr));overflow-x:auto}.toolbar>*{margin:6px 0} }
  </style>
</head>
<body>
  <header><div><h1>Setup Error Focus Viewer</h1><div class="muted" id="caseTitle"></div></div><code id="runTitle"></code></header>
  <main>
    <div class="toolbar">
      <label>Case</label><select id="caseSelect"></select>
      <label>Brightness</label><input id="brightness" type="range" min="50" max="200" value="100">
      <label>Contrast</label><input id="contrast" type="range" min="50" max="220" value="100">
      <label>Zoom</label><input id="zoom" type="range" min="100" max="300" value="100">
    </div>
    <div class="viewer">
      <div></div><div class="head">Axial</div><div class="head">Coronal</div><div class="head">Sagittal</div>
      <div class="rowLabel">C</div><div class="frame"><img id="reference_axial"></div><div class="frame"><img id="reference_coronal"></div><div class="frame"><img id="reference_sagittal"></div>
      <div class="rowLabel">B</div><div class="frame"><img id="moving_axial"></div><div class="frame"><img id="moving_coronal"></div><div class="frame"><img id="moving_sagittal"></div>
      <div class="rowLabel">B_hat</div><div class="frame"><img id="prediction_axial"></div><div class="frame"><img id="prediction_coronal"></div><div class="frame"><img id="prediction_sagittal"></div>
      <div class="rowLabel">slice</div>
      <div><input id="axialSlider" type="range" min="0"><div class="muted" id="axialMeta"></div></div>
      <div><input id="coronalSlider" type="range" min="0"><div class="muted" id="coronalMeta"></div></div>
      <div><input id="sagittalSlider" type="range" min="0"><div class="muted" id="sagittalMeta"></div></div>
    </div>
    <div class="metrics" id="metrics"></div>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const DATA = window.VIEWER_DATA, cases = DATA.cases, caseSelect = document.getElementById("caseSelect");
    const planes = ["axial","coronal","sagittal"], volumes = ["reference","moving","prediction"];
    function text(v){ return typeof v === "object" ? JSON.stringify(v) : String(v ?? "-"); }
    function current(){ return cases[Number(caseSelect.value || 0)]; }
    function nearest(slices, center){ let idx=0, dist=Infinity; slices.forEach((s,i)=>{ const d=Math.abs(s.index-center); if(d<dist){idx=i;dist=d;} }); return idx; }
    function renderPlane(plane){
      const item=current(), slider=document.getElementById(`${plane}Slider`);
      const slices=item.volumes.reference.planes[plane].slices;
      slider.max=Math.max(slices.length-1,0);
      const idx=Math.min(Number(slider.value||0), slices.length-1);
      slider.value=idx;
      for(const v of volumes) document.getElementById(`${v}_${plane}`).src=item.volumes[v].planes[plane].slices[idx].png;
      document.getElementById(`${plane}Meta`).textContent=`${slices[idx].index} (${idx+1} / ${slices.length})`;
    }
    function render(){
      const item=current();
      document.getElementById("caseTitle").textContent=`${item.pair_key}: B=${item.moving_case_key}, C=${item.reference_case_key}`;
      document.getElementById("runTitle").textContent=DATA.output_dir;
      for(const plane of planes){
        const slider=document.getElementById(`${plane}Slider`);
        slider.value=nearest(item.volumes.reference.planes[plane].slices, item.volumes.reference.planes[plane].center_index);
        renderPlane(plane);
      }
      document.getElementById("metrics").innerHTML=Object.entries(item.metrics).map(([k,v])=>`<div>${k}</div><div><code>${text(v)}</code></div>`).join("");
    }
    function filters(){
      document.documentElement.style.setProperty("--brightness", Number(document.getElementById("brightness").value)/100);
      document.documentElement.style.setProperty("--contrast", Number(document.getElementById("contrast").value)/100);
      document.documentElement.style.setProperty("--zoom", Number(document.getElementById("zoom").value)/100);
    }
    cases.forEach((item,i)=>caseSelect.add(new Option(`${item.pair_key} ΔSI=${Number(item.metrics.delta_SI_mm).toFixed(2)} Δyaw=${Number(item.metrics.delta_yaw_deg).toFixed(2)}`, String(i))));
    caseSelect.addEventListener("change", render);
    for(const plane of planes) document.getElementById(`${plane}Slider`).addEventListener("input",()=>renderPlane(plane));
    for(const id of ["brightness","contrast","zoom"]) document.getElementById(id).addEventListener("input", filters);
    filters(); render();
  </script>
</body>
</html>
"""


def _export_focus_viewer(
    output_dir: Path,
    pairs: list[FocusPair],
    volume_info: dict[str, dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    *,
    slice_stride: int,
) -> Path:
    viewer_dir = output_dir / "viewers" / "focus_triplane"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    comparison_by_pair = {str(row["pair_key"]): row for row in comparison_rows}
    cases = []
    for pair in pairs:
        info = volume_info[pair.pair_key]
        spacing = [float(item) for item in info["processed_spacing_xyz"]]
        case_dir = viewer_dir / "assets" / pair.pair_key
        reference = np.load(info["reference_norm_path"]).astype(np.float32)
        moving = np.load(info["moving_norm_path"]).astype(np.float32)
        prediction = np.load(info["prediction_norm_path"]).astype(np.float32)
        cases.append(
            {
                "pair_key": pair.pair_key,
                "patient_id": pair.patient_id,
                "reference_case_key": pair.reference_case_key,
                "moving_case_key": pair.moving_case_key,
                "prediction_case_key": f"{pair.moving_case_key}_B_hat",
                "metrics": comparison_by_pair[pair.pair_key],
                "volumes": {
                    "reference": _export_volume(reference, "C_reference", case_dir, viewer_dir, slice_stride, spacing),
                    "moving": _export_volume(moving, "B_treatment", case_dir, viewer_dir, slice_stride, spacing),
                    "prediction": _export_volume(prediction, "B_hat_reconstruction", case_dir, viewer_dir, slice_stride, spacing),
                },
            }
        )
    viewer_data = {
        "schema": "x2cbct_setup_error_focus_viewer.v1",
        "output_dir": str(output_dir.resolve()),
        "display_order": ["C_reference", "B_treatment", "B_hat_reconstruction"],
        "slice_stride": slice_stride,
        "case_count": len(cases),
        "cases": cases,
    }
    _write_json(viewer_data, viewer_dir / "viewer_data.json")
    (viewer_dir / "viewer_data.js").write_text("window.VIEWER_DATA = " + json.dumps(viewer_data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    (viewer_dir / "index.html").write_text(_viewer_html(), encoding="utf-8")
    return viewer_dir / "index.html"


def _write_run_readme(args: argparse.Namespace, rows: list[dict[str, Any]], viewer_html: Path) -> None:
    lines = [
        "# Setup error focus experiment output",
        "",
        "本目录由 `tools/run_setup_error_focus_experiment.py` 生成。",
        "",
        "## 核心文件",
        "",
        "- `predictions/prediction_manifest.json`: 10 个治疗日样本 B 的模型推理清单。",
        "- `predictions/generated_cbct/`: B 对应的重建结果 B_hat。",
        "- `setup_error/setup_error_comparison.csv`: B->C 正式 setup error 与 B_hat->C 新配准结果的对照表。",
        "- `setup_error/registration_pred/`: B_hat->C 的注册明细。",
        "- `viewers/focus_triplane/index.html`: 三排离线交互式 viewer。",
        "",
        "## 固定协议",
        "",
        "- C: 基准病例真实 CBCT。",
        "- B: 真实治疗日 CBCT。",
        "- B_hat: B 的投影经模型重建出的 CBCT。",
        "- 真实 setup error 读取 `runs/setup_error/model_experiments/_shared_baseline/registration_si_yaw_fixed/registration_results.csv`。",
        "- 模型 setup error 通过现有 setup error 注册函数计算 B_hat->C。",
        "",
        "## 本次运行",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- view_count: `{args.view_count}`",
        f"- eval_grid_size: `{args.eval_grid_size}`",
        f"- viewer: `{viewer_html}`",
        "",
        "## 病例列表",
        "",
        "| pair | true S-I | pred S-I | delta S-I | true yaw | pred yaw | delta yaw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_key']} | {float(row['true_setup_error_SI_mm']):.3f} | "
            f"{float(row['pred_setup_error_SI_mm']):.3f} | {float(row['delta_SI_mm']):.3f} | "
            f"{float(row['true_setup_error_yaw_deg']):.3f} | {float(row['pred_setup_error_yaw_deg']):.3f} | "
            f"{float(row['delta_yaw_deg']):.3f} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self_projection1208 focus setup-error experiment.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", default=list(DEFAULT_CACHE_ROOTS))
    parser.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--roi-reviewed", type=Path, default=DEFAULT_ROI_REVIEWED)
    parser.add_argument("--formal-registration-results", type=Path, default=DEFAULT_FORMAL_RESULTS)
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument("--eval-grid-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--viewer-slice-stride", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.cache_root = [path.resolve() for path in args.cache_root]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_pairs(args.pairs_csv)
    formal_results = _read_formal_results(args.formal_registration_results)
    predictions, _ = _run_inference_for_pairs(args, pairs)
    reg_pairs, roi_by_case, volume_info = _prepare_registration_inputs(args, pairs, predictions)
    pred_results = _run_prediction_registration(args, reg_pairs, roi_by_case)
    rows = _comparison_rows(pairs, predictions, pred_results, formal_results)
    _write_comparison_outputs(args.output_dir, rows)
    viewer_html = _export_focus_viewer(args.output_dir, pairs, volume_info, rows, slice_stride=args.viewer_slice_stride)
    _write_run_readme(args, rows, viewer_html)
    summary = {
        "ok": True,
        "output_dir": str(args.output_dir),
        "comparison_csv": str(args.output_dir / "setup_error" / "setup_error_comparison.csv"),
        "viewer_html": str(viewer_html),
        "case_count": len(rows),
    }
    _write_json(summary, args.output_dir / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
