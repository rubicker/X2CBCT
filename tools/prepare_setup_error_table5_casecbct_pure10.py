from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_setup_error_focus_experiment import (  # noqa: E402
    _denormalize_cbct,
    _find_case_payload,
    _find_sample_index,
    _write_nifti,
)
from x2cbct_difnet.cached_dataset import load_cache_manifest  # noqa: E402
from x2cbct_difnet.setup_error_registration import (  # noqa: E402
    load_roi_review,
    register_pair,
    write_registration_outputs,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "setup_error"
    / "table5_casecbct_pure10_v5_grid128.json"
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"selection metrics are empty: {path}")
    return rows


def _rank(rows: list[dict[str, str]], row: dict[str, str], field: str) -> int:
    ordered = sorted(rows, key=lambda item: float(item[field]), reverse=True)
    return ordered.index(row) + 1


def _source_case_key(row: dict[str, str]) -> str:
    augmentation_id = str(row["augmentation_id"])
    suffix = f"_{augmentation_id}"
    case_key = str(row["case_key"])
    if not case_key.endswith(suffix):
        raise ValueError(
            f"case_key does not end in its augmentation id: {case_key}, "
            f"{augmentation_id}"
        )
    return case_key[: -len(suffix)]


def _selection_audit(
    config: dict[str, Any],
    metrics_rows: list[dict[str, str]],
    prediction_manifest: dict[str, Any],
    identity_cache_root: Path,
    augmented_cache_root: Path,
) -> list[dict[str, Any]]:
    manifest_split = str(prediction_manifest.get("split", ""))
    if manifest_split != "test":
        raise ValueError(
            f"selection prediction manifest must be split=test, got {manifest_split!r}"
        )

    by_source: dict[str, list[dict[str, str]]] = {}
    for row in metrics_rows:
        by_source.setdefault(_source_case_key(row), []).append(row)
    source_summary = [
        {
            "source_case_key": key,
            "mean_psnr": float(np.mean([float(row["psnr"]) for row in rows])),
            "mean_ssim": float(np.mean([float(row["ssim"]) for row in rows])),
            "variant_count": len(rows),
        }
        for key, rows in by_source.items()
    ]
    source_psnr_order = sorted(
        source_summary, key=lambda item: item["mean_psnr"], reverse=True
    )
    source_ssim_order = sorted(
        source_summary, key=lambda item: item["mean_ssim"], reverse=True
    )

    audits: list[dict[str, Any]] = []
    identity_manifest = load_cache_manifest(identity_cache_root)
    augmented_manifest = load_cache_manifest(augmented_cache_root)
    view_key = f"view_{int(config['view_count'])}"
    for experiment in config["experiments"]:
        patient_id = str(experiment["patient_id"])
        case_id = str(experiment["case_id"])
        augmentation_id = str(experiment["moving_augmentation_id"])
        expected_case_key = f"patient{patient_id}_case{case_id}_{augmentation_id}"
        matches = [
            row for row in metrics_rows if str(row["case_key"]) == expected_case_key
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one metrics row for {expected_case_key}, got {len(matches)}"
            )
        metric = matches[0]
        translation_mm = float(metric["translation_mm"])
        rotation_deg = float(metric["rotation_deg"])
        if (
            translation_mm != float(experiment["translation_mm"])
            or rotation_deg != float(experiment["rotation_deg"])
        ):
            raise ValueError(
                f"metric transform mismatch for {expected_case_key}: "
                f"({translation_mm}, {rotation_deg})"
            )

        same_variant = [
            row
            for row in metrics_rows
            if str(row["augmentation_id"]) == augmentation_id
        ]
        source_key = f"patient{patient_id}_case{case_id}"
        source = next(
            item for item in source_summary if item["source_case_key"] == source_key
        )

        identity_index = _find_sample_index(
            identity_cache_root,
            "test",
            view_key,
            patient_id,
            case_id,
            "identity",
        )
        moving_index = _find_sample_index(
            augmented_cache_root,
            "test",
            view_key,
            patient_id,
            case_id,
            augmentation_id,
        )
        if identity_index is None or moving_index is None:
            raise FileNotFoundError(
                f"held-out cache samples are missing for {expected_case_key}: "
                f"identity_index={identity_index}, moving_index={moving_index}"
            )
        identity_samples = [
            row
            for row in identity_manifest["samples"]
            if row["split"] == "test" and row["view_key"] == view_key
        ]
        moving_samples = [
            row
            for row in augmented_manifest["samples"]
            if row["split"] == "test" and row["view_key"] == view_key
        ]
        identity_sample = identity_samples[identity_index]
        moving_sample = moving_samples[moving_index]

        audits.append(
            {
                "axis": experiment["axis"],
                "patient_id": patient_id,
                "case_id": case_id,
                "source_case_key": source_key,
                "case_key": expected_case_key,
                "augmentation_id": augmentation_id,
                "translation_mm": translation_mm,
                "rotation_deg": rotation_deg,
                "psnr": float(metric["psnr"]),
                "ssim": float(metric["ssim"]),
                "pure_variant_case_count": len(same_variant),
                "pure_variant_psnr_rank": _rank(same_variant, metric, "psnr"),
                "pure_variant_ssim_rank": _rank(same_variant, metric, "ssim"),
                "source_variant_count": int(source["variant_count"]),
                "source_mean_psnr": float(source["mean_psnr"]),
                "source_mean_ssim": float(source["mean_ssim"]),
                "source_mean_psnr_rank": source_psnr_order.index(source) + 1,
                "source_mean_ssim_rank": source_ssim_order.index(source) + 1,
                "identity_split": str(identity_sample["split"]),
                "moving_split": str(moving_sample["split"]),
                "identity_sample_path": str(
                    identity_cache_root / str(identity_sample["sample_path"])
                ),
                "moving_sample_path": str(
                    augmented_cache_root / str(moving_sample["sample_path"])
                ),
                "selection_note": experiment["selection_note"],
            }
        )
    return audits


def _existing_complete(output_root: Path, config: dict[str, Any]) -> bool:
    audit_path = output_root / "selection_audit.json"
    result_path = output_root / "registration_results.csv"
    if not audit_path.is_file() or not result_path.is_file():
        return False
    try:
        audit = _read_json(audit_path)
        with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        audit.get("schema") == config["schema"]
        and len(audit.get("selections", [])) == len(config["experiments"])
        and len(rows) == len(config["experiments"])
        and all(str(row.get("status", "")).lower() == "ok" for row in rows)
    )


def prepare(config_path: Path, *, force: bool = False, itk_threads: int = 4) -> dict[str, Any]:
    config_path = _project_path(config_path)
    config = _read_json(config_path)
    identity_cache_root = _project_path(config["identity_cache_root"])
    augmented_cache_root = _project_path(config["augmented_cache_root"])
    metrics_path = _project_path(config["selection_metrics_csv"])
    prediction_manifest_path = _project_path(
        config["selection_prediction_manifest"]
    )
    roi_path = _project_path(config["roi_reviewed"])
    output_root = _project_path(config["source_registration_output"])

    required = [
        identity_cache_root / "cache_manifest.json",
        augmented_cache_root / "cache_manifest.json",
        metrics_path,
        prediction_manifest_path,
        roi_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required preparation inputs are missing: {missing}")

    if output_root.exists() and any(output_root.iterdir()):
        if not force and _existing_complete(output_root, config):
            return {
                "ok": True,
                "status": "reused",
                "output_root": str(output_root),
                "registration_results": str(
                    output_root / "registration_results.csv"
                ),
            }
        if not force:
            raise FileExistsError(
                f"preparation output is non-empty but incomplete: {output_root}; "
                "use --force only after reviewing that directory"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_rows = _read_metrics(metrics_path)
    prediction_manifest = _read_json(prediction_manifest_path)
    audits = _selection_audit(
        config,
        metrics_rows,
        prediction_manifest,
        identity_cache_root,
        augmented_cache_root,
    )

    cache_roots = [identity_cache_root, augmented_cache_root]
    roi_by_case = load_roi_review(roi_path)
    registration_pairs: list[dict[str, Any]] = []
    volume_rows: list[dict[str, Any]] = []
    for experiment in config["experiments"]:
        patient_id = str(experiment["patient_id"])
        case_id = str(experiment["case_id"])
        moving_augmentation_id = str(experiment["moving_augmentation_id"])
        ref_payload, ref_manifest, ref_cache_manifest = _find_case_payload(
            cache_roots,
            patient_id,
            case_id,
            str(experiment["reference_augmentation_id"]),
        )
        mov_payload, mov_manifest, mov_cache_manifest = _find_case_payload(
            cache_roots,
            patient_id,
            case_id,
            moving_augmentation_id,
        )
        ref_norm = np.load(ref_payload).astype(np.float32)
        mov_norm = np.load(mov_payload).astype(np.float32)
        expected_shape = (int(config["source_grid_size"]),) * 3
        if ref_norm.shape != expected_shape or mov_norm.shape != expected_shape:
            raise ValueError(
                f"source-grid shape mismatch for patient={patient_id}, case={case_id}: "
                f"{ref_norm.shape}, {mov_norm.shape}, expected={expected_shape}"
            )
        case_dir = output_root / "registration_inputs" / str(experiment["axis"])
        ref_nii = _write_nifti(
            _denormalize_cbct(
                ref_norm, ref_cache_manifest["cbct_normalization"]
            ),
            ref_manifest,
            case_dir / "C_identity_cbct_hu.nii.gz",
        )
        mov_nii = _write_nifti(
            _denormalize_cbct(
                mov_norm, mov_cache_manifest["cbct_normalization"]
            ),
            mov_manifest,
            case_dir / "B_augmented_cbct_hu.nii.gz",
        )
        registration_pairs.append(
            {
                "patient_id": patient_id,
                "reference_case_id": case_id,
                "moving_case_id": case_id,
                "reference_cbct_path": str(ref_nii),
                "moving_cbct_path": str(mov_nii),
            }
        )
        volume_rows.append(
            {
                "axis": experiment["axis"],
                "reference_payload": str(ref_payload),
                "moving_payload": str(mov_payload),
                "reference_nifti": str(ref_nii),
                "moving_nifti": str(mov_nii),
                "processed_size_xyz": ref_manifest["processed_size_xyz"],
                "processed_spacing_xyz": ref_manifest["processed_spacing_xyz"],
            }
        )

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(max(1, int(itk_threads)))
    results = []
    initializations: list[dict[str, Any]] = []
    for pair, experiment in zip(registration_pairs, config["experiments"]):
        result, init_rows = register_pair(pair, roi_by_case)
        results.append(result)
        for item in init_rows:
            initializations.append(
                {
                    "axis": experiment["axis"],
                    "augmentation_id": experiment["moving_augmentation_id"],
                    **asdict(item),
                }
            )
        print(
            f"[source-registration] axis={experiment['axis']} "
            f"case=patient{pair['patient_id']}_case{pair['moving_case_id']} "
            f"status={result.status} SI={result.setup_error_SI_mm:.4f}mm "
            f"yaw={result.setup_error_yaw_deg:.4f}deg",
            flush=True,
        )
    write_registration_outputs(output_root, results, initializations)
    if any(result.status != "ok" for result in results):
        raise RuntimeError(
            "one or more source-grid registrations failed; inspect "
            f"{output_root / 'registration_results.csv'}"
        )

    _write_json(
        output_root / "selection_audit.json",
        {
            "schema": config["schema"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(config_path),
            "selection_metrics_csv": str(metrics_path),
            "selection_prediction_manifest": str(prediction_manifest_path),
            "prediction_manifest_split": prediction_manifest.get("split"),
            "selections": audits,
            "registration_inputs": volume_rows,
        },
    )
    return {
        "ok": True,
        "status": "created",
        "output_root": str(output_root),
        "registration_results": str(output_root / "registration_results.csv"),
        "selections": audits,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the two high-quality held-out pure-10 samples and create "
            "their 256-grid identity-to-augmented registration baseline."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--itk-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = prepare(
            args.config,
            force=bool(args.force),
            itk_threads=int(args.itk_threads),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
