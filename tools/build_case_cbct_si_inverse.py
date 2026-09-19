from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.metrics import metric_dict
from x2cbct_difnet.setup_augmentation import volume_background_value
from x2cbct_lungmask.si_correction import inverse_si_translation


DEFAULT_CASE_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "output"
    / "capsule_test"
    / "E2-SH-V5"
    / "validation"
    / "epoch_0150_full"
    / "test"
    / "prediction_manifest.json"
)
DEFAULT_REFERENCE_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "output"
    / "dif_net_baseline"
    / "v5_luna16_noaug"
    / "validation"
    / "epoch_0200_full_final"
    / "test"
    / "prediction_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "segmentation_lungmask"
    / "reconstruction_manifests"
    / "case_cbct_view5_si_p10_inverse"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_row_volume(
    value: str,
    *,
    manifest_path: Path,
    volume_dir_name: str,
) -> Path:
    raw = Path(value)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        candidates.append(manifest_path.parent / raw)
    # Older copied manifests can retain the source run directory. The basename
    # is stable, so prefer the volume folder next to the selected manifest.
    candidates.append(manifest_path.parent / volume_dir_name / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"could not resolve volume path {value!r}; tried:\n{rendered}")


def _prediction_rows(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = manifest.get("predictions") or manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no predictions/cases list: {path}")
    return rows


def _mean_std(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size != values.size:
            raise ValueError(f"metric {key} contains non-finite values")
        result[key] = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    return result


def _flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(values[key]) for key in ("mae_norm", "mae_hu", "psnr", "ssim")}


def _wrong_direction_shift(
    volume: np.ndarray,
    *,
    shift_voxels_z: float,
    fill_value: float,
) -> np.ndarray:
    return scipy.ndimage.shift(
        np.asarray(volume, dtype=np.float32),
        shift=(abs(float(shift_voxels_z)), 0.0, 0.0),
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=False,
    ).astype(np.float32)


def build(args: argparse.Namespace) -> dict[str, Any]:
    case_manifest_path = args.case_manifest.resolve()
    reference_manifest_path = args.reference_manifest.resolve()
    output_dir = args.output_dir.resolve()
    generated_dir = output_dir / "generated_cbct"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite to replace arrays: {output_dir}")
    generated_dir.mkdir(parents=True, exist_ok=True)

    case_manifest = _read_json(case_manifest_path)
    reference_manifest = _read_json(reference_manifest_path)
    selected = [
        row
        for row in _prediction_rows(case_manifest, case_manifest_path)
        if str(row.get("augmentation_id") or "") == args.augmentation_id
    ]
    if len(selected) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} rows for {args.augmentation_id}, found {len(selected)}"
        )
    for row in selected:
        if float(row.get("translation_mm", float("nan"))) != float(args.translation_mm):
            raise ValueError(f"unexpected translation for case {row.get('case_id')}: {row.get('translation_mm')}")
        if float(row.get("rotation_deg", float("nan"))) != float(args.rotation_deg):
            raise ValueError(f"unexpected yaw for case {row.get('case_id')}: {row.get('rotation_deg')}")

    references = {
        str(row.get("case_id") or ""): row
        for row in _prediction_rows(reference_manifest, reference_manifest_path)
    }
    selected_ids = [str(row.get("source_case_id") or "") for row in selected]
    if len(set(selected_ids)) != len(selected_ids) or any(not case_id for case_id in selected_ids):
        raise ValueError("selected CaSE-CBCT rows do not map one-to-one to source_case_id")
    missing_reference = sorted(set(selected_ids) - set(references))
    if missing_reference:
        raise ValueError(f"{len(missing_reference)} selected cases are absent from the no-augmentation reference")

    output_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for position, row in enumerate(sorted(selected, key=lambda value: str(value["source_case_id"])), start=1):
        source_case_id = str(row["source_case_id"])
        reference_row = references[source_case_id]
        generated_path = _resolve_row_volume(
            str(row["generated_cbct_path"]),
            manifest_path=case_manifest_path,
            volume_dir_name="generated_cbct",
        )
        augmented_target_path = _resolve_row_volume(
            str(row["target_cbct_path"]),
            manifest_path=case_manifest_path,
            volume_dir_name="target_cbct",
        )
        reference_target_path = _resolve_row_volume(
            str(reference_row["target_cbct_path"]),
            manifest_path=reference_manifest_path,
            volume_dir_name="target_cbct",
        )
        generated = np.load(generated_path).astype(np.float32)
        augmented_target = np.load(augmented_target_path).astype(np.float32)
        reference_target = np.load(reference_target_path).astype(np.float32)
        if generated.shape != augmented_target.shape or generated.shape != reference_target.shape:
            raise ValueError(
                f"shape mismatch for {source_case_id}: generated={generated.shape}, "
                f"augmented_target={augmented_target.shape}, reference={reference_target.shape}"
            )

        source_shape = tuple(int(value) for value in row["processed_shape_zyx"])
        source_spacing = tuple(float(value) for value in row["processed_spacing_xyz"])
        generated_correction = inverse_si_translation(
            generated,
            translation_mm=args.translation_mm,
            source_shape_zyx=source_shape,
            source_spacing_xyz=source_spacing,
        )
        target_correction = inverse_si_translation(
            augmented_target,
            translation_mm=args.translation_mm,
            source_shape_zyx=source_shape,
            source_spacing_xyz=source_spacing,
        )
        wrong_target = _wrong_direction_shift(
            augmented_target,
            shift_voxels_z=generated_correction.shift_voxels_zyx[0],
            fill_value=volume_background_value(augmented_target),
        )

        output_path = generated_dir / f"luna16_{source_case_id}_view_5.npy"
        np.save(output_path, generated_correction.volume)
        pred_before = metric_dict(generated, reference_target, normalization=row["normalization"])
        pred_after = metric_dict(generated_correction.volume, reference_target, normalization=row["normalization"])
        target_before = metric_dict(augmented_target, reference_target, normalization=row["normalization"])
        target_after = metric_dict(target_correction.volume, reference_target, normalization=row["normalization"])
        target_wrong = metric_dict(wrong_target, reference_target, normalization=row["normalization"])

        metric_row = {
            "case_id": source_case_id,
            "source_augmented_case_id": str(row["case_id"]),
            "translation_mm": float(args.translation_mm),
            "rotation_deg": float(args.rotation_deg),
            "effective_spacing_z_mm": float(generated_correction.effective_spacing_xyz[2]),
            "inverse_shift_voxels_z": float(generated_correction.shift_voxels_zyx[0]),
            "prediction_fill_value": float(generated_correction.fill_value),
            **_flatten_metrics("prediction_before", pred_before),
            **_flatten_metrics("prediction_after", pred_after),
            **_flatten_metrics("target_before", target_before),
            **_flatten_metrics("target_after", target_after),
            **_flatten_metrics("target_wrong_direction", target_wrong),
            "prediction_psnr_delta": float(pred_after["psnr"] - pred_before["psnr"]),
            "prediction_ssim_delta": float(pred_after["ssim"] - pred_before["ssim"]),
            "prediction_mae_hu_delta": float(pred_after["mae_hu"] - pred_before["mae_hu"]),
            "target_correct_direction_wins": bool(
                target_after["psnr"] > target_before["psnr"]
                and target_after["psnr"] > target_wrong["psnr"]
            ),
        }
        metric_rows.append(metric_row)

        output_row = dict(row)
        output_row.update(
            {
                "patient_id": f"luna16_{source_case_id}",
                "case_id": source_case_id,
                "case_key": f"luna16_{source_case_id}",
                "augmentation_id": None,
                "translation_mm": 0,
                "rotation_deg": 0,
                "source_patient_id": row.get("patient_id"),
                "source_case_id": source_case_id,
                "source_case_key": row.get("case_key"),
                "generated_cbct_path": _project_relative(output_path),
                "target_cbct_path": _project_relative(reference_target_path),
                "metrics": pred_after,
                "correction": {
                    "schema": "x2cbct_inverse_si_translation.v1",
                    "source_augmentation_id": args.augmentation_id,
                    "source_translation_mm": float(args.translation_mm),
                    "source_rotation_deg": float(args.rotation_deg),
                    "axis": "array axis 0 (Z/S-I)",
                    "inverse_shift_mm_zyx": [-float(args.translation_mm), 0.0, 0.0],
                    "inverse_shift_voxels_zyx": list(generated_correction.shift_voxels_zyx),
                    "effective_spacing_xyz": list(generated_correction.effective_spacing_xyz),
                    "interpolation": "scipy.ndimage.shift order=1 prefilter=False",
                    "boundary_mode": "constant",
                    "fill_value": float(generated_correction.fill_value),
                },
            }
        )
        output_rows.append(output_row)
        print(
            f"corrected {position}/{len(selected)} case_id={source_case_id} "
            f"PSNR {pred_before['psnr']:.4f}->{pred_after['psnr']:.4f}",
            flush=True,
        )

    correct_direction_count = sum(bool(row["target_correct_direction_wins"]) for row in metric_rows)
    if correct_direction_count != len(metric_rows):
        raise RuntimeError(
            f"inverse direction validation failed for {len(metric_rows) - correct_direction_count} cases"
        )

    metric_keys = [
        key
        for key, value in metric_rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and (
            key.startswith("prediction_")
            or key.startswith("target_")
            or key in {"effective_spacing_z_mm", "inverse_shift_voxels_z"}
        )
    ]
    summary = {
        "schema": "x2cbct_case_si_inverse_summary.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(metric_rows),
        "augmentation_id": args.augmentation_id,
        "source_translation_mm": float(args.translation_mm),
        "source_rotation_deg": float(args.rotation_deg),
        "correct_direction_validation_count": correct_direction_count,
        "metrics": _mean_std(metric_rows, metric_keys),
    }
    prediction_manifest = {
        "schema": "x2cbct_case_inverse_si_predictions.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_prediction_manifest": _project_relative(case_manifest_path),
        "reference_prediction_manifest": _project_relative(reference_manifest_path),
        "source_prediction_manifest_sha256": _sha256(case_manifest_path),
        "reference_prediction_manifest_sha256": _sha256(reference_manifest_path),
        "view_count": int(output_rows[0]["view_count"]),
        "view_key": f"view_{int(output_rows[0]['view_count'])}",
        "normalization": output_rows[0]["normalization"],
        "correction": output_rows[0]["correction"],
        "predictions": output_rows,
    }
    _write_json(prediction_manifest, output_dir / "prediction_manifest.json")
    _write_json(summary, output_dir / "metrics_summary.json")
    with (output_dir / "metrics_per_case.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    reproducibility = {
        "schema": "x2cbct_case_si_inverse_reproducibility.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command_arguments": {
            "case_manifest": str(case_manifest_path),
            "reference_manifest": str(reference_manifest_path),
            "augmentation_id": args.augmentation_id,
            "translation_mm": float(args.translation_mm),
            "rotation_deg": float(args.rotation_deg),
            "expected_cases": int(args.expected_cases),
            "output_dir": str(output_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "input_hashes": {
            "case_manifest_sha256": _sha256(case_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
        },
        "output_inventory": {
            "generated_array_count": len(list(generated_dir.glob("*.npy"))),
            "generated_array_shape": list(np.load(next(generated_dir.glob("*.npy")), mmap_mode="r").shape),
            "generated_array_dtype": str(np.load(next(generated_dir.glob("*.npy")), mmap_mode="r").dtype),
        },
    }
    _write_json(reproducibility, output_dir / "reproducibility.json")
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "case_count": len(output_rows),
        "correct_direction_validation_count": correct_direction_count,
        "metrics_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select CaSE-CBCT S-I-only augmented reconstructions and undo their translation."
    )
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--augmentation-id", default="si_p10_rot_0")
    parser.add_argument("--translation-mm", type=float, default=10.0)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--expected-cases", type=int, default=149)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
