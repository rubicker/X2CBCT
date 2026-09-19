from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import write_json
from x2cbct_difnet.normalization import PROJECTION_MAX, PROJECTION_MIN, normalize_projection


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _case_key(row: dict[str, Any]) -> str:
    return f"patient{row['patient_id']}_case{row['case_id']}"


def _split_lookup(split_manifest: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in ("train", "test"):
        for row in split_manifest[split]["cases"]:
            out[_case_key(row)] = split
    return out


def _resize_projection(array: np.ndarray, size: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(array, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"projection must be 2D after squeeze, got {arr.shape}")
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    return {
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1.0)),
        "p5": float(np.percentile(finite, 5.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "zero_ratio": float(np.mean(finite <= 0.0)),
        "clip_low_ratio": float(np.mean(finite < PROJECTION_MIN)),
        "clip_high_ratio": float(np.mean(finite > PROJECTION_MAX)),
    }


def _mean_std(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-12)


def _mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(1.4826 * mad, 1e-12)


def _add_zscores(rows: list[dict[str, Any]], field: str, prefix: str) -> None:
    mean, std = _mean_std(rows, field)
    median, mad_scale = _mad([float(row[field]) for row in rows])
    for row in rows:
        value = float(row[field])
        row[f"{prefix}_mean"] = mean
        row[f"{prefix}_std"] = std
        row[f"{prefix}_z"] = (value - mean) / std
        row[f"{prefix}_robust_median"] = median
        row[f"{prefix}_robust_z"] = (value - median) / mad_scale


def _add_group_zscores(rows: list[dict[str, Any]], key: str, field: str, prefix: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    for group_rows in groups.values():
        _add_zscores(group_rows, field, prefix)


def _aggregate(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in key_fields)].append(row)
    out = []
    for key, items in groups.items():
        payload = {field: key[index] for index, field in enumerate(key_fields)}
        values = np.asarray([float(item["norm_mean"]) for item in items], dtype=np.float64)
        raw_values = np.asarray([float(item["raw_mean"]) for item in items], dtype=np.float64)
        payload.update(
            {
                "projection_count": len(items),
                "norm_mean": float(np.mean(values)),
                "norm_mean_min": float(np.min(values)),
                "norm_mean_max": float(np.max(values)),
                "raw_mean": float(np.mean(raw_values)),
                "raw_mean_min": float(np.min(raw_values)),
                "raw_mean_max": float(np.max(raw_values)),
                "clip_high_ratio_mean": float(np.mean([float(item["clip_high_ratio"]) for item in items])),
                "zero_ratio_mean": float(np.mean([float(item["zero_ratio"]) for item in items])),
            }
        )
        out.append(payload)
    return sorted(out, key=lambda item: tuple(item[field] for field in key_fields))


def _mark_outliers(
    rows: list[dict[str, Any]],
    *,
    z_threshold: float,
    robust_z_threshold: float,
    angle_z_threshold: float | None = None,
) -> list[dict[str, Any]]:
    outliers = []
    for row in rows:
        reasons = []
        z = float(row.get("global_z", 0.0))
        rz = float(row.get("global_robust_z", 0.0))
        if z <= -z_threshold:
            reasons.append("dark_global_z")
        if z >= z_threshold:
            reasons.append("bright_global_z")
        if rz <= -robust_z_threshold:
            reasons.append("dark_global_robust_z")
        if rz >= robust_z_threshold:
            reasons.append("bright_global_robust_z")
        if angle_z_threshold is not None:
            az = float(row.get("angle_z", 0.0))
            arz = float(row.get("angle_robust_z", 0.0))
            if az <= -angle_z_threshold:
                reasons.append("dark_angle_z")
            if az >= angle_z_threshold:
                reasons.append("bright_angle_z")
            if arz <= -robust_z_threshold:
                reasons.append("dark_angle_robust_z")
            if arz >= robust_z_threshold:
                reasons.append("bright_angle_robust_z")
        row["outlier_reasons"] = ";".join(dict.fromkeys(reasons))
        if reasons:
            outliers.append(row)
    return outliers


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary(rows: list[dict[str, Any]], field: str = "norm_mean") -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(np.percentile(values, 1.0)),
        "p5": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    fixed = _read_json(args.fixed_manifest)
    split_lookup = _split_lookup(_read_json(args.split_manifest))
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    records = fixed["records"][: args.max_cases] if args.max_cases is not None else fixed["records"]
    for record in records:
        case_key = _case_key(record)
        split = split_lookup.get(case_key, "unknown")
        selected = record["selected"][args.view_key]
        for selected_index, projection in enumerate(selected):
            path = Path(record["case_path"]) / projection["file"]
            base = {
                "split": split,
                "patient_id": str(record["patient_id"]),
                "case_id": str(record["case_id"]),
                "case_key": case_key,
                "view_key": args.view_key,
                "selected_index": selected_index,
                "target_angle_deg": float(projection["target_angle_deg"]),
                "gantry_angle_deg": float(projection["gantry_angle_deg"]),
                "angular_residual_deg": float(projection["angular_residual_deg"]),
                "file": projection["file"],
                "projection_path": str(path.resolve()),
            }
            if not path.exists():
                missing.append({**base, "reason": "missing_projection"})
                continue
            image = sitk.ReadImage(str(path))
            raw = np.squeeze(sitk.GetArrayFromImage(image).astype(np.float32))
            resized = _resize_projection(raw, args.projection_size)
            raw_stats = _stats(raw)
            norm = normalize_projection(resized)
            norm_stats = _stats(norm)
            rows.append(
                {
                    **base,
                    "raw_shape_yx": [int(item) for item in raw.shape],
                    "spacing_xy": [float(item) for item in image.GetSpacing()[:2]],
                    "raw_min": raw_stats["min"],
                    "raw_p1": raw_stats["p1"],
                    "raw_p50": raw_stats["p50"],
                    "raw_p99": raw_stats["p99"],
                    "raw_max": raw_stats["max"],
                    "raw_mean": raw_stats["mean"],
                    "raw_std": raw_stats["std"],
                    "norm_mean": norm_stats["mean"],
                    "norm_p1": norm_stats["p1"],
                    "norm_p50": norm_stats["p50"],
                    "norm_p99": norm_stats["p99"],
                    "norm_max": norm_stats["max"],
                    "norm_std": norm_stats["std"],
                    "zero_ratio": raw_stats["zero_ratio"],
                    "clip_high_ratio": raw_stats["clip_high_ratio"],
                }
            )
    if not rows:
        raise ValueError("no projection rows were readable")

    _add_zscores(rows, "norm_mean", "global")
    _add_group_zscores(rows, "split", "norm_mean", "split")
    _add_group_zscores(rows, "target_angle_deg", "norm_mean", "angle")
    projection_outliers = _mark_outliers(
        rows,
        z_threshold=args.z_threshold,
        robust_z_threshold=args.robust_z_threshold,
        angle_z_threshold=args.angle_z_threshold,
    )

    case_rows = _aggregate(rows, ["split", "patient_id", "case_id", "case_key"])
    _add_zscores(case_rows, "norm_mean", "global")
    _add_group_zscores(case_rows, "split", "norm_mean", "split")
    case_outliers = _mark_outliers(case_rows, z_threshold=args.z_threshold, robust_z_threshold=args.robust_z_threshold)

    patient_rows = _aggregate(rows, ["split", "patient_id"])
    _add_zscores(patient_rows, "norm_mean", "global")
    _add_group_zscores(patient_rows, "split", "norm_mean", "split")
    patient_outliers = _mark_outliers(patient_rows, z_threshold=args.z_threshold, robust_z_threshold=args.robust_z_threshold)

    angle_rows = _aggregate(rows, ["target_angle_deg"])
    report = {
        "schema": "x2cbct_self_projection1208_projection_intensity_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_manifest": str(args.fixed_manifest.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "view_key": args.view_key,
        "projection_size": args.projection_size,
        "normalization": {"rule": "clip(projection, 0, 8.6) / 8.6", "min": PROJECTION_MIN, "max": PROJECTION_MAX},
        "thresholds": {
            "z_threshold": args.z_threshold,
            "robust_z_threshold": args.robust_z_threshold,
            "angle_z_threshold": args.angle_z_threshold,
        },
        "projection_summary_all": _summary(rows),
        "projection_summary_by_split": {
            split: _summary([row for row in rows if row["split"] == split]) for split in sorted({row["split"] for row in rows})
        },
        "case_summary_all": _summary(case_rows),
        "patient_summary_all": _summary(patient_rows),
        "projection_count": len(rows),
        "case_count": len(case_rows),
        "patient_count": len(patient_rows),
        "missing_projection_count": len(missing),
        "missing_projections": missing,
        "projection_outlier_count": len(projection_outliers),
        "case_outlier_count": len(case_outliers),
        "patient_outlier_count": len(patient_outliers),
        "projection_outliers": sorted(projection_outliers, key=lambda row: row["norm_mean"]),
        "case_outliers": sorted(case_outliers, key=lambda row: row["norm_mean"]),
        "patient_outliers": sorted(patient_outliers, key=lambda row: row["norm_mean"]),
        "angle_summary": angle_rows,
    }
    write_json(report, args.output)

    projection_fields = [
        "split",
        "patient_id",
        "case_id",
        "case_key",
        "target_angle_deg",
        "gantry_angle_deg",
        "file",
        "raw_mean",
        "raw_p1",
        "raw_p50",
        "raw_p99",
        "raw_max",
        "norm_mean",
        "norm_p1",
        "norm_p50",
        "norm_p99",
        "norm_max",
        "zero_ratio",
        "clip_high_ratio",
        "global_z",
        "split_z",
        "angle_z",
        "global_robust_z",
        "angle_robust_z",
        "outlier_reasons",
        "projection_path",
    ]
    _write_csv(args.projection_csv, sorted(rows, key=lambda row: row["norm_mean"]), projection_fields)
    _write_csv(args.case_csv, sorted(case_rows, key=lambda row: row["norm_mean"]), list(case_rows[0].keys()))
    _write_csv(args.patient_csv, sorted(patient_rows, key=lambda row: row["norm_mean"]), list(patient_rows[0].keys()))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit self_projection1208 selected projection intensity distribution.")
    parser.add_argument("--fixed-manifest", type=Path, default=PROJECT_ROOT / "data_manifest" / "fixed_view_sampling_v0b.json")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "data_manifest" / "splits_v0b_train_test.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_projection_intensity" / "projection_intensity_audit.json")
    parser.add_argument("--projection-csv", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_projection_intensity" / "projection_intensity_rows.csv")
    parser.add_argument("--case-csv", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_projection_intensity" / "case_projection_intensity.csv")
    parser.add_argument("--patient-csv", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_projection_intensity" / "patient_projection_intensity.csv")
    parser.add_argument("--view-key", choices=("view_2", "view_5", "view_10"), default="view_10")
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--robust-z-threshold", type=float, default=3.0)
    parser.add_argument("--angle-z-threshold", type=float, default=2.0)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    report = build_audit(args)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "projection_csv": str(args.projection_csv.resolve()),
                "case_csv": str(args.case_csv.resolve()),
                "patient_csv": str(args.patient_csv.resolve()),
                "projection_count": report["projection_count"],
                "case_count": report["case_count"],
                "missing_projection_count": report["missing_projection_count"],
                "projection_outlier_count": report["projection_outlier_count"],
                "case_outlier_count": report["case_outlier_count"],
                "patient_outlier_count": report["patient_outlier_count"],
                "projection_summary_all": report["projection_summary_all"],
                "projection_summary_by_split": report["projection_summary_by_split"],
                "dark_case_outliers": report["case_outliers"][:20],
                "dark_patient_outliers": report["patient_outliers"][:20],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
