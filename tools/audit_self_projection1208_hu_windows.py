from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import write_json

PERCENTILES = (0.1, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5, 99.9)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _case_key(row: dict[str, Any]) -> str:
    return f"patient{row['patient_id']}_case{row['case_id']}"


def _split_lookup(split_manifest: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ("train", "test"):
        for row in split_manifest[split]["cases"]:
            lookup[_case_key(row)] = split
    return lookup


def _resize_to_cube(array: np.ndarray, target_size: int | None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if target_size is None or arr.shape == (target_size, target_size, target_size):
        return arr
    factors = [target_size / dim for dim in arr.shape]
    return scipy.ndimage.zoom(arr, factors, order=1, prefilter=False).astype(np.float32)


def _sample(values: np.ndarray, max_voxels: int, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= max_voxels:
        return arr
    rng = np.random.default_rng(seed)
    indices = rng.choice(arr.size, size=max_voxels, replace=False)
    return arr[indices]


def _iqr_flags(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    iqr = q3 - q1
    if iqr <= 1e-8:
        return float(q1), float(q3)
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _stats_for_case(
    cbct_path: Path,
    *,
    target_size: int | None,
    max_voxels: int,
    fixed_min: float,
    fixed_max: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    image = sitk.ReadImage(str(cbct_path))
    raw = sitk.GetArrayFromImage(image).astype(np.float32)
    processed = _resize_to_cube(raw, target_size)
    values = _sample(processed[np.isfinite(processed)], max_voxels, seed)
    percentiles = np.percentile(values, PERCENTILES)
    stats = {
        "raw_shape_zyx": [int(item) for item in raw.shape],
        "spacing_xyz": [float(item) for item in image.GetSpacing()],
        "target_size": target_size,
        "sampled_voxel_count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(PERCENTILES, percentiles)},
        "fixed_window": {
            "hu_min": float(fixed_min),
            "hu_max": float(fixed_max),
            "clip_low_ratio": float(np.mean(values < fixed_min)),
            "clip_high_ratio": float(np.mean(values > fixed_max)),
            "clip_total_ratio": float(np.mean((values < fixed_min) | (values > fixed_max))),
        },
    }
    return stats, values


def _summarize_cases(cases: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = cases if split == "all" else [case for case in cases if case["split"] == split]
    if not selected:
        return {"case_count": 0}
    keys = [
        "p0_5",
        "p99_5",
        "span_p0_5_p99_5",
        "clip_low_ratio",
        "clip_high_ratio",
        "clip_total_ratio",
    ]
    summary: dict[str, Any] = {"case_count": len(selected)}
    for key in keys:
        values = [float(case[key]) for case in selected]
        summary[key] = {
            "min": float(np.min(values)),
            "p25": float(np.percentile(values, 25.0)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75.0)),
            "max": float(np.max(values)),
        }
    return summary


def _group_by_patient(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["patient_id"]), []).append(case)
    rows = []
    for patient_id, patient_cases in sorted(grouped.items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "patient_id": patient_id,
                "case_count": len(patient_cases),
                "splits": sorted({case["split"] for case in patient_cases}),
                "p0_5_median": float(np.median([case["p0_5"] for case in patient_cases])),
                "p99_5_median": float(np.median([case["p99_5"] for case in patient_cases])),
                "span_median": float(np.median([case["span_p0_5_p99_5"] for case in patient_cases])),
                "clip_total_ratio_median": float(np.median([case["clip_total_ratio"] for case in patient_cases])),
                "clip_high_ratio_max": float(np.max([case["clip_high_ratio"] for case in patient_cases])),
            }
        )
    return rows


def _write_case_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "patient_id",
        "case_id",
        "case_key",
        "p0_5",
        "p99_5",
        "span_p0_5_p99_5",
        "min",
        "max",
        "mean",
        "clip_low_ratio",
        "clip_high_ratio",
        "clip_total_ratio",
        "outlier_reasons",
        "cbct_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key) for key in fieldnames})


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    fixed = _read_json(args.fixed_manifest)
    split_manifest = _read_json(args.split_manifest)
    split_by_key = _split_lookup(split_manifest)
    cases: list[dict[str, Any]] = []
    missing_cases: list[dict[str, Any]] = []
    samples_by_split: dict[str, list[np.ndarray]] = {"train": [], "test": [], "all": []}

    records = fixed["records"]
    if args.max_cases is not None:
        records = records[: args.max_cases]
    for index, record in enumerate(records):
        key = _case_key(record)
        split = split_by_key.get(key, "unknown")
        cbct_path = Path(record["case_path"]) / "CBCT.nii.gz"
        if not cbct_path.exists():
            missing_cases.append(
                {
                    "split": split,
                    "patient_id": str(record["patient_id"]),
                    "case_id": str(record["case_id"]),
                    "case_key": key,
                    "cbct_path": str(cbct_path.resolve()),
                    "reason": "missing_cbct",
                }
            )
            continue
        stats, sampled = _stats_for_case(
            cbct_path,
            target_size=args.target_size,
            max_voxels=args.sample_voxels_per_case,
            fixed_min=args.fixed_min,
            fixed_max=args.fixed_max,
            seed=args.seed + index,
        )
        p = stats["percentiles"]
        row = {
            "split": split,
            "patient_id": str(record["patient_id"]),
            "case_id": str(record["case_id"]),
            "case_key": key,
            "cbct_path": str(cbct_path.resolve()),
            "p0_5": p["p0_5"],
            "p99_5": p["p99_5"],
            "span_p0_5_p99_5": p["p99_5"] - p["p0_5"],
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "clip_low_ratio": stats["fixed_window"]["clip_low_ratio"],
            "clip_high_ratio": stats["fixed_window"]["clip_high_ratio"],
            "clip_total_ratio": stats["fixed_window"]["clip_total_ratio"],
            "stats": stats,
        }
        cases.append(row)
        if split in ("train", "test"):
            samples_by_split[split].append(sampled)
        samples_by_split["all"].append(sampled)

    outlier_fields = ["p0_5", "p99_5", "span_p0_5_p99_5", "clip_total_ratio", "clip_high_ratio"]
    fences = {field: _iqr_flags([case[field] for case in cases]) for field in outlier_fields}
    for case in cases:
        reasons = []
        for field, (low, high) in fences.items():
            value = float(case[field])
            if value < low or value > high:
                reasons.append(field)
        if case["clip_total_ratio"] >= args.clip_outlier_ratio:
            reasons.append(f"clip_total_ratio>={args.clip_outlier_ratio:g}")
        case["outlier_reasons"] = ";".join(dict.fromkeys(reasons))

    global_windows = {}
    for split, arrays in samples_by_split.items():
        merged = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
        if merged.size:
            global_windows[split] = {
                "sampled_voxel_count": int(merged.size),
                "p0_5": float(np.percentile(merged, 0.5)),
                "p99_5": float(np.percentile(merged, 99.5)),
                "p0_1": float(np.percentile(merged, 0.1)),
                "p99_9": float(np.percentile(merged, 99.9)),
            }

    cases = sorted(cases, key=lambda item: (item["split"], int(item["patient_id"]), int(item["case_id"])))
    outliers = [case for case in cases if case["outlier_reasons"]]
    report = {
        "schema": "x2cbct_self_projection1208_hu_window_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_manifest": str(args.fixed_manifest.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "target_size": args.target_size,
        "sample_voxels_per_case": args.sample_voxels_per_case,
        "fixed_window": {"hu_min": args.fixed_min, "hu_max": args.fixed_max},
        "global_windows": global_windows,
        "case_summary": {
            "all": _summarize_cases(cases, "all"),
            "train": _summarize_cases(cases, "train"),
            "test": _summarize_cases(cases, "test"),
        },
        "patient_summary": _group_by_patient(cases),
        "outlier_policy": {
            "iqr_fields": outlier_fields,
            "iqr_fences": {key: {"low": value[0], "high": value[1]} for key, value in fences.items()},
            "clip_outlier_ratio": args.clip_outlier_ratio,
        },
        "missing_case_count": len(missing_cases),
        "missing_cases": missing_cases,
        "outlier_case_count": len(outliers),
        "outlier_cases": outliers,
        "cases": cases,
    }
    write_json(report, args.output)
    _write_case_csv(args.case_csv, cases)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit self_projection1208 CBCT HU windows per case and patient.")
    parser.add_argument("--fixed-manifest", type=Path, default=PROJECT_ROOT / "data_manifest" / "fixed_view_sampling_v0b.json")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "data_manifest" / "splits_v0b_train_test.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_hu_windows" / "hu_window_audit.json")
    parser.add_argument("--case-csv", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "self_projection1208_hu_windows" / "case_hu_windows.csv")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--sample-voxels-per-case", type=int, default=200_000)
    parser.add_argument("--fixed-min", type=float, default=-290.0)
    parser.add_argument("--fixed-max", type=float, default=2060.0)
    parser.add_argument("--clip-outlier-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    report = build_audit(args)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "case_csv": str(args.case_csv.resolve()),
                "case_count": len(report["cases"]),
                "outlier_case_count": report["outlier_case_count"],
                "global_windows": report["global_windows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
