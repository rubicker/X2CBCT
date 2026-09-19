from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import write_json
from x2cbct_difnet.ct_only_preprocess import parse_triplet
from x2cbct_difnet.normalization import LEGACY_CBCT_NORMALIZATION, cbct_normalization_profile
from x2cbct_difnet.normalization_profile import estimate_grouped_hu_profiles, parse_percentiles
from x2cbct_difnet.synthrad2025 import load_registry, preprocess_cbct_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess SynthRAD2025 chest/abdomen CBCT to 128^3 normalized arrays.")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "data_manifest" / "synthrad2025_task2_chest_abdomen_registry.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data_cache" / "synthrad2025_task2" / "preprocessed_cbct_128")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--resample-spacing", default="0.8,0.8,0.8", help="xyz isotropic spacing in mm before center crop/pad")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--anatomy", choices=("chest", "abdomen", "all"), default="all")
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="percentile")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-window-split", choices=("train", "all"), default="train")
    parser.add_argument("--hu-fixed-min", type=float)
    parser.add_argument("--hu-fixed-max", type=float)
    parser.add_argument("--mask-policy", choices=("body", "finite"), default="body")
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=200_000)
    args = parser.parse_args()
    resample_spacing_xyz = parse_triplet(args.resample_spacing, value_type=float)

    registry = load_registry(args.registry)
    rows = [
        item
        for item in registry["cases"]
        if args.anatomy == "all" or item["anatomy"] == args.anatomy
    ]
    if args.max_cases is not None:
        rows = rows[: int(args.max_cases)]
    if args.hu_window_mode == "legacy":
        profiles = {key: dict(LEGACY_CBCT_NORMALIZATION) for key in sorted({str(row["anatomy"]) for row in rows})}
        profile_case_stats = []
    elif args.hu_window_mode == "fixed":
        if args.hu_fixed_min is None or args.hu_fixed_max is None:
            parser.error("--hu-fixed-min and --hu-fixed-max are required when --hu-window-mode fixed")
        profiles = {
            key: cbct_normalization_profile(
                hu_min=args.hu_fixed_min,
                hu_max=args.hu_fixed_max,
                profile_id=f"{key}_fixed_{args.hu_fixed_min:g}_{args.hu_fixed_max:g}",
                mode="fixed",
            )
            for key in sorted({str(row["anatomy"]) for row in rows})
        }
        profile_case_stats = []
    else:
        profiles, profile_case_stats = estimate_grouped_hu_profiles(
            rows,
            image_path=lambda item: Path(item["cbct_path"]),
            mask_path=lambda item: Path(item["mask_path"]) if item.get("mask_path") else None,
            group_key="anatomy",
            source_split=args.hu_window_split,
            target_size=args.target_size,
            percentiles=parse_percentiles(args.hu_percentiles),
            mask_policy=args.mask_policy,
            max_voxels_per_case=args.hu_sample_voxels_per_case,
        )
    processed = []
    failures = []
    for row in rows:
        try:
            processed.append(
                preprocess_cbct_case(
                    row,
                    args.output_root,
                    target_size=args.target_size,
                    resample_spacing_xyz=resample_spacing_xyz,
                    cbct_normalization=profiles[str(row["anatomy"])],
                )
            )
        except Exception as exc:
            failed = dict(row)
            failed["status"] = "failed"
            failed["error"] = repr(exc)
            failures.append(failed)
            print(json.dumps({"ok": False, "case_id": row.get("case_id"), "error": repr(exc)}, ensure_ascii=False), flush=True)
    report = {
        "schema": "x2cbct_synthrad2025_task2_preprocess_report.v1",
        "geometry_branch": "simulated_drr",
        "axis_convention": {
            "array_axis": "zyx",
            "metadata_axis": "xyz",
            "simpleitk_spacing_axis": "xyz",
            "simpleitk_size_axis": "xyz",
        },
        "registry": str(args.registry),
        "output_root": str(args.output_root),
        "target_size": args.target_size,
        "resample_spacing_xyz": [float(item) for item in resample_spacing_xyz],
        "preprocess_policy": "spacing_aware_resample_then_center_crop_pad",
        "cbct_normalization": {
            "mode": args.hu_window_mode,
            "scope": "anatomy",
            "source_split": args.hu_window_split,
            "profiles": profiles,
            "case_stats": profile_case_stats,
        },
        "processed_count": len(processed),
        "failed_count": len(failures),
        "processed_cases": processed,
        "failed_cases": failures,
    }
    report_path = args.output_root / "preprocess_report.json"
    write_json(report, report_path)
    print(json.dumps({"ok": not failures, "report": str(report_path), "processed_count": len(processed), "failed_count": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
