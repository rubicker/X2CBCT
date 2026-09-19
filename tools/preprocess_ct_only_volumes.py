from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.ct_only_preprocess import (
    load_config,
    parse_triplet,
    preprocess_ct_only_dataset,
)
from x2cbct_difnet.normalization import cbct_normalization_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess CT-only/CBCT-only volumes with spacing-aware resampling for simulated_drr caches."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--dataset-name", default="luna16")
    parser.add_argument("--image-path-key", default="ct_path")
    parser.add_argument("--case-id-key", default="seriesuid")
    parser.add_argument("--resample-spacing", default="1.6,1.6,1.6", help="xyz spacing in mm")
    parser.add_argument("--target-shape", default="128,128,128", help="zyx array shape")
    parser.add_argument("--pad-value-hu", type=float, default=-1024.0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="percentile")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-window-split", choices=("train", "all"), default="train")
    parser.add_argument("--hu-fixed-min", type=float)
    parser.add_argument("--hu-fixed-max", type=float)
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=200_000)
    parser.add_argument("--drop-hu-after-normalization", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    overrides = {
        "dataset_name": args.dataset_name,
        "resample_spacing_xyz": parse_triplet(args.resample_spacing, value_type=float),
        "target_shape_zyx": parse_triplet(args.target_shape, value_type=int),
        "pad_value_hu": args.pad_value_hu,
        "hu_window_mode": args.hu_window_mode,
    }
    if args.hu_window_mode != "fixed":
        overrides.update(
            {
                "hu_percentiles": args.hu_percentiles,
                "hu_window_split": args.hu_window_split,
                "hu_sample_voxels_per_case": args.hu_sample_voxels_per_case,
            }
        )
    config = load_config(args.config_json, overrides)
    fixed_normalization = None
    if args.hu_window_mode == "fixed":
        if args.hu_fixed_min is None or args.hu_fixed_max is None:
            parser.error("--hu-fixed-min and --hu-fixed-max are required when --hu-window-mode fixed")
        fixed_normalization = cbct_normalization_profile(
            hu_min=args.hu_fixed_min,
            hu_max=args.hu_fixed_max,
            profile_id=f"{config.dataset_name}_fixed_{args.hu_fixed_min:g}_{args.hu_fixed_max:g}",
            mode="fixed",
        )
    report = preprocess_ct_only_dataset(
        registry_path=args.registry,
        output_root=args.output_root,
        config=config,
        image_path_key=args.image_path_key,
        case_id_key=args.case_id_key,
        max_cases=args.max_cases,
        command=" ".join(sys.argv),
        fixed_normalization=fixed_normalization,
        keep_hu_volumes=not args.drop_hu_after_normalization,
        progress_interval=max(0, int(args.progress_interval)),
    )
    report_path = args.output_root / "preprocess_report.json"
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(report_path),
                "processed_count": report["processed_count"],
                "spacing_xyz": list(config.resample_spacing_xyz),
                "target_shape_zyx": list(config.target_shape_zyx),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
