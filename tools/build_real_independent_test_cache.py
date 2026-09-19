from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import (
    CacheBuildOptions,
    PROJECTION_CORRECTION_MODES,
    PROJECTION_CORRECTION_NONE,
    build_difnet_v0b_cache,
    default_projection_correction_profile_path,
    read_json,
)
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.setup_augmentation import setup_augmentation_config


def _default_output_root(derivative_manifest: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = derivative_manifest.stem.replace("_noaug", "-test-noaug").replace("_aug", "-test-aug")
    return PROJECT_ROOT / "data_cache" / "real" / f"{name}-{stamp}"


def _views(raw: str) -> tuple[str, ...]:
    values = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        values.append(view_key_from_count(int(token)) if token.isdigit() else token)
    return tuple(dict.fromkeys(values))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a self_projection1208 DIF-Net V1 independent test cache from a fixed derivative manifest."
    )
    parser.add_argument("--derivative-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--views", default="2,5,10")
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="fixed")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-fixed-min", type=float, default=-290.0)
    parser.add_argument("--hu-fixed-max", type=float, default=2060.0)
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=200_000)
    parser.add_argument(
        "--projection-correction-mode",
        choices=PROJECTION_CORRECTION_MODES,
        default=PROJECTION_CORRECTION_NONE,
    )
    parser.add_argument("--projection-correction-view", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument(
        "--projection-correction-profile",
        type=Path,
        default=default_projection_correction_profile_path(PROJECT_ROOT),
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-missing-cases",
        action="store_true",
        help="Allow building a local subset when fixed-list test cases are missing required local files.",
    )
    args = parser.parse_args()

    derivative_manifest = read_json(args.derivative_manifest)
    if derivative_manifest.get("dataset") != "self_projection1208":
        raise ValueError("--derivative-manifest must describe self_projection1208")
    augmentation = derivative_manifest.get("augmentation", {})
    setup_config = setup_augmentation_config(
        enabled=bool(augmentation.get("enabled")),
        translation_mm=augmentation.get("translation_mm"),
        rotation_deg=augmentation.get("rotation_deg"),
    )
    output_root = (args.output_root or _default_output_root(args.derivative_manifest)).resolve()
    manifest = build_difnet_v0b_cache(
        CacheBuildOptions(
            project_root=PROJECT_ROOT,
            output_root=output_root,
            views=_views(args.views),
            projection_size=args.projection_size,
            target_size=args.target_size,
            max_train_cases=0,
            max_test_cases=args.max_test_cases,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            command=" ".join(sys.argv),
            hu_window_mode=args.hu_window_mode,
            hu_percentiles=tuple(float(item.strip()) for item in args.hu_percentiles.split(",") if item.strip()),
            hu_fixed_min=args.hu_fixed_min,
            hu_fixed_max=args.hu_fixed_max,
            hu_sample_voxels_per_case=args.hu_sample_voxels_per_case,
            projection_correction_mode=args.projection_correction_mode,
            projection_correction_view_key=view_key_from_count(args.projection_correction_view),
            projection_correction_profile_path=args.projection_correction_profile,
            setup_augmentation=setup_config,
            target_volume_mode="full",
            progress_interval=args.progress_interval,
            records_manifest_path=args.derivative_manifest,
            records_manifest_split="test",
            records_output_split="test",
            allow_missing_required_case_files=args.allow_missing_cases,
        )
    )
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(output_root),
                "dry_run": args.dry_run,
                "view_sample_counts": manifest["view_sample_counts"],
                "failed_case_count": len(manifest["failed_cases"]),
                "derivative_manifest": str(args.derivative_manifest.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
