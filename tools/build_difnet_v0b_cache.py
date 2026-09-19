from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import CacheBuildOptions, build_difnet_v0b_cache, default_projection_correction_profile_path
from x2cbct_difnet.cache import PROJECTION_CORRECTION_MODES, PROJECTION_CORRECTION_NONE
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.setup_augmentation import setup_augmentation_config
from x2cbct_difnet.split_policy import PATIENT_LEVEL_POLICY, normalize_split_policy


def _default_cache_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "data_cache" / "difnet_v0b" / f"v0b-cache-{stamp}"


def _views(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(view_key_from_count(item) for item in SUPPORTED_VIEW_COUNTS)
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(view_key_from_count(int(part)))
    return tuple(dict.fromkeys(values))


def _start_visualization_watcher(args: argparse.Namespace, output_root: Path) -> subprocess.Popen[str] | None:
    if not args.visualize_progress:
        return None
    watcher = PROJECT_ROOT / "tools" / "watch_difnet_v0b_cache_build.py"
    visualization_root = args.visualization_root or (
        PROJECT_ROOT / "runs" / "viewers" / "difnet_v0b_cache_progress" / output_root.name
    )
    visualization_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(watcher),
        "--cache-root",
        str(output_root),
        "--output-dir",
        str(visualization_root),
        "--view-count",
        str(args.visualization_view_count),
        "--split",
        str(args.visualization_split),
        "--max-preview-samples",
        str(args.visualization_max_preview_samples),
        "--interval-seconds",
        str(args.visualization_interval_seconds),
        "--watch",
        "--stop-when-manifest",
    ]
    stdout = (visualization_root / "watcher_stdout.log").open("a", encoding="utf-8")
    stderr = (visualization_root / "watcher_stderr.log").open("a", encoding="utf-8")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    print(
        json.dumps(
            {
                "visualization_watcher_pid": process.pid,
                "visualization_index": str((visualization_root / "index.html").resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DIF-Net V0b preprocessing cache.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--views", default="all", help="all or comma-separated view counts, e.g. 2,5,10")
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="fixed")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-fixed-min", type=float, default=-290.0)
    parser.add_argument("--hu-fixed-max", type=float, default=2060.0)
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=200_000)
    parser.add_argument("--split-policy", type=normalize_split_policy, default=PATIENT_LEVEL_POLICY)
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=20260517)
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
    parser.add_argument("--enable-setup-augmentation", action="store_true")
    parser.add_argument("--target-volume-mode", choices=("full", "base_reference"), default="full")
    parser.add_argument(
        "--translation-mm",
        action="append",
        help="Repeat for each integer S-I translation bin. Must include 0 when setup augmentation is enabled.",
    )
    parser.add_argument(
        "--rotation-deg",
        action="append",
        help="Repeat for each integer yaw rotation bin. Must include 0 when setup augmentation is enabled.",
    )
    parser.add_argument("--visualize-progress", action="store_true")
    parser.add_argument("--visualization-root", type=Path)
    parser.add_argument("--visualization-interval-seconds", type=float, default=15.0)
    parser.add_argument("--visualization-view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument("--visualization-split", choices=("train", "eval", "test", "all"), default="all")
    parser.add_argument("--visualization-max-preview-samples", type=int, default=6)
    parser.add_argument("--progress-interval", type=int, default=25, help="Print console progress every N cases. Use 0 to disable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or _default_cache_root()
    options = CacheBuildOptions(
        project_root=PROJECT_ROOT,
        output_root=output_root,
        views=_views(args.views),
        projection_size=args.projection_size,
        target_size=args.target_size,
        max_train_cases=args.max_train_cases,
        max_test_cases=args.max_test_cases,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        command=" ".join(sys.argv),
        hu_window_mode=args.hu_window_mode,
        hu_percentiles=tuple(float(item.strip()) for item in args.hu_percentiles.split(",") if item.strip()),
        hu_fixed_min=args.hu_fixed_min,
        hu_fixed_max=args.hu_fixed_max,
        hu_sample_voxels_per_case=args.hu_sample_voxels_per_case,
        split_policy=args.split_policy,
        split_train_ratio=args.split_train_ratio,
        split_seed=args.split_seed,
        projection_correction_mode=args.projection_correction_mode,
        projection_correction_view_key=view_key_from_count(args.projection_correction_view),
        projection_correction_profile_path=args.projection_correction_profile,
        target_volume_mode=args.target_volume_mode,
        progress_interval=args.progress_interval,
        setup_augmentation=setup_augmentation_config(
            enabled=args.enable_setup_augmentation,
            translation_mm=args.translation_mm,
            rotation_deg=args.rotation_deg,
        ),
    )
    _start_visualization_watcher(args, output_root)
    manifest = build_difnet_v0b_cache(options)
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(output_root.resolve()),
                "dry_run": args.dry_run,
                "view_sample_counts": manifest["view_sample_counts"],
                "failed_case_count": len(manifest["failed_cases"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
