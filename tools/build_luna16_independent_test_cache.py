from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json
from x2cbct_difnet.luna16_v1_preprocess import materialize_luna16_registry


def _default_output_root(derivative_manifest: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = derivative_manifest.stem.replace("_noaug", "-test-noaug").replace("_aug", "-test-aug")
    return PROJECT_ROOT / "data_cache" / "luna16" / f"{name}-{stamp}"


def _safe_rmtree(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    target = path.resolve()
    if not str(target).lower().startswith(str(root).lower()):
        raise RuntimeError(f"refusing to remove outside workspace: {target}")
    if target in {root, root / "data_cache", root / "dataset_list"}:
        raise RuntimeError(f"refusing to remove broad workspace path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _run(command: list[str], *, stage: str) -> None:
    print(json.dumps({"event": "stage_start", "stage": stage}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(json.dumps({"event": "stage_done", "stage": stage}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a LUNA16 DIF-Net V1 independent test cache from a fixed derivative manifest.")
    parser.add_argument("--derivative-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset_list" / "LUNA16")
    parser.add_argument("--size-preset", choices=("128", "256"), default="256")
    parser.add_argument("--resample-spacing")
    parser.add_argument("--target-shape")
    parser.add_argument("--projector", choices=("tigre", "fallback-ray-sum"), default="tigre")
    parser.add_argument("--detector-size", type=int)
    parser.add_argument("--detector-spacing-mm", type=float)
    parser.add_argument("--dso-mm", type=float, default=800.0)
    parser.add_argument("--dsd-mm", type=float, default=1200.0)
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="fixed")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-fixed-min", type=float, default=-1000.0)
    parser.add_argument("--hu-fixed-max", type=float, default=1024.0)
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=20_000)
    parser.add_argument("--projection-loader-denominator", type=float, default=0.2)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    derivative_manifest = read_json(args.derivative_manifest)
    if derivative_manifest.get("dataset") != "luna16":
        raise ValueError("--derivative-manifest must describe luna16")
    augmentation = derivative_manifest.get("augmentation", {})
    preset = int(args.size_preset)
    spacing = args.resample_spacing or ("1.6,1.6,1.6" if preset == 256 else "3.2,3.2,3.2")
    spacing_values = tuple(float(item.strip()) for item in spacing.split(","))
    target_shape = args.target_shape or f"{preset},{preset},{preset}"
    detector_size = int(args.detector_size or preset)
    detector_spacing = float(args.detector_spacing_mm or (spacing_values[0] * float(args.dsd_mm) / float(args.dso_mm)))
    output_root = (args.output_root or _default_output_root(args.derivative_manifest)).resolve()
    work_root = output_root.parent / f"_{output_root.name}_build_tmp"

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        _safe_rmtree(output_root)
    if work_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"work root already exists: {work_root}")
        _safe_rmtree(work_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    registry_path = work_root / "registry" / "luna16_test_registry.json"
    materialize_luna16_registry(
        manifest_path=args.derivative_manifest,
        output_path=registry_path,
        source_split="test",
        output_split="test",
        root=args.root,
        command=" ".join(sys.argv),
    )
    preprocess_root = work_root / "preprocessed"
    drr_root = work_root / "drr"
    preprocess_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "preprocess_ct_only_volumes.py"),
        "--registry",
        str(registry_path),
        "--output-root",
        str(preprocess_root),
        "--dataset-name",
        "luna16",
        "--image-path-key",
        "ct_path",
        "--case-id-key",
        "seriesuid",
        "--resample-spacing",
        spacing,
        "--target-shape",
        target_shape,
        "--hu-window-mode",
        args.hu_window_mode,
        "--drop-hu-after-normalization",
        "--progress-interval",
        str(args.progress_interval),
    ]
    if args.hu_window_mode == "fixed":
        preprocess_cmd.extend(["--hu-fixed-min", str(args.hu_fixed_min), "--hu-fixed-max", str(args.hu_fixed_max)])
    else:
        preprocess_cmd.extend(["--hu-percentiles", args.hu_percentiles, "--hu-window-split", "all", "--hu-sample-voxels-per-case", str(args.hu_sample_voxels_per_case)])
    if args.max_test_cases is not None:
        preprocess_cmd.extend(["--max-cases", str(args.max_test_cases)])
    _run(preprocess_cmd, stage="preprocess_ct")

    drr_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "build_synthrad2025_drr.py"),
        "--preprocess-report",
        str(preprocess_root / "preprocess_report.json"),
        "--output-root",
        str(drr_root),
        "--projector",
        args.projector,
        "--detector-size",
        str(detector_size),
        "--detector-spacing-mm",
        str(detector_spacing),
        "--projection-count",
        "10",
        "--dso-mm",
        str(args.dso_mm),
        "--dsd-mm",
        str(args.dsd_mm),
        "--projection-normalization",
        "c2rv-per-case-max",
        "--projection-loader-denominator",
        str(args.projection_loader_denominator),
        "--target-volume-mode",
        "full",
        "--progress-interval",
        str(args.progress_interval),
    ]
    if augmentation.get("enabled"):
        drr_cmd.append("--enable-setup-augmentation")
        for value in augmentation.get("translation_mm", [-10, 0, 10]):
            drr_cmd.extend(["--translation-mm", str(value)])
        for value in augmentation.get("rotation_deg", [-10, 0, 10]):
            drr_cmd.extend(["--rotation-deg", str(value)])
    if args.max_test_cases is not None:
        drr_cmd.extend(["--max-cases", str(args.max_test_cases)])
    _run(drr_cmd, stage="generate_drr")

    view_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "build_ct_only_view_cache.py"),
        "--drr-manifest",
        str(drr_root / "drr_manifest.json"),
        "--output-root",
        str(output_root),
        "--dataset-name",
        "luna16",
        "--views",
        "2,5,8,10",
        "--target-size",
        str(preset),
        "--asset-mode",
        "copy",
    ]
    _run(view_cmd, stage="build_view_cache")
    summary = {
        "schema": "x2cbct_luna16_v1_independent_test_build.v1",
        "output_root": str(output_root),
        "test_cache_root": str((output_root / "luna16").resolve()),
        "derivative_manifest": str(args.derivative_manifest.resolve()),
        "registry": str(registry_path.resolve()),
        "target_volume_mode": "full",
        "views": ["view_2", "view_5", "view_8", "view_10"],
        "setup_augmentation": augmentation,
    }
    (output_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.keep_intermediate:
        _safe_rmtree(work_root)
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
