from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_augmentation import setup_augmentation_config


def _parse_triplet(value: str) -> tuple[int, int, int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return items[0], items[1], items[2]


@dataclass(frozen=True)
class SizeConfig:
    target_shape: tuple[int, int, int]
    target_shape_text: str
    resample_spacing_xyz: tuple[float, float, float]
    resample_spacing_text: str
    volume_size: int
    detector_size: int
    detector_spacing_mm: float
    dso_mm: float
    dsd_mm: float
    label: str


def _format_mm(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _parse_float_triplet(value: str) -> tuple[float, float, float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return items[0], items[1], items[2]


def resolve_size_config(args: argparse.Namespace) -> SizeConfig:
    preset_size = int(args.size_preset)
    preset_spacing = 3.2 if preset_size == 128 else 1.6
    resample_spacing = _parse_float_triplet(args.resample_spacing) if args.resample_spacing else (preset_spacing, preset_spacing, preset_spacing)
    if len({round(item, 6) for item in resample_spacing}) != 1:
        raise ValueError(f"LUNA16 C2RV/DeepSparse protocol requires isotropic spacing, got {resample_spacing}")
    target_shape = _parse_triplet(args.target_shape) if args.target_shape else (preset_size, preset_size, preset_size)
    if len(set(target_shape)) != 1:
        raise ValueError(f"LUNA16 DIF-Net cache requires a cubic target shape, got {target_shape}")
    volume_size = int(target_shape[0])
    detector_size = int(args.detector_size) if args.detector_size is not None else volume_size
    if volume_size != detector_size and not args.allow_mismatched_size:
        raise ValueError(
            "volume size and projection detector size must match; "
            f"got volume={volume_size}, detector={detector_size}. "
            "Use --allow-mismatched-size only for an explicit ablation."
        )
    dso_mm = float(args.dso_mm)
    dsd_mm = float(args.dsd_mm)
    detector_spacing_mm = float(args.detector_spacing_mm) if args.detector_spacing_mm is not None else float(resample_spacing[0]) * dsd_mm / dso_mm
    return SizeConfig(
        target_shape=target_shape,
        target_shape_text=",".join(str(item) for item in target_shape),
        resample_spacing_xyz=resample_spacing,
        resample_spacing_text=",".join(f"{item:g}" for item in resample_spacing),
        volume_size=volume_size,
        detector_size=detector_size,
        detector_spacing_mm=detector_spacing_mm,
        dso_mm=dso_mm,
        dsd_mm=dsd_mm,
        label=f"vol{volume_size}_proj{detector_size}",
    )


def _inside_workspace(path: Path) -> bool:
    root = PROJECT_ROOT.resolve()
    target = path.resolve()
    try:
        return str(target).lower().startswith(str(root).lower())
    except RuntimeError:
        return False


def _safe_rmtree(path: Path, *, allow_outside_workspace: bool = False) -> None:
    resolved = path.resolve()
    if not allow_outside_workspace and not _inside_workspace(resolved):
        raise RuntimeError(f"refusing to remove outside workspace: {resolved}")
    if resolved.anchor and str(resolved) == resolved.anchor:
        raise RuntimeError(f"refusing to remove drive root: {resolved}")
    if resolved in {PROJECT_ROOT.resolve(), PROJECT_ROOT.resolve() / "data_cache", PROJECT_ROOT.resolve() / "dataset_list", resolved.home()}:
        raise RuntimeError(f"refusing to remove broad workspace path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _append_command_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(command: list[str], *, stage: str, command_log: Path) -> None:
    tool_name = Path(command[1]).name if len(command) > 1 else Path(command[0]).name
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "tool": tool_name,
        "command": command,
    }
    _append_command_log(command_log, payload)
    print(json.dumps({"event": "stage_start", "stage": stage, "tool": tool_name}, ensure_ascii=False), flush=True)
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        print(json.dumps({"event": "stage_failed", "stage": stage, "tool": tool_name}, ensure_ascii=False), flush=True)
        raise
    print(json.dumps({"event": "stage_done", "stage": stage, "tool": tool_name}, ensure_ascii=False), flush=True)


def _python() -> str:
    return sys.executable


def _default_output_root(args: argparse.Namespace) -> Path:
    size_config = resolve_size_config(args)
    spacing = _format_mm(size_config.resample_spacing_xyz[0]) + "mm"
    detector_spacing = _format_mm(size_config.detector_spacing_mm) + "mm"
    suffix = f"c2rvds_{spacing}_{size_config.label}_det{detector_spacing}"
    if int(args.drr_projection_count) != 50:
        suffix = f"{suffix}_projcnt{int(args.drr_projection_count)}"
    if args.split_policy != "c2rv":
        suffix = f"{suffix}_{args.split_policy}_seed{args.seed}"
    return PROJECT_ROOT / "data_cache" / "luna16" / f"difnet_cache_{suffix}"


def build_luna16_dataset(args: argparse.Namespace) -> dict[str, object]:
    size_config = resolve_size_config(args)
    setup_config = setup_augmentation_config(
        enabled=args.enable_setup_augmentation,
        translation_mm=args.translation_mm,
        rotation_deg=args.rotation_deg,
    )
    if int(args.drr_projection_count) <= 0:
        raise ValueError(f"--drr-projection-count must be positive, got {args.drr_projection_count}")
    output_root = (args.output_root or _default_output_root(args)).resolve()
    work_root = (args.work_root or (output_root.parent / f"_{output_root.name}_build_tmp")).resolve()
    registry_output = args.registry_output.resolve()
    report_output = args.profile_output.resolve()
    preprocess_root = work_root / "preprocessed"
    drr_root = work_root / "drr"
    command_log = output_root / "build_commands.jsonl"

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        _safe_rmtree(output_root, allow_outside_workspace=True)
    if work_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"work root already exists: {work_root}")
        _safe_rmtree(work_root, allow_outside_workspace=True)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    scan_cmd = [
        _python(),
        str(PROJECT_ROOT / "tools" / "scan_luna16_dataset_profile.py"),
        "--root",
        str(args.root),
        "--output",
        str(report_output),
        "--registry-output",
        str(registry_output),
        "--geometry-policy",
        args.geometry_policy,
        "--expected-inplane",
        args.expected_inplane,
        "--train-ratio",
        str(args.train_ratio),
        "--seed",
        str(args.seed),
        "--split-policy",
        args.split_policy,
        "--splits-json",
        str(args.splits_json),
    ]
    if args.hu_window_mode != "fixed" and not args.metadata_only_scan:
        scan_cmd.extend(
            [
                "--hu-percentiles",
                args.hu_percentiles,
                "--hu-window-split",
                args.hu_window_split,
                "--hu-sample-voxels-per-case",
                str(args.hu_sample_voxels_per_case),
            ]
        )
    if args.metadata_only_scan or args.hu_window_mode == "fixed":
        scan_cmd.append("--metadata-only")
    _run(scan_cmd, stage="scan_registry", command_log=command_log)

    preprocess_cmd = [
        _python(),
        str(PROJECT_ROOT / "tools" / "preprocess_ct_only_volumes.py"),
        "--registry",
        str(registry_output),
        "--output-root",
        str(preprocess_root),
        "--dataset-name",
        "luna16",
        "--image-path-key",
        "ct_path",
        "--case-id-key",
        "seriesuid",
        "--resample-spacing",
        size_config.resample_spacing_text,
        "--target-shape",
        size_config.target_shape_text,
        "--hu-window-mode",
        args.hu_window_mode,
        "--drop-hu-after-normalization",
        "--progress-interval",
        str(args.progress_interval),
    ]
    if args.hu_window_mode == "fixed":
        preprocess_cmd.extend(["--hu-fixed-min", str(args.hu_fixed_min), "--hu-fixed-max", str(args.hu_fixed_max)])
    else:
        preprocess_cmd.extend(
            [
                "--hu-percentiles",
                args.hu_percentiles,
                "--hu-window-split",
                args.hu_window_split,
                "--hu-sample-voxels-per-case",
                str(args.hu_sample_voxels_per_case),
            ]
        )
    if args.max_cases is not None:
        preprocess_cmd.extend(["--max-cases", str(args.max_cases)])
    _run(preprocess_cmd, stage="preprocess_ct", command_log=command_log)

    drr_cmd = [
        _python(),
        str(PROJECT_ROOT / "tools" / "build_synthrad2025_drr.py"),
        "--preprocess-report",
        str(preprocess_root / "preprocess_report.json"),
        "--output-root",
        str(drr_root),
        "--projector",
        args.projector,
        "--detector-size",
        str(size_config.detector_size),
        "--detector-spacing-mm",
        str(size_config.detector_spacing_mm),
        "--projection-count",
        str(args.drr_projection_count),
        "--dso-mm",
        str(size_config.dso_mm),
        "--dsd-mm",
        str(size_config.dsd_mm),
        "--projection-normalization",
        "c2rv-per-case-max",
        "--projection-loader-denominator",
        str(args.projection_loader_denominator),
        "--progress-interval",
        str(args.progress_interval),
        "--target-volume-mode",
        args.target_volume_mode,
    ]
    if args.max_cases is not None:
        drr_cmd.extend(["--max-cases", str(args.max_cases)])
    if args.enable_setup_augmentation:
        drr_cmd.append("--enable-setup-augmentation")
    for value in args.translation_mm or []:
        drr_cmd.extend(["--translation-mm", str(value)])
    for value in args.rotation_deg or []:
        drr_cmd.extend(["--rotation-deg", str(value)])
    _run(drr_cmd, stage="generate_drr", command_log=command_log)

    view_cmd = [
        _python(),
        str(PROJECT_ROOT / "tools" / "build_ct_only_view_cache.py"),
        "--drr-manifest",
        str(drr_root / "drr_manifest.json"),
        "--output-root",
        str(output_root),
        "--dataset-name",
        "luna16",
        "--views",
        args.views,
        "--target-size",
        str(size_config.volume_size),
        "--asset-mode",
        "copy",
    ]
    _run(view_cmd, stage="build_view_cache", command_log=command_log)

    manifest_path = output_root / "luna16" / "cache_manifest.json"
    summary = {
        "schema": "x2cbct_luna16_difnet_dataset_build.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "train_cache_root": str((output_root / "luna16").resolve()),
        "cache_manifest": str(manifest_path.resolve()),
        "command_log": str(command_log.resolve()),
        "registry": str(registry_output),
        "profile_report": str(report_output),
        "work_root": str(work_root),
        "intermediate_cleanup": not args.keep_intermediate,
        "views": args.views,
        "split_policy": args.split_policy,
        "splits_json": str(args.splits_json.resolve()),
        "size_preset": args.size_preset,
        "size_label": size_config.label,
        "target_shape": size_config.target_shape_text,
        "resample_spacing_xyz": list(size_config.resample_spacing_xyz),
        "volume_size": size_config.volume_size,
        "projection_size": size_config.detector_size,
        "drr_projection_count": int(args.drr_projection_count),
        "detector_size": size_config.detector_size,
        "detector_spacing_mm": size_config.detector_spacing_mm,
        "dso_mm": size_config.dso_mm,
        "dsd_mm": size_config.dsd_mm,
        "hu_window": [float(args.hu_fixed_min), float(args.hu_fixed_max)] if args.hu_window_mode == "fixed" else None,
        "projection_normalization": "c2rv-per-case-max",
        "projection_loader_denominator": float(args.projection_loader_denominator),
        "allow_mismatched_size": bool(args.allow_mismatched_size),
        "asset_mode": "copy",
        "target_volume_mode": args.target_volume_mode,
        "setup_augmentation": setup_config.to_dict(),
    }
    (output_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.keep_intermediate:
        _safe_rmtree(work_root, allow_outside_workspace=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a self-contained LUNA16 DIF-Net view cache in one command.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset_list" / "LUNA16")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--registry-output", type=Path, default=PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json")
    parser.add_argument("--profile-output", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "luna16_dataset_profile" / "luna16_profile_c2rv.json")
    parser.add_argument("--geometry-policy", choices=("compatible", "strict-exact"), default="compatible")
    parser.add_argument("--expected-inplane", default="512,512")
    parser.add_argument("--split-policy", choices=("c2rv", "patient"), default="c2rv")
    parser.add_argument("--splits-json", type=Path, default=PROJECT_ROOT / "external" / "c2rv_cbct_official" / "data" / "LUNA16" / "splits.json")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--resample-spacing", help="xyz spacing in mm. Defaults to 1.6 for 256 and 3.2 for 128.")
    parser.add_argument("--size-preset", choices=("128", "256"), default="256")
    parser.add_argument("--target-shape", help="Optional cubic z,y,x override. Defaults to --size-preset.")
    parser.add_argument("--detector-size", type=int, help="Optional projection detector size override. Defaults to the cubic target size.")
    parser.add_argument("--detector-spacing-mm", type=float, help="Defaults to voxel spacing * DSD / DSO.")
    parser.add_argument("--drr-projection-count", type=int, default=50, help="Strict C2RV uses 50. Use 10 for the view_2/view_5/view_10 sparse-view cache.")
    parser.add_argument("--dso-mm", type=float, default=800.0)
    parser.add_argument("--dsd-mm", type=float, default=1200.0)
    parser.add_argument("--allow-mismatched-size", action="store_true")
    parser.add_argument("--views", default="2,5,10")
    parser.add_argument("--projector", choices=("tigre", "fallback-ray-sum"), default="tigre")
    parser.add_argument("--hu-window-mode", choices=("percentile", "legacy", "fixed"), default="fixed")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-window-split", choices=("train", "all"), default="train")
    parser.add_argument("--hu-fixed-min", type=float, default=-1000.0)
    parser.add_argument("--hu-fixed-max", type=float, default=1024.0)
    parser.add_argument("--projection-loader-denominator", type=float, default=0.2)
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=20_000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--enable-setup-augmentation", action="store_true")
    parser.add_argument("--translation-mm", action="append")
    parser.add_argument("--rotation-deg", action="append")
    parser.add_argument("--target-volume-mode", choices=("full", "base_reference"), default="full")
    parser.add_argument("--metadata-only-scan", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = build_luna16_dataset(args)
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
