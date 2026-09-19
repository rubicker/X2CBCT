from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json, write_json
from x2cbct_difnet.setup_augmentation import (
    iter_augmentations,
    setup_augmentation_config,
    transform_volume_setup,
    volume_background_value,
)
from x2cbct_difnet.official_drr import (
    OFFICIAL_DIFNET_PROJECTION_NORM,
    OfficialDrrConfig,
    generate_raw_drr,
    normalize_c2rv_projection,
    normalize_official_difnet_projection,
    official_angles_deg,
)
from x2cbct_difnet.normalization import normalize_projection
from x2cbct_difnet.volume_grid import is_isotropic_spacing


def _drr_config_for_row(row: dict, fallback: OfficialDrrConfig) -> tuple[OfficialDrrConfig, dict]:
    shape = row.get("processed_shape_zyx") or row.get("target_shape_zyx")
    spacing = row.get("processed_spacing_xyz") or row.get("target_spacing_xyz")
    notes: dict = {}
    voxel_size = int(shape[0]) if shape and len(set(int(item) for item in shape)) == 1 else fallback.voxel_size
    voxel_spacing = fallback.voxel_spacing_mm
    if spacing is not None:
        if is_isotropic_spacing(spacing):
            voxel_spacing = float(spacing[0])
            notes["voxel_spacing_source"] = "processed_spacing_xyz"
        else:
            notes["voxel_spacing_source"] = "fallback_due_non_isotropic_processed_spacing"
            notes["non_isotropic_processed_spacing_xyz"] = [float(item) for item in spacing]
    return (
        OfficialDrrConfig(
            projection_count=fallback.projection_count,
            total_angle_deg=fallback.total_angle_deg,
            start_angle_deg=fallback.start_angle_deg,
            detector_size=fallback.detector_size,
            voxel_size=voxel_size,
            dsd_mm=fallback.dsd_mm,
            dso_mm=fallback.dso_mm,
            detector_spacing_mm=fallback.detector_spacing_mm,
            voxel_spacing_mm=voxel_spacing,
        ),
        notes,
    )


def _infer_detector_size(rows: list[dict], requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    for row in rows:
        shape = row.get("processed_shape_zyx") or row.get("target_shape_zyx")
        if shape and len(shape) == 3 and len(set(int(item) for item in shape)) == 1:
            return int(shape[0])
    return OfficialDrrConfig().detector_size


def _normalization_mode(requested: str, projector: str) -> str:
    if requested != "auto":
        return requested
    return "official-difnet-fixed" if projector == "tigre" else "smoke-float16"


def _normalize_drr_projection(
    projections: np.ndarray,
    *,
    mode: str,
    loader_denominator: float,
) -> tuple[np.ndarray, dict]:
    if mode == "official-difnet-fixed":
        return normalize_official_difnet_projection(projections), {
            "projection_value_space": "official_uint8_0_255",
            "projection_loader_scale": 255.0,
            "official_projection_norm": OFFICIAL_DIFNET_PROJECTION_NORM,
            "projection_normalization": "official_difnet_fixed_0p10_uint8",
        }
    if mode == "c2rv-per-case-max":
        normalized, meta = normalize_c2rv_projection(projections, loader_denominator=loader_denominator)
        return normalized, {
            "projection_value_space": "c2rv_uint8_0_255",
            "official_projection_norm": None,
            **meta,
        }
    if mode == "smoke-float16":
        return normalize_projection(projections).astype(np.float16), {
            "projection_value_space": "smoke_float16_0_1",
            "projection_loader_scale": 1.0,
            "official_projection_norm": None,
            "projection_normalization": "normalize_projection_float16",
        }
    raise ValueError(f"unsupported projection normalization: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official-geometry DRR projections for preprocessed SynthRAD2025 CBCT.")
    parser.add_argument("--preprocess-report", type=Path, default=PROJECT_ROOT / "data_cache" / "synthrad2025_task2" / "preprocessed_cbct_128" / "preprocess_report.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data_cache" / "synthrad2025_task2" / "drr_128")
    parser.add_argument("--projector", choices=("tigre", "fallback-ray-sum"), default="tigre")
    parser.add_argument("--detector-size", type=int, help="Projection detector size. Defaults to the cubic processed volume size.")
    parser.add_argument("--detector-spacing-mm", type=float)
    parser.add_argument("--dso-mm", type=float)
    parser.add_argument("--dsd-mm", type=float)
    parser.add_argument("--projection-count", type=int, default=50, help="Number of equally spaced DRR projections to generate.")
    parser.add_argument(
        "--projection-normalization",
        choices=("auto", "official-difnet-fixed", "c2rv-per-case-max", "smoke-float16"),
        default="auto",
    )
    parser.add_argument("--projection-loader-denominator", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--enable-setup-augmentation", action="store_true")
    parser.add_argument("--translation-mm", action="append")
    parser.add_argument("--rotation-deg", action="append")
    parser.add_argument(
        "--target-volume-mode",
        choices=("full", "base_reference"),
        default="full",
        help="full saves setup-transformed target volumes; base_reference keeps only the base target volume path.",
    )
    args = parser.parse_args()

    report = read_json(args.preprocess_report)
    rows = list(report.get("processed_cases", []))
    if args.max_cases is not None:
        rows = rows[: int(args.max_cases)]
    if int(args.projection_count) <= 0:
        raise ValueError(f"--projection-count must be positive, got {args.projection_count}")
    detector_size = _infer_detector_size(rows, args.detector_size)
    base = OfficialDrrConfig()
    default_config = OfficialDrrConfig(
        projection_count=int(args.projection_count),
        detector_size=detector_size,
        dsd_mm=float(args.dsd_mm if args.dsd_mm is not None else base.dsd_mm),
        dso_mm=float(args.dso_mm if args.dso_mm is not None else base.dso_mm),
        detector_spacing_mm=float(args.detector_spacing_mm if args.detector_spacing_mm is not None else base.detector_spacing_mm),
    )
    normalization_mode = _normalization_mode(args.projection_normalization, args.projector)
    setup_config = setup_augmentation_config(
        enabled=args.enable_setup_augmentation,
        translation_mm=args.translation_mm,
        rotation_deg=args.rotation_deg,
    )
    augmentations = iter_augmentations(setup_config)
    generated = []
    failures = []
    total_rows = len(rows) * len(augmentations)
    progress_interval = max(0, int(args.progress_interval))
    done_count = 0
    for row in rows:
        source_case_id = str(row["case_id"])
        source_volume = np.load(row["processed_cbct_path"]).astype(np.float32)
        config, config_notes = _drr_config_for_row(row, default_config)
        spacing = row.get("processed_spacing_xyz") or row.get("target_spacing_xyz") or [config.voxel_spacing_mm] * 3
        fill_value = volume_background_value(source_volume)
        for augmentation in augmentations:
            done_count += 1
            try:
                if setup_config.enabled:
                    case_id = f"{source_case_id}_{augmentation.id}"
                    volume = transform_volume_setup(
                        source_volume,
                        translation_mm=int(augmentation.translation_mm),
                        rotation_deg=int(augmentation.rotation_deg),
                        spacing_z_mm=float(spacing[2]),
                        fill_value=fill_value,
                    )
                else:
                    case_id = source_case_id
                    volume = source_volume
                angles = official_angles_deg(config)
                raw_projections = generate_raw_drr(volume, projector=args.projector, config=config)
                projections, projection_meta = _normalize_drr_projection(
                    raw_projections,
                    mode=normalization_mode,
                    loader_denominator=args.projection_loader_denominator,
                )
                group = str(row.get("anatomy", row.get("dataset_name", "ct_only")))
                case_dir = args.output_root / group / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                target_volume_mode = str(args.target_volume_mode)
                if setup_config.enabled and target_volume_mode == "full":
                    processed_cbct_path = case_dir / "ct_norm.npy"
                    np.save(processed_cbct_path, volume.astype(np.float16))
                else:
                    processed_cbct_path = Path(row["processed_cbct_path"]).resolve()
                projection_paths = []
                for projection_index, projection in enumerate(projections):
                    path = case_dir / f"projection_{projection_index:03d}.npy"
                    np.save(path, projection)
                    projection_paths.append(str(path.resolve()))
                case_manifest = {
                    **row,
                    "case_id": case_id,
                    "source_case_id": source_case_id,
                    "status": "drr_generated",
                    "projector": args.projector,
                    "processed_cbct_path": str(processed_cbct_path),
                    "target_volume_mode": target_volume_mode,
                    **projection_meta,
                    "drr_config": config.to_dict(),
                    "drr_config_notes": config_notes,
                    "projection_geometry": {
                        "type": "simulated_drr",
                        "projector": args.projector,
                        "voxel_size": config.voxel_size,
                        "voxel_spacing_mm": config.voxel_spacing_mm,
                        "detector_size": config.detector_size,
                        "detector_spacing_mm": config.detector_spacing_mm,
                        "dso_mm": config.dso_mm,
                        "dsd_mm": config.dsd_mm,
                        "angles_deg": [float(item) for item in angles],
                        "off_origin_mm": [0.0, 0.0, 0.0],
                        "canonical_center_is_isocenter": True,
                    },
                    "angles_deg": [float(item) for item in angles],
                    "projection_paths": projection_paths,
                }
                if setup_config.enabled:
                    case_manifest["setup_augmentation"] = {
                        **augmentation.to_dict(),
                        "source_case_id": source_case_id,
                        "volume_transform_order": "rotate_yaw_then_translate_si",
                        "volume_fill_value": float(fill_value),
                    }
                write_json(case_manifest, case_dir / "drr_case_manifest.json")
                generated.append(case_manifest)
                if progress_interval > 0 and (done_count == 1 or done_count == total_rows or done_count % progress_interval == 0):
                    print(
                        json.dumps(
                            {
                                "event": "drr_progress",
                                "done": done_count,
                                "total": total_rows,
                                "case_id": case_id,
                                "projection_count": len(projection_paths),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            except Exception as exc:
                failed = dict(row)
                failed["status"] = "failed"
                failed["source_case_id"] = source_case_id
                failed["augmentation_id"] = augmentation.id if setup_config.enabled else None
                failed["translation_mm"] = int(augmentation.translation_mm)
                failed["rotation_deg"] = int(augmentation.rotation_deg)
                failed["error"] = repr(exc)
                failures.append(failed)
                print(json.dumps({"ok": False, "case_id": source_case_id, "error": repr(exc)}, ensure_ascii=False), flush=True)
    manifest_path = args.output_root / "drr_manifest.json"
    split_policies = sorted({str(row.get("split_policy")) for row in generated if row.get("split_policy")})
    manifest_projection_value_spaces = sorted({str(row.get("projection_value_space")) for row in generated if row.get("projection_value_space")})
    manifest_loader_scales = sorted({float(row.get("projection_loader_scale")) for row in generated if row.get("projection_loader_scale") is not None})
    out = {
        "schema": "x2cbct_synthrad2025_task2_drr_manifest.v1",
        "manifest_path": str(manifest_path.resolve()),
        "preprocess_report": str(args.preprocess_report),
        "output_root": str(args.output_root),
        "projector": args.projector,
        "projection_value_space": manifest_projection_value_spaces[0] if len(manifest_projection_value_spaces) == 1 else manifest_projection_value_spaces,
        "projection_loader_scale": manifest_loader_scales[0] if len(manifest_loader_scales) == 1 else None,
        "projection_loader_scale_policy": "constant" if len(manifest_loader_scales) == 1 else "per_case",
        "projection_normalization": normalization_mode,
        "target_volume_mode": args.target_volume_mode,
        "official_projection_norm": OFFICIAL_DIFNET_PROJECTION_NORM if normalization_mode == "official-difnet-fixed" else None,
        "cbct_normalization": report.get("cbct_normalization"),
        "registry_split": report.get("registry_split"),
        "split_policy": split_policies[0] if len(split_policies) == 1 else split_policies,
        "drr_config": default_config.to_dict(),
        "drr_config_policy": "per_case_uses_processed_isotropic_spacing_when_available",
        "projection_geometry": {
            "type": "simulated_drr",
            "projector": args.projector,
            "drr_config_policy": "per_case_uses_processed_isotropic_spacing_when_available",
            "off_origin_mm": [0.0, 0.0, 0.0],
            "canonical_center_is_isocenter": True,
        },
        "setup_augmentation": setup_config.to_dict(),
        "generated_count": len(generated),
        "failed_count": len(failures),
        "cases": generated,
        "failed_cases": failures,
    }
    write_json(out, manifest_path)
    print(json.dumps({"ok": not failures, "manifest": str(manifest_path), "generated_count": len(generated), "failed_count": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
