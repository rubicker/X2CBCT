from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .cache import write_json
from .normalization import LEGACY_CBCT_NORMALIZATION, cbct_normalization_profile, normalize_cbct_hu
from .normalization_profile import deterministic_sample, parse_percentiles, profile_from_values
from .volume_grid import axis_convention, volume_grid_payload


IDENTITY_DIRECTION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@dataclass(frozen=True)
class CtOnlyPreprocessConfig:
    dataset_name: str = "ct_only"
    geometry_branch: str = "simulated_drr"
    resample_spacing_xyz: tuple[float, float, float] = (1.6, 1.6, 1.6)
    target_shape_zyx: tuple[int, int, int] = (128, 128, 128)
    crop_pad_policy: str = "center"
    pad_value_hu: float = -1024.0
    hu_window_mode: str = "percentile"
    hu_percentiles: tuple[float, float] = (0.5, 99.5)
    hu_window_split: str = "train"
    hu_sample_voxels_per_case: int = 200_000

    def __post_init__(self) -> None:
        if self.geometry_branch != "simulated_drr":
            raise ValueError("CT-only preprocessing only supports geometry_branch=simulated_drr")
        if self.crop_pad_policy != "center":
            raise ValueError("only center crop/pad is supported")
        if len(self.resample_spacing_xyz) != 3 or len(self.target_shape_zyx) != 3:
            raise ValueError("spacing and shape must have exactly 3 values")
        if any(float(item) <= 0 for item in self.resample_spacing_xyz):
            raise ValueError("resample spacing must be positive")
        if any(int(item) <= 0 for item in self.target_shape_zyx):
            raise ValueError("target shape must be positive")


def parse_triplet(text: str, *, value_type: type = float) -> tuple[Any, Any, Any]:
    values = [value_type(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {text!r}")
    return values[0], values[1], values[2]


def load_config(path: Path | None, overrides: dict[str, Any] | None = None) -> CtOnlyPreprocessConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with path.open("r", encoding="utf-8-sig") as handle:
            data.update(json.load(handle))
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    if "resample_spacing_xyz" in data:
        data["resample_spacing_xyz"] = tuple(float(item) for item in data["resample_spacing_xyz"])
    if "target_shape_zyx" in data:
        data["target_shape_zyx"] = tuple(int(item) for item in data["target_shape_zyx"])
    if "hu_percentiles" in data:
        data["hu_percentiles"] = parse_percentiles(data["hu_percentiles"])
    return CtOnlyPreprocessConfig(**data)


def load_registry_rows(registry_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with registry_path.open("r", encoding="utf-8-sig") as handle:
        registry = json.load(handle)
    rows = list(registry.get("cases", []))
    if not rows:
        raise ValueError(f"registry has no cases: {registry_path}")
    return registry, rows


def resolve_registry_path(value: str | Path, registry_path: Path, registry: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    raw_root = registry.get("raw_root")
    if raw_root:
        candidate = Path(raw_root) / path
        if candidate.exists():
            return candidate
    return registry_path.parent / path


def resample_image_to_spacing(
    image: sitk.Image,
    spacing_xyz: tuple[float, float, float],
    *,
    default_value: float,
) -> sitk.Image:
    raw_size_xyz = [int(item) for item in image.GetSize()]
    raw_spacing_xyz = [float(item) for item in image.GetSpacing()]
    output_size_xyz = [
        max(1, int(round(raw_size_xyz[axis] * raw_spacing_xyz[axis] / float(spacing_xyz[axis]))))
        for axis in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(float(item) for item in spacing_xyz))
    resampler.SetSize(output_size_xyz)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def center_crop_pad_zyx(
    array: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    *,
    pad_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(array, dtype=np.float32)
    output = np.full(tuple(int(item) for item in target_shape_zyx), float(pad_value), dtype=np.float32)
    src_slices = []
    dst_slices = []
    source_start = []
    source_end = []
    pad_before = []
    pad_after = []
    for axis, target_size in enumerate(target_shape_zyx):
        source_size = int(arr.shape[axis])
        target_size = int(target_size)
        if source_size >= target_size:
            src_start = (source_size - target_size) // 2
            dst_start = 0
            length = target_size
        else:
            src_start = 0
            dst_start = (target_size - source_size) // 2
            length = source_size
        src_slices.append(slice(src_start, src_start + length))
        dst_slices.append(slice(dst_start, dst_start + length))
        source_start.append(int(src_start))
        source_end.append(int(src_start + length))
        pad_before.append(int(dst_start))
        pad_after.append(int(target_size - dst_start - length))
    output[tuple(dst_slices)] = arr[tuple(src_slices)]
    trace = {
        "policy": "center",
        "input_shape_zyx": [int(item) for item in arr.shape],
        "output_shape_zyx": [int(item) for item in output.shape],
        "source_start_zyx": source_start,
        "source_end_zyx": source_end,
        "pad_before_zyx": pad_before,
        "pad_after_zyx": pad_after,
        "pad_value_hu": float(pad_value),
    }
    return output, trace


def _case_id(row: dict[str, Any], case_id_key: str) -> str:
    value = row.get(case_id_key, row.get("case_id", row.get("seriesuid", row.get("patient_id"))))
    if value is None:
        raise ValueError(f"row is missing case id key {case_id_key!r}")
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return cleaned.strip("_") or "case"


def preprocess_case_to_hu(
    row: dict[str, Any],
    *,
    registry: dict[str, Any],
    registry_path: Path,
    output_root: Path,
    config: CtOnlyPreprocessConfig,
    image_path_key: str = "ct_path",
    case_id_key: str = "seriesuid",
) -> dict[str, Any]:
    if image_path_key not in row:
        raise ValueError(f"row is missing image path key {image_path_key!r}")
    case_id = _case_id(row, case_id_key)
    image_path = resolve_registry_path(row[image_path_key], registry_path, registry)
    image = sitk.ReadImage(str(image_path))
    raw_array = sitk.GetArrayFromImage(image).astype(np.float32)
    resampled = resample_image_to_spacing(
        image,
        config.resample_spacing_xyz,
        default_value=config.pad_value_hu,
    )
    resampled_array = sitk.GetArrayFromImage(resampled).astype(np.float32)
    processed_hu, crop_pad = center_crop_pad_zyx(
        resampled_array,
        config.target_shape_zyx,
        pad_value=config.pad_value_hu,
    )
    case_dir = output_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    hu_path = case_dir / "ct_hu.npy"
    np.save(hu_path, processed_hu.astype(np.float32))
    raw_spacing_xyz = tuple(float(item) for item in image.GetSpacing())
    raw_origin_xyz = tuple(float(item) for item in image.GetOrigin())
    raw_direction = [float(item) for item in image.GetDirection()]
    grid_payload = volume_grid_payload(
        raw_shape_zyx=raw_array.shape,
        raw_spacing_xyz=raw_spacing_xyz,
        raw_origin_xyz=raw_origin_xyz,
        raw_direction=raw_direction,
        processed_shape_zyx=config.target_shape_zyx,
        processed_spacing_xyz=config.resample_spacing_xyz,
        processed_origin_xyz=[0.0, 0.0, 0.0],
        processed_direction=raw_direction or IDENTITY_DIRECTION,
    )
    split = str(row.get("split", "all"))
    payload = {
        **row,
        "status": "hu_preprocessed",
        "dataset_name": config.dataset_name,
        "case_id": case_id,
        "split": split,
        "geometry_branch": config.geometry_branch,
        **grid_payload,
        "raw_image_path": str(image_path.resolve()),
        "processed_hu_path": str(hu_path.resolve()),
        "processed_cbct_path": str((case_dir / "ct_norm.npy").resolve()),
        "shape_zyx": [int(item) for item in config.target_shape_zyx],
        "spacing_xyz": [float(item) for item in config.resample_spacing_xyz],
        "origin_xyz": [0.0, 0.0, 0.0],
        "direction": raw_direction or IDENTITY_DIRECTION,
        "preprocess": {
            "policy": "spacing_aware_resample_then_center_crop_pad",
            "resample_spacing_xyz": [float(item) for item in config.resample_spacing_xyz],
            "target_shape_zyx": [int(item) for item in config.target_shape_zyx],
            "crop_pad_policy": config.crop_pad_policy,
            "crop_pad": crop_pad,
            "canonical_center_is_isocenter": True,
            "raw_origin_used_for_training_geometry": False,
            "resampler": "SimpleITK.ResampleImageFilter.linear",
        },
        "hu_stats_after_preprocess": {
            "min": float(np.nanmin(processed_hu)),
            "max": float(np.nanmax(processed_hu)),
            "mean": float(np.nanmean(processed_hu)),
        },
    }
    write_json(payload, case_dir / "case_manifest.json")
    return payload


def estimate_profile_from_processed_cases(
    cases: list[dict[str, Any]],
    config: CtOnlyPreprocessConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.hu_window_mode == "legacy":
        return dict(LEGACY_CBCT_NORMALIZATION), []
    selected = [item for item in cases if config.hu_window_split == "all" or item.get("split") == config.hu_window_split]
    if not selected:
        raise ValueError(f"no processed cases for hu_window_split={config.hu_window_split}")
    samples = []
    stats = []
    for index, item in enumerate(selected):
        arr = np.load(item["processed_hu_path"]).astype(np.float32)
        values = deterministic_sample(arr[np.isfinite(arr)], config.hu_sample_voxels_per_case, 20260527 + index)
        samples.append(values)
        stats.append(
            {
                "case_id": item["case_id"],
                "split": item.get("split"),
                "processed_shape_zyx": item["processed_shape_zyx"],
                "sampled_voxel_count": int(values.size),
                "min": float(np.min(values)) if values.size else None,
                "max": float(np.max(values)) if values.size else None,
            }
        )
    if config.hu_window_mode == "fixed":
        raise ValueError("fixed HU window must be supplied by the CLI before calling this helper")
    merged = np.concatenate([item for item in samples if item.size])
    profile = profile_from_values(
        merged,
        percentiles=config.hu_percentiles,
        profile_id=f"{config.dataset_name}_processed_p{config.hu_percentiles[0]:g}_p{config.hu_percentiles[1]:g}",
        source={
            "scope": config.dataset_name,
            "source_split": config.hu_window_split,
            "preprocess_policy": "spacing_aware_resample_then_center_crop_pad",
            "resample_spacing_xyz": list(config.resample_spacing_xyz),
            "target_shape_zyx": list(config.target_shape_zyx),
        },
    )
    return profile, stats


def finalize_normalized_cases(
    cases: list[dict[str, Any]],
    normalization: dict[str, Any],
    *,
    keep_hu_volumes: bool = True,
    progress_interval: int = 0,
) -> list[dict[str, Any]]:
    finalized = []
    total = len(cases)
    for index, item in enumerate(cases, start=1):
        hu_path = Path(item["processed_hu_path"])
        hu = np.load(hu_path).astype(np.float32)
        norm = normalize_cbct_hu(hu, normalization=normalization).astype(np.float16)
        out_path = Path(item["processed_cbct_path"])
        np.save(out_path, norm)
        if not keep_hu_volumes and hu_path.exists():
            hu_path.unlink()
        updated = {
            **item,
            "status": "preprocessed",
            "cbct_normalization": normalization,
            "processed_hu_retained": bool(keep_hu_volumes),
        }
        write_json(updated, out_path.parent / "case_manifest.json")
        finalized.append(updated)
        if progress_interval > 0 and (index == 1 or index == total or index % progress_interval == 0):
            print(
                json.dumps(
                    {"event": "preprocess_progress", "phase": "normalize", "done": index, "total": total},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return finalized


def preprocess_ct_only_dataset(
    *,
    registry_path: Path,
    output_root: Path,
    config: CtOnlyPreprocessConfig,
    image_path_key: str = "ct_path",
    case_id_key: str = "seriesuid",
    max_cases: int | None = None,
    command: str = "",
    fixed_normalization: dict[str, Any] | None = None,
    keep_hu_volumes: bool = True,
    progress_interval: int = 0,
) -> dict[str, Any]:
    registry, rows = load_registry_rows(registry_path)
    if max_cases is not None:
        rows = rows[: int(max_cases)]
    output_root.mkdir(parents=True, exist_ok=True)
    processed = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        processed.append(
            preprocess_case_to_hu(
                row,
                registry=registry,
                registry_path=registry_path,
                output_root=output_root,
                config=config,
                image_path_key=image_path_key,
                case_id_key=case_id_key,
            )
        )
        if progress_interval > 0 and (index == 1 or index == total or index % progress_interval == 0):
            print(
                json.dumps(
                    {"event": "preprocess_progress", "phase": "resample_crop", "done": index, "total": total},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if fixed_normalization is not None:
        normalization = fixed_normalization
        profile_stats = []
    else:
        normalization, profile_stats = estimate_profile_from_processed_cases(processed, config)
    finalized = finalize_normalized_cases(
        processed,
        normalization,
        keep_hu_volumes=keep_hu_volumes,
        progress_interval=progress_interval,
    )
    report = {
        "schema": "x2cbct_ct_only_preprocess_report.v1",
        "geometry_branch": config.geometry_branch,
        "axis_convention": axis_convention(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_command": command,
        "registry": str(registry_path.resolve()),
        "registry_split": registry.get("split"),
        "output_root": str(output_root.resolve()),
        "dataset_name": config.dataset_name,
        "config": {
            "resample_spacing_xyz": list(config.resample_spacing_xyz),
            "target_shape_zyx": list(config.target_shape_zyx),
            "crop_pad_policy": config.crop_pad_policy,
            "pad_value_hu": config.pad_value_hu,
            "hu_window_mode": config.hu_window_mode,
            "hu_percentiles": None if config.hu_window_mode == "fixed" else list(config.hu_percentiles),
            "hu_window_split": None if config.hu_window_mode == "fixed" else config.hu_window_split,
            "keep_hu_volumes": bool(keep_hu_volumes),
        },
        "cbct_normalization": {
            "mode": config.hu_window_mode,
            "scope": config.dataset_name,
            "source_split": config.hu_window_split,
            "profiles": {config.dataset_name: normalization},
            "case_stats": profile_stats,
        },
        "processed_count": len(finalized),
        "processed_cases": finalized,
    }
    write_json(report, output_root / "preprocess_report.json")
    return report
