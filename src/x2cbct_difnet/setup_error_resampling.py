from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class EvaluationGrid:
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    physical_fov_mm_xyz: tuple[float, float, float]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(reversed(self.size_xyz))


def _triple(
    manifest: Mapping[str, Any],
    key: str,
    *,
    cast: type[int] | type[float],
) -> tuple[int, int, int] | tuple[float, float, float]:
    values = manifest.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"case manifest has no valid {key}")
    return tuple(cast(value) for value in values)


def evaluation_grid(
    case_manifest: Mapping[str, Any],
    grid_size: int,
) -> EvaluationGrid:
    if int(grid_size) < 2:
        raise ValueError("evaluation grid size must be >= 2")
    source_size = _triple(case_manifest, "processed_size_xyz", cast=int)
    source_spacing = _triple(case_manifest, "processed_spacing_xyz", cast=float)
    source_origin = _triple(case_manifest, "processed_origin_xyz", cast=float)
    direction_values = case_manifest.get("processed_direction")
    if not isinstance(direction_values, (list, tuple)) or len(direction_values) != 9:
        raise ValueError("case manifest has no valid processed_direction")
    direction = tuple(float(value) for value in direction_values)
    target_size = (int(grid_size),) * 3
    physical_fov = tuple(
        float(spacing) * float(size - 1)
        for size, spacing in zip(source_size, source_spacing)
    )
    target_spacing = tuple(
        extent / float(target - 1)
        for extent, target in zip(physical_fov, target_size)
    )
    return EvaluationGrid(
        size_xyz=target_size,
        spacing_xyz=target_spacing,
        origin_xyz=source_origin,
        direction=direction,
        physical_fov_mm_xyz=physical_fov,
    )


def manifest_for_evaluation_grid(
    case_manifest: Mapping[str, Any],
    grid_size: int,
) -> dict[str, Any]:
    grid = evaluation_grid(case_manifest, grid_size)
    result = copy.deepcopy(dict(case_manifest))
    result.update(
        {
            "processed_shape_zyx": list(grid.shape_zyx),
            "processed_size_xyz": list(grid.size_xyz),
            "processed_spacing_xyz": list(grid.spacing_xyz),
            "processed_origin_xyz": list(grid.origin_xyz),
            "processed_direction": list(grid.direction),
            "processed_physical_fov_mm_xyz": list(grid.physical_fov_mm_xyz),
            "shape_zyx": list(grid.shape_zyx),
            "spacing_xyz": list(grid.spacing_xyz),
            "origin_xyz": list(grid.origin_xyz),
            "direction": list(grid.direction),
            "target_size": int(grid_size),
        }
    )
    preprocess = copy.deepcopy(result.get("preprocess", {}))
    preprocess.update(
        {
            "evaluation_grid_policy": "endpoint_aligned_preserve_physical_extent",
            "evaluation_grid_size": int(grid_size),
            "evaluation_grid_interpolator": "linear",
        }
    )
    result["preprocess"] = preprocess
    return result


def _image_from_array(
    array_zyx: np.ndarray,
    case_manifest: Mapping[str, Any],
) -> sitk.Image:
    source_size = _triple(case_manifest, "processed_size_xyz", cast=int)
    expected_shape = tuple(reversed(source_size))
    array = np.asarray(array_zyx, dtype=np.float32)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"CBCT array shape {tuple(array.shape)} does not match manifest "
            f"processed size {source_size}"
        )
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(
        tuple(
            float(value)
            for value in _triple(
                case_manifest,
                "processed_spacing_xyz",
                cast=float,
            )
        )
    )
    image.SetOrigin(
        tuple(
            float(value)
            for value in _triple(
                case_manifest,
                "processed_origin_xyz",
                cast=float,
            )
        )
    )
    direction = case_manifest.get("processed_direction")
    if not isinstance(direction, (list, tuple)) or len(direction) != 9:
        raise ValueError("case manifest has no valid processed_direction")
    image.SetDirection(tuple(float(value) for value in direction))
    return image


def resample_to_evaluation_grid(
    array_zyx: np.ndarray,
    case_manifest: Mapping[str, Any],
    grid_size: int,
    *,
    default_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_manifest = manifest_for_evaluation_grid(case_manifest, grid_size)
    source = _image_from_array(array_zyx, case_manifest)
    target = sitk.Image([int(grid_size)] * 3, sitk.sitkFloat32)
    target.SetSpacing(
        tuple(float(value) for value in target_manifest["processed_spacing_xyz"])
    )
    target.SetOrigin(
        tuple(float(value) for value in target_manifest["processed_origin_xyz"])
    )
    target.SetDirection(
        tuple(float(value) for value in target_manifest["processed_direction"])
    )
    resampled = sitk.Resample(
        source,
        target,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        float(default_value),
        sitk.sitkFloat32,
    )
    return (
        sitk.GetArrayFromImage(resampled).astype(np.float32),
        target_manifest,
    )


def rescale_roi_row(
    row: Mapping[str, Any],
    *,
    source_size_xyz: tuple[int, int, int],
    target_size_xyz: tuple[int, int, int],
) -> dict[str, Any]:
    result = dict(row)
    volume_scale = 1.0
    for axis, source_size, target_size in zip(
        ("x", "y", "z"),
        source_size_xyz,
        target_size_xyz,
    ):
        if source_size < 2 or target_size < 2:
            raise ValueError("ROI resampling requires all grid dimensions >= 2")
        scale = float(target_size - 1) / float(source_size - 1)
        volume_scale *= scale
        source_min = int(row[f"{axis}_min"])
        source_max = int(row[f"{axis}_max"])
        result[f"{axis}_min"] = max(
            0,
            min(target_size - 1, int(math.floor(source_min * scale))),
        )
        result[f"{axis}_max"] = max(
            0,
            min(target_size - 1, int(math.ceil(source_max * scale))),
        )
        result[f"crop_margin_{axis}_voxels"] = max(
            1,
            int(round(12.0 * scale)),
        )

    result["box_volume_voxels"] = int(
        (int(result["x_max"]) - int(result["x_min"]) + 1)
        * (int(result["y_max"]) - int(result["y_min"]) + 1)
        * (int(result["z_max"]) - int(result["z_min"]) + 1)
    )
    if row.get("mask_voxel_count") not in (None, ""):
        result["mask_voxel_count"] = max(
            1,
            int(round(float(row["mask_voxel_count"]) * volume_scale)),
        )
    note = str(row.get("note", "")).strip()
    suffix = (
        "ROI indices rescaled for endpoint-aligned evaluation grid "
        f"{source_size_xyz}->{target_size_xyz}"
    )
    result["note"] = f"{note}; {suffix}" if note else suffix
    return result
