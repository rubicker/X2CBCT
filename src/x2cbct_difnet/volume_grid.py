from __future__ import annotations

from typing import Any

import cv2
import numpy as np


AXIS_CONVENTION: dict[str, str] = {
    "array_axis": "zyx",
    "metadata_axis": "xyz",
    "simpleitk_spacing_axis": "xyz",
    "simpleitk_size_axis": "xyz",
}


def axis_convention() -> dict[str, str]:
    return dict(AXIS_CONVENTION)


def shape_zyx_to_size_xyz(shape_zyx: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    z, y, x = [int(item) for item in shape_zyx]
    return x, y, z


def size_xyz_to_shape_zyx(size_xyz: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    x, y, z = [int(item) for item in size_xyz]
    return z, y, x


def physical_fov_mm_xyz(
    size_xyz: tuple[int, int, int] | list[int],
    spacing_xyz: tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    return tuple(
        float(max(int(size) - 1, 0) * float(spacing))
        for size, spacing in zip(size_xyz, spacing_xyz)
    )


def resized_spacing_preserve_fov_xyz(
    raw_shape_zyx: tuple[int, int, int] | list[int],
    raw_spacing_xyz: tuple[float, float, float] | list[float],
    target_shape_zyx: tuple[int, int, int] | list[int],
) -> tuple[float, float, float]:
    raw_size_xyz = shape_zyx_to_size_xyz(raw_shape_zyx)
    target_size_xyz = shape_zyx_to_size_xyz(target_shape_zyx)
    raw_fov = physical_fov_mm_xyz(raw_size_xyz, raw_spacing_xyz)
    return tuple(
        float(raw_fov[axis] / max(int(target_size_xyz[axis]) - 1, 1))
        for axis in range(3)
    )


def volume_grid_payload(
    *,
    raw_shape_zyx: tuple[int, int, int] | list[int],
    raw_spacing_xyz: tuple[float, float, float] | list[float],
    raw_origin_xyz: tuple[float, float, float] | list[float],
    raw_direction: tuple[float, ...] | list[float],
    processed_shape_zyx: tuple[int, int, int] | list[int],
    processed_spacing_xyz: tuple[float, float, float] | list[float],
    processed_origin_xyz: tuple[float, float, float] | list[float],
    processed_direction: tuple[float, ...] | list[float],
) -> dict[str, Any]:
    raw_size_xyz = shape_zyx_to_size_xyz(raw_shape_zyx)
    processed_size_xyz = shape_zyx_to_size_xyz(processed_shape_zyx)
    return {
        "axis_convention": axis_convention(),
        "raw_shape_zyx": [int(item) for item in raw_shape_zyx],
        "raw_size_xyz": [int(item) for item in raw_size_xyz],
        "raw_spacing_xyz": [float(item) for item in raw_spacing_xyz],
        "raw_origin_xyz": [float(item) for item in raw_origin_xyz],
        "raw_direction": [float(item) for item in raw_direction],
        "raw_physical_fov_mm_xyz": [float(item) for item in physical_fov_mm_xyz(raw_size_xyz, raw_spacing_xyz)],
        "processed_shape_zyx": [int(item) for item in processed_shape_zyx],
        "processed_size_xyz": [int(item) for item in processed_size_xyz],
        "processed_spacing_xyz": [float(item) for item in processed_spacing_xyz],
        "processed_origin_xyz": [float(item) for item in processed_origin_xyz],
        "processed_direction": [float(item) for item in processed_direction],
        "processed_physical_fov_mm_xyz": [
            float(item) for item in physical_fov_mm_xyz(processed_size_xyz, processed_spacing_xyz)
        ],
    }


def processed_grid_from_case_manifest(case_manifest: dict[str, Any]) -> dict[str, Any]:
    shape_zyx = case_manifest.get("processed_shape_zyx", case_manifest.get("target_shape_zyx", case_manifest.get("shape_zyx")))
    spacing_xyz = case_manifest.get(
        "processed_spacing_xyz",
        case_manifest.get("target_spacing_xyz", case_manifest.get("spacing_xyz")),
    )
    origin_xyz = case_manifest.get("processed_origin_xyz", case_manifest.get("origin_xyz", [0.0, 0.0, 0.0]))
    direction = case_manifest.get(
        "processed_direction",
        case_manifest.get("direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    if shape_zyx is None or spacing_xyz is None:
        raise ValueError("case manifest is missing processed shape/spacing")
    return {
        "shape_zyx": [int(item) for item in shape_zyx],
        "size_xyz": [int(item) for item in shape_zyx_to_size_xyz(shape_zyx)],
        "spacing_xyz": [float(item) for item in spacing_xyz],
        "origin_xyz": [float(item) for item in origin_xyz],
        "direction": [float(item) for item in direction],
    }


def plane_spacing(
    plane: str,
    spacing_xyz: tuple[float, float, float] | list[float],
) -> tuple[float, float]:
    spacing_x, spacing_y, spacing_z = [float(item) for item in spacing_xyz]
    if plane == "axial":
        return spacing_y, spacing_x
    if plane == "coronal":
        return spacing_z, spacing_x
    if plane == "sagittal":
        return spacing_z, spacing_y
    raise ValueError(f"unknown plane: {plane}")


def resize_plane_to_spacing(
    panel: np.ndarray,
    plane: str,
    spacing_xyz: tuple[float, float, float] | list[float],
    *,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    row_spacing, column_spacing = plane_spacing(plane, spacing_xyz)
    base = max(min(row_spacing, column_spacing), 1e-8)
    target_h = max(1, int(round(panel.shape[0] * row_spacing / base)))
    target_w = max(1, int(round(panel.shape[1] * column_spacing / base)))
    if (target_h, target_w) == tuple(panel.shape[:2]):
        return panel
    return cv2.resize(panel, (target_w, target_h), interpolation=interpolation)


def is_isotropic_spacing(spacing_xyz: tuple[float, float, float] | list[float], eps: float = 1e-5) -> bool:
    sx, sy, sz = [float(item) for item in spacing_xyz]
    return abs(sx - sy) <= eps and abs(sx - sz) <= eps
