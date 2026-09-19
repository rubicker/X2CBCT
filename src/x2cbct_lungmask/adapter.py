from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from x2cbct_difnet.volume_grid import resized_spacing_preserve_fov_xyz


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PreparedVolume:
    hu_zyx: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    source_shape_zyx: tuple[int, int, int]
    target_shape_zyx: tuple[int, int, int]

    def to_sitk(self) -> sitk.Image:
        image = sitk.GetImageFromArray(np.asarray(self.hu_zyx, dtype=np.float32))
        image.SetSpacing(self.spacing_xyz)
        image.SetOrigin(self.origin_xyz)
        image.SetDirection(self.direction)
        return image


def _tuple_int(values: Sequence[Any], *, length: int, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != length or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain {length} positive integers, got {result}")
    return result


def _tuple_float(values: Sequence[Any], *, length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(np.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values, got {result}")
    return result


def resize_linear_zyx(volume: np.ndarray, target_shape_zyx: Sequence[int]) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D ZYX volume, got shape {array.shape}")
    target_shape = _tuple_int(target_shape_zyx, length=3, name="target_shape_zyx")
    if tuple(array.shape) == target_shape:
        return np.ascontiguousarray(array, dtype=np.float32)
    factors = [target_shape[index] / array.shape[index] for index in range(3)]
    resized = scipy.ndimage.zoom(array, factors, order=1, prefilter=False)
    if tuple(resized.shape) != target_shape:
        raise RuntimeError(f"resize produced {resized.shape}, expected {target_shape}")
    return np.ascontiguousarray(resized, dtype=np.float32)


def normalized_to_hu(volume: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    if not normalization:
        raise ValueError("normalization metadata is required for normalized reconstruction volumes")
    hu_min = float(normalization["hu_min"])
    hu_max = float(normalization["hu_max"])
    if not np.isfinite(hu_min) or not np.isfinite(hu_max) or hu_max <= hu_min:
        raise ValueError(f"invalid HU range: [{hu_min}, {hu_max}]")
    array = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("volume contains NaN or infinite values")
    clipped = np.clip(array, 0.0, 1.0)
    return (clipped * (hu_max - hu_min) + hu_min).astype(np.float32)


def prepare_volume(
    volume: np.ndarray,
    *,
    source_shape_zyx: Sequence[int],
    source_spacing_xyz: Sequence[float],
    target_shape_zyx: Sequence[int] = (128, 128, 128),
    source_origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    source_direction: Sequence[float] = IDENTITY_DIRECTION,
    normalization: Mapping[str, Any] | None = None,
    value_space: str = "normalized",
) -> PreparedVolume:
    array = np.asarray(volume)
    source_shape = _tuple_int(source_shape_zyx, length=3, name="source_shape_zyx")
    target_shape = _tuple_int(target_shape_zyx, length=3, name="target_shape_zyx")
    spacing = _tuple_float(source_spacing_xyz, length=3, name="source_spacing_xyz")
    origin = _tuple_float(source_origin_xyz, length=3, name="source_origin_xyz")
    direction = _tuple_float(source_direction, length=9, name="source_direction")

    if tuple(array.shape) not in {source_shape, target_shape}:
        raise ValueError(
            f"volume shape {tuple(array.shape)} matches neither source grid {source_shape} "
            f"nor target grid {target_shape}"
        )

    resized = resize_linear_zyx(array, target_shape)
    if value_space == "normalized":
        if normalization is None:
            raise ValueError("normalization metadata is required when value_space='normalized'")
        hu = normalized_to_hu(resized, normalization)
    elif value_space == "hu":
        hu = np.asarray(resized, dtype=np.float32)
        if not np.isfinite(hu).all():
            raise ValueError("HU volume contains NaN or infinite values")
    else:
        raise ValueError(f"unsupported value_space: {value_space}")

    target_spacing = resized_spacing_preserve_fov_xyz(source_shape, spacing, target_shape)
    return PreparedVolume(
        hu_zyx=np.ascontiguousarray(hu, dtype=np.float32),
        spacing_xyz=tuple(float(value) for value in target_spacing),
        origin_xyz=origin,
        direction=direction,
        source_shape_zyx=source_shape,
        target_shape_zyx=target_shape,
    )
