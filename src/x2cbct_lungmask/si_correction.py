from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.ndimage

from x2cbct_difnet.setup_augmentation import volume_background_value
from x2cbct_difnet.volume_grid import resized_spacing_preserve_fov_xyz


@dataclass(frozen=True)
class SiCorrectionResult:
    volume: np.ndarray
    effective_spacing_xyz: tuple[float, float, float]
    shift_voxels_zyx: tuple[float, float, float]
    fill_value: float


def inverse_si_translation(
    volume: np.ndarray,
    *,
    translation_mm: float,
    source_shape_zyx: Sequence[int],
    source_spacing_xyz: Sequence[float],
    fill_value: float | None = None,
) -> SiCorrectionResult:
    """Undo an S-I setup translation on a ZYX reconstruction volume.

    The setup augmentation is applied on array axis 0 (Z/S-I). A positive
    augmentation therefore requires a negative shift here. The output spacing
    is derived from the source grid by preserving endpoint-to-endpoint FOV,
    which is also the convention used by the 256-to-128 experiment adapter.
    """

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D ZYX, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("volume contains NaN or infinite values")

    source_shape = tuple(int(value) for value in source_shape_zyx)
    source_spacing = tuple(float(value) for value in source_spacing_xyz)
    if len(source_shape) != 3 or any(value <= 0 for value in source_shape):
        raise ValueError(f"invalid source_shape_zyx: {source_shape}")
    if len(source_spacing) != 3 or any(not np.isfinite(value) or value <= 0 for value in source_spacing):
        raise ValueError(f"invalid source_spacing_xyz: {source_spacing}")

    effective_spacing = resized_spacing_preserve_fov_xyz(
        source_shape,
        source_spacing,
        tuple(int(value) for value in array.shape),
    )
    spacing_z_mm = float(effective_spacing[2])
    shift_zyx = (-float(translation_mm) / spacing_z_mm, 0.0, 0.0)
    resolved_fill = volume_background_value(array) if fill_value is None else float(fill_value)
    corrected = scipy.ndimage.shift(
        array,
        shift=shift_zyx,
        order=1,
        mode="constant",
        cval=resolved_fill,
        prefilter=False,
    ).astype(np.float32)
    return SiCorrectionResult(
        volume=np.ascontiguousarray(corrected, dtype=np.float32),
        effective_spacing_xyz=tuple(float(value) for value in effective_spacing),
        shift_voxels_zyx=tuple(float(value) for value in shift_zyx),
        fill_value=resolved_fill,
    )
