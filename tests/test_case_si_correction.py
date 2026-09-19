from __future__ import annotations

import numpy as np
import pytest

from x2cbct_difnet.setup_augmentation import transform_volume_setup
from x2cbct_lungmask.si_correction import inverse_si_translation


def test_inverse_si_translation_undoes_positive_z_shift() -> None:
    volume = np.zeros((17, 9, 9), dtype=np.float32)
    volume[5:9, 3:6, 3:6] = 1.0
    augmented = transform_volume_setup(
        volume,
        translation_mm=2.0,
        rotation_deg=0.0,
        spacing_z_mm=1.0,
        fill_value=0.0,
    )

    corrected = inverse_si_translation(
        augmented,
        translation_mm=2.0,
        source_shape_zyx=volume.shape,
        source_spacing_xyz=(1.0, 1.0, 1.0),
        fill_value=0.0,
    )

    assert corrected.shift_voxels_zyx == pytest.approx((-2.0, 0.0, 0.0))
    assert corrected.volume == pytest.approx(volume)


def test_inverse_si_translation_uses_endpoint_fov_spacing_for_128_grid() -> None:
    volume = np.zeros((128, 128, 128), dtype=np.float32)

    corrected = inverse_si_translation(
        volume,
        translation_mm=10.0,
        source_shape_zyx=(256, 256, 256),
        source_spacing_xyz=(1.6, 1.6, 1.6),
        fill_value=0.0,
    )

    expected_spacing = 255.0 * 1.6 / 127.0
    assert corrected.effective_spacing_xyz == pytest.approx((expected_spacing,) * 3)
    assert corrected.shift_voxels_zyx == pytest.approx((-10.0 / expected_spacing, 0.0, 0.0))
