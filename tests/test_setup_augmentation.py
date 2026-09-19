from __future__ import annotations

import numpy as np
import pytest

from x2cbct_difnet.setup_augmentation import (
    DEFAULT_SETUP_ROTATION_DEG,
    DEFAULT_SETUP_TRANSLATION_MM,
    bin_label,
    iter_augmentations,
    setup_augmentation_config,
    sample_transformed_volume_setup,
    shift_image_vertical,
    transform_volume_setup,
    volume_background_value,
)


def test_setup_augmentation_defaults_and_labels() -> None:
    config = setup_augmentation_config(enabled=True)

    assert config.translation_mm == DEFAULT_SETUP_TRANSLATION_MM
    assert config.rotation_deg == DEFAULT_SETUP_ROTATION_DEG
    assert config.combination_count == 9
    assert [item.id for item in iter_augmentations(config)[:3]] == [
        "si_m10_rot_m10",
        "si_m10_rot_0",
        "si_m10_rot_p10",
    ]
    assert bin_label(-10) == "m10"
    assert bin_label(0) == "0"
    assert bin_label(10) == "p10"


def test_setup_augmentation_rejects_missing_zero_and_non_integer_bins() -> None:
    with pytest.raises(ValueError, match="must include 0"):
        setup_augmentation_config(enabled=True, translation_mm=["-4", "4"], rotation_deg=["0"])

    with pytest.raises(ValueError, match="only accepts integer"):
        setup_augmentation_config(enabled=True, translation_mm=["0.5"], rotation_deg=["0"])


def test_setup_augmentation_shift_helpers_preserve_shape() -> None:
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 1.0
    shifted = transform_volume_setup(
        volume,
        translation_mm=2,
        rotation_deg=0,
        spacing_z_mm=1.0,
        fill_value=0.0,
    )
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = 1.0
    shifted_image = shift_image_vertical(image, -1.0, fill_value=0.0)

    assert shifted.shape == volume.shape
    assert shifted[4, 2, 2] == pytest.approx(1.0)
    assert shifted_image.shape == image.shape
    assert shifted_image[1, 2] == pytest.approx(1.0)


def test_setup_augmentation_preserves_fractional_transform_values() -> None:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[4, 4, 6] = 1.0

    transformed = transform_volume_setup(
        volume,
        translation_mm=0.625,
        rotation_deg=8.1234,
        spacing_z_mm=1.0,
        fill_value=0.0,
    )

    assert transformed.shape == volume.shape
    assert transformed.dtype == np.float32
    assert not np.array_equal(transformed, volume)


def test_sample_transformed_volume_setup_matches_materialized_transform() -> None:
    z, y, x = np.meshgrid(
        np.arange(7, dtype=np.float32),
        np.arange(7, dtype=np.float32),
        np.arange(7, dtype=np.float32),
        indexing="ij",
    )
    volume = z * 100.0 + y * 10.0 + x
    fill_value = volume_background_value(volume)
    transformed = transform_volume_setup(
        volume,
        translation_mm=2,
        rotation_deg=6,
        spacing_z_mm=1.0,
        fill_value=fill_value,
    )
    indices = np.asarray(
        [
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 3, 1],
        ],
        dtype=np.int64,
    )

    sampled = sample_transformed_volume_setup(
        volume,
        indices,
        translation_mm=2,
        rotation_deg=6,
        spacing_z_mm=1.0,
        fill_value=fill_value,
    )

    assert sampled == pytest.approx(transformed[indices[:, 0], indices[:, 1], indices[:, 2]], abs=2e-4)
