from __future__ import annotations

import numpy as np
import pytest

from x2cbct_difnet.translation_probe import (
    common_z_extent_for_shift,
    detector_pixel_shift_for_si,
    edge_background_value,
    percentile_window,
    shift_projection_vertical,
    shift_volume_head,
    to_uint8_window,
    volume_background_value,
)


def test_si_shift_uses_sid_sdd_magnification() -> None:
    meta = detector_pixel_shift_for_si(8.0, sid_mm=1000.0, sdd_mm=1536.0, detector_spacing_y_mm=0.8)

    assert meta["magnification"] == pytest.approx(1.536)
    assert meta["detector_shift_mm"] == pytest.approx(-12.288)
    assert meta["raw_pixel_shift_y"] == pytest.approx(-15.36)


def test_projection_positive_head_shift_moves_content_up() -> None:
    projection = np.zeros((9, 7), dtype=np.float32)
    projection[5, 2:5] = 10.0

    shifted = shift_projection_vertical(projection, -2.0, fill_value=0.0)

    assert shifted[3, 2:5].max() == pytest.approx(10.0)
    assert shifted[5, 2:5].max() == pytest.approx(0.0)


def test_volume_positive_head_shift_moves_content_to_larger_z_index() -> None:
    volume = np.full((7, 5, 5), -1024.0, dtype=np.float32)
    volume[2, 2, 2] = 100.0

    shifted = shift_volume_head(volume, 2.0, spacing_z_mm=1.0)

    assert shifted[4, 2, 2] == pytest.approx(100.0)
    assert shifted[2, 2, 2] == pytest.approx(-1024.0)


def test_edge_background_uses_projection_border_median() -> None:
    projection = np.ones((5, 5), dtype=np.float32) * 3.0
    projection[2, 2] = 99.0

    assert edge_background_value(projection) == pytest.approx(3.0)


def test_volume_background_uses_volume_border_median() -> None:
    volume = np.ones((5, 5, 5), dtype=np.float32) * 7.0
    volume[2, 2, 2] = 999.0

    assert volume_background_value(volume) == pytest.approx(7.0)


def test_common_z_extent_for_positive_shift_removes_new_padding() -> None:
    original_slice, shifted_slice = common_z_extent_for_shift(10, 3.0)

    assert original_slice == slice(0, 7)
    assert shifted_slice == slice(3, 10)


def test_common_z_extent_for_negative_shift_removes_new_padding() -> None:
    original_slice, shifted_slice = common_z_extent_for_shift(10, -3.0)

    assert original_slice == slice(3, 10)
    assert shifted_slice == slice(0, 7)


def test_shared_display_window_prevents_air_fill_from_brightening_shifted_cbct() -> None:
    original = np.full((20, 20), 500.0, dtype=np.float32)
    original[4:16, 4:16] = 1000.0
    shifted = original.copy()
    shifted[:4, :] = -1024.0

    independent = to_uint8_window(shifted, percentile_window(shifted))
    shared = to_uint8_window(shifted, percentile_window(original))

    assert independent.mean() > shared.mean()
    assert shared.mean() <= to_uint8_window(original, percentile_window(original)).mean()
