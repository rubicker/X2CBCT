from __future__ import annotations

import numpy as np
import pytest

from x2cbct_lungmask.adapter import normalized_to_hu, prepare_volume
from x2cbct_lungmask.experiment import paired_comparisons
from x2cbct_lungmask.metrics import average_symmetric_surface_distance_mm, dice_coefficient


NORMALIZATION = {
    "mode": "fixed",
    "hu_min": -1000.0,
    "hu_max": 1024.0,
    "value_range": [0.0, 1.0],
}


def test_normalized_to_hu_uses_project_fixed_profile() -> None:
    volume = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

    hu = normalized_to_hu(volume, NORMALIZATION)

    assert hu.tolist() == pytest.approx([-1000.0, 12.0, 1024.0])


def test_prepare_volume_resizes_256_to_128_and_preserves_endpoint_fov() -> None:
    volume = np.zeros((256, 256, 256), dtype=np.float32)

    prepared = prepare_volume(
        volume,
        source_shape_zyx=(256, 256, 256),
        source_spacing_xyz=(1.6, 1.6, 1.6),
        target_shape_zyx=(128, 128, 128),
        normalization=NORMALIZATION,
    )

    assert prepared.hu_zyx.shape == (128, 128, 128)
    assert prepared.spacing_xyz == pytest.approx((408.0 / 127.0,) * 3)
    assert float(prepared.hu_zyx.min()) == pytest.approx(-1000.0)


def test_prepare_volume_accepts_already_resized_prediction_with_source_grid_metadata() -> None:
    prediction = np.ones((128, 128, 128), dtype=np.float32)

    prepared = prepare_volume(
        prediction,
        source_shape_zyx=(256, 256, 256),
        source_spacing_xyz=(1.6, 1.6, 1.6),
        target_shape_zyx=(128, 128, 128),
        normalization=NORMALIZATION,
    )

    assert prepared.hu_zyx.shape == (128, 128, 128)
    assert prepared.spacing_xyz == pytest.approx((408.0 / 127.0,) * 3)
    assert float(prepared.hu_zyx.max()) == pytest.approx(1024.0)


def test_dice_and_asd_are_zero_distance_for_identical_masks() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    assert dice_coefficient(mask, mask) == pytest.approx(1.0)
    assert average_symmetric_surface_distance_mm(mask, mask, spacing_xyz=(2.0, 2.0, 2.0)) == pytest.approx(0.0)


def test_asd_is_nan_when_one_lung_mask_is_empty() -> None:
    empty = np.zeros((8, 8, 8), dtype=bool)
    present = empty.copy()
    present[3:5, 3:5, 3:5] = True

    assert dice_coefficient(empty, present) == pytest.approx(0.0)
    assert np.isnan(average_symmetric_surface_distance_mm(empty, present, spacing_xyz=(1.0, 1.0, 1.0)))


def test_paired_comparison_reports_positive_improvement_for_better_primary() -> None:
    rows = []
    for case_id, primary_dice, baseline_dice, primary_asd, baseline_asd in (
        ("a", 95.0, 90.0, 1.0, 2.0),
        ("b", 96.0, 91.0, 1.2, 2.2),
    ):
        rows.extend(
            [
                {
                    "case_id": case_id,
                    "method": "primary",
                    "status": "ok",
                    "bilateral_dice_percent": primary_dice,
                    "bilateral_asd_mm": primary_asd,
                },
                {
                    "case_id": case_id,
                    "method": "baseline",
                    "status": "ok",
                    "bilateral_dice_percent": baseline_dice,
                    "bilateral_asd_mm": baseline_asd,
                },
            ]
        )

    comparisons = paired_comparisons(rows, primary_method="primary", methods=["primary", "baseline"], seed=1)

    assert len(comparisons) == 2
    assert all(row["mean_improvement"] > 0 for row in comparisons)
    assert all(row["paired_case_count"] == 2 for row in comparisons)
