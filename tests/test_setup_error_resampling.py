from __future__ import annotations

import numpy as np

from x2cbct_difnet.setup_error_resampling import (
    evaluation_grid,
    resample_to_evaluation_grid,
    rescale_roi_row,
)


def _manifest(size: int = 256) -> dict[str, object]:
    return {
        "processed_size_xyz": [size, size, size],
        "processed_shape_zyx": [size, size, size],
        "processed_spacing_xyz": [269.0 / 255.0, 269.0 / 255.0, 263.0 / 255.0],
        "processed_origin_xyz": [-135.0, -135.0, -131.0],
        "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }


def test_evaluation_grid_preserves_endpoint_physical_extent() -> None:
    grid = evaluation_grid(_manifest(), 128)

    assert grid.size_xyz == (128, 128, 128)
    assert np.allclose(grid.spacing_xyz, (269.0 / 127.0, 269.0 / 127.0, 263.0 / 127.0))
    assert np.allclose(
        np.asarray(grid.spacing_xyz) * 127.0,
        np.asarray([269.0, 269.0, 263.0]),
    )
    assert grid.origin_xyz == (-135.0, -135.0, -131.0)


def test_resample_to_evaluation_grid_keeps_endpoint_samples() -> None:
    manifest = {
        "processed_size_xyz": [5, 5, 5],
        "processed_shape_zyx": [5, 5, 5],
        "processed_spacing_xyz": [1.0, 1.0, 1.0],
        "processed_origin_xyz": [0.0, 0.0, 0.0],
        "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    z, y, x = np.meshgrid(
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
        indexing="ij",
    )
    array = x + 10.0 * y + 100.0 * z

    output, output_manifest = resample_to_evaluation_grid(array, manifest, 3)

    assert output.shape == (3, 3, 3)
    assert np.allclose(output_manifest["processed_spacing_xyz"], [2.0, 2.0, 2.0])
    assert output[0, 0, 0] == 0.0
    assert output[-1, -1, -1] == 444.0
    assert output[1, 1, 1] == 222.0


def test_roi_indices_and_crop_margin_are_scaled_to_128_grid() -> None:
    row = {
        "x_min": "78",
        "x_max": "190",
        "y_min": "116",
        "y_max": "244",
        "z_min": "34",
        "z_max": "228",
        "threshold_low": "700.0",
        "threshold_high": "1700.0",
        "status": "accepted",
        "note": "",
        "mask_voxel_count": "1411007",
    }

    scaled = rescale_roi_row(
        row,
        source_size_xyz=(256, 256, 256),
        target_size_xyz=(128, 128, 128),
    )

    assert scaled["x_min"] == 38
    assert scaled["x_max"] == 95
    assert scaled["y_min"] == 57
    assert scaled["y_max"] == 122
    assert scaled["z_min"] == 16
    assert scaled["z_max"] == 114
    assert scaled["crop_margin_x_voxels"] == 6
    assert scaled["crop_margin_y_voxels"] == 6
    assert scaled["crop_margin_z_voxels"] == 6
