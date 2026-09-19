from __future__ import annotations

from pathlib import Path

import numpy as np

from x2cbct_difnet.calibration import load_calibration_report
from x2cbct_difnet.dataset import DifNetV0bDataset
from x2cbct_difnet.env import check_environment
from x2cbct_difnet.geometry import ClinicalGeometryAdapter
from x2cbct_difnet.manifest import load_fixed_view_manifest, load_split_manifest
from x2cbct_difnet.official import run_official_forward_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_root_environment_has_v0_dependencies() -> None:
    report = check_environment(PROJECT_ROOT)

    assert report.python_executable.endswith(".conda\\x2cbct-v0\\python.exe")
    assert report.modules["torch"].available
    assert report.modules["SimpleITK"].available
    assert report.modules["yaml"].available
    assert report.modules["cv2"].available


def test_v0b_manifests_preserve_fixed_view_and_patient_split_contract() -> None:
    fixed = load_fixed_view_manifest(PROJECT_ROOT)
    split = load_split_manifest(PROJECT_ROOT)

    assert fixed.included_case_count == 272
    assert fixed.excluded_case_count == 9
    assert len(fixed.records) == 272
    assert len(fixed.records[0].selected["view_2"]) == 2
    assert split.train.case_count == 218
    assert split.test.case_count == 54
    assert split.patient_overlap == set()


def test_clinical_geometry_adapter_projects_points_with_calibration_metadata() -> None:
    report = load_calibration_report(PROJECT_ROOT)
    adapter = ClinicalGeometryAdapter(report)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [25.0, -10.0, 5.0],
            [-30.0, 15.0, -8.0],
        ],
        dtype=np.float32,
    )

    projected = adapter.project_points_to_grid(
        points,
        gantry_angle_deg=199.814422607,
        detector_shape=(512, 512),
        detector_spacing=(0.8, 0.8),
        feature_shape=(64, 64),
    )

    assert projected.grid.shape == (3, 2)
    assert projected.valid_mask.shape == (3,)
    assert np.isfinite(projected.grid).all()
    assert projected.metadata["candidate_id"] == report.candidate_id
    assert projected.metadata["adapter_version"] == report.adapter_version


def test_v0b_dataset_item_matches_official_difnet_shape_contract() -> None:
    dataset = DifNetV0bDataset(
        PROJECT_ROOT,
        split="train",
        view_key="view_2",
        num_points=16,
        projection_size=64,
        seed=20260522,
    )

    item = dataset[0]

    assert item["projs"].shape == (2, 1, 64, 64)
    assert item["points"].shape == (16, 3)
    assert item["proj_points"].shape == (2, 16, 2)
    assert item["p_gt"].shape == (1, 16)
    assert np.isfinite(item["projs"]).all()
    assert np.isfinite(item["points"]).all()
    assert np.isfinite(item["proj_points"]).all()
    assert np.isfinite(item["p_gt"]).all()
    assert item["trace"]["view_key"] == "view_2"
    assert len(item["trace"]["selected_projections"]) == 2


def test_official_difnet_forward_smoke_runs_on_synthetic_item() -> None:
    report = run_official_forward_smoke(
        PROJECT_ROOT,
        num_views=2,
        image_size=64,
        num_points=32,
        mid_channels=8,
        combine="mlp",
    )

    assert report["output_shape"] == [1, 1, 32]
    assert report["target_shape"] == [1, 1, 32]
    assert report["loss"] >= 0.0
