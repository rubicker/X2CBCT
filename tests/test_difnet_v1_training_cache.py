from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.v1_training_cache import derive_v1_training_cache
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset
from x2cbct_difnet.point_blocks import official_lattice_metadata
from x2cbct_difnet.volume_grid import axis_convention
import x2cbct_difnet.cached_dataset as cached_dataset_module


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_real_base_cache(root: Path) -> Path:
    cbct_dir = root / "cases" / "patient1_case2"
    case_dir = root / "cases" / "patient1_case2_si_0_rot_0"
    projection_dir = root / "projections" / "patient1_case2_si_0_rot_0"
    sample_path = root / "view_10" / "train" / "patient1_case2_si_0_rot_0_view_10.npz"
    cbct_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cbct_dir / "cbct_norm.npy", np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8))
    for index in range(10):
        np.save(projection_dir / f"projection_{index:03d}.npy", np.full((8, 8), index, dtype=np.float32))
    _write_json(
        case_dir / "case_manifest.json",
        {
            "case_key": "patient1_case2_si_0_rot_0",
            "patient_id": "1",
            "case_id": "2",
            "raw_case_path": str(root / "raw" / "patient1" / "case2"),
            "cbct_payload_path": "cases/patient1_case2/cbct_norm.npy",
            "shape_zyx": [8, 8, 8],
            "processed_shape_zyx": [8, 8, 8],
            "processed_spacing_xyz": [1.0, 1.0, 1.0],
            "processed_origin_xyz": [0.0, 0.0, 0.0],
            "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "target_size": 8,
            "point_blocks": official_lattice_metadata(8),
        },
    )
    np.savez_compressed(
        sample_path,
        sample_id=np.array("patient1_case2_si_0_rot_0_view_10"),
        split=np.array("train"),
        view_key=np.array("view_10"),
        case_key=np.array("patient1_case2_si_0_rot_0"),
        patient_id=np.array("1"),
        case_id=np.array("2"),
        case_manifest_path=np.array("cases/patient1_case2_si_0_rot_0/case_manifest.json"),
        cbct_payload_path=np.array("cases/patient1_case2/cbct_norm.npy"),
        projection_payload_paths=np.array([f"projections/patient1_case2_si_0_rot_0/projection_{index:03d}.npy" for index in range(10)]),
        projection_files=np.array([f"projection_{index:03d}.npy" for index in range(10)]),
        target_angles_deg=np.array([200 + index * 15 for index in range(10)], dtype=np.float32),
        gantry_angles_deg=np.array([200 + index * 15 for index in range(10)], dtype=np.float32),
        angular_residuals_deg=np.zeros(10, dtype=np.float32),
        detector_shape=np.array([8, 8], dtype=np.int32),
        detector_spacing_yx=np.array([1.0, 1.0], dtype=np.float32),
        projection_size=np.array(8, dtype=np.int32),
        target_volume_mode=np.array("base_reference"),
        augmentation_id=np.array("si_0_rot_0"),
        translation_mm=np.array(0, dtype=np.int32),
        rotation_deg=np.array(0, dtype=np.int32),
        source_patient_id=np.array("1"),
        source_case_id=np.array("2"),
        source_case_key=np.array("patient1_case2"),
    )
    manifest = {
        "schema": "x2cbct_difnet_v0b_cache.v2",
        "geometry_branch": "real_projection",
        "axis_convention": axis_convention(),
        "cache_version": "real_base",
        "calibration_report": {
            "path": str(PROJECT_ROOT / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"),
            "status": "pass",
            "candidate_id": "dummy",
            "adapter_version": "dummy",
        },
        "target_size": 8,
        "projection_size": 8,
        "point_sampling": {**official_lattice_metadata(8), "fixed_training_points_saved": False},
        "target_volume_mode": "base_reference",
        "setup_augmentation": {"enabled": True, "combination_count_per_source_case": 15},
        "views": {"view_10": [200, 215, 230, 245, 260, 275, 290, 305, 320, 335]},
        "split": {"patient_overlap": [], "allow_patient_overlap": False},
        "samples": [
            {
                "sample_id": "patient1_case2_si_0_rot_0_view_10",
                "split": "train",
                "view_key": "view_10",
                "patient_id": "1",
                "case_id": "2",
                "sample_path": "view_10/train/patient1_case2_si_0_rot_0_view_10.npz",
                "case_key": "patient1_case2_si_0_rot_0",
                "case_manifest_path": "cases/patient1_case2_si_0_rot_0/case_manifest.json",
                "cbct_payload_path": "cases/patient1_case2/cbct_norm.npy",
                "projection_payload_paths": [f"projections/patient1_case2_si_0_rot_0/projection_{index:03d}.npy" for index in range(10)],
                "selected_projections": [
                    {
                        "payload_path": f"projections/patient1_case2_si_0_rot_0/projection_{index:03d}.npy",
                        "metadata_path": f"projections/patient1_case2_si_0_rot_0/projection_{index:03d}.json",
                        "file": f"projection_{index:03d}.npy",
                        "gantry_angle_deg": 200 + index * 15,
                    }
                    for index in range(10)
                ],
                "setup_augmentation": {
                    "augmentation_id": "si_0_rot_0",
                    "translation_mm": 0,
                    "rotation_deg": 0,
                    "source_case_key": "patient1_case2",
                },
            }
        ],
        "failed_cases": [],
    }
    return _write_json(root / "cache_manifest.json", manifest)


def _real_derivative_manifest(path: Path, *, augmented: bool) -> Path:
    return _write_json(
        path,
        {
            "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
            "name": "self_projection1208_patient_aug" if augmented else "self_projection1208_patient_noaug",
            "dataset": "self_projection1208",
            "split_policy": "patient",
            "augmentation": {
                "enabled": augmented,
                "translation_mm": [-4, 0, 4] if augmented else [0],
                "rotation_deg": [-6, -3, 0, 3, 6] if augmented else [0],
                "combination_count_per_source_case": 15 if augmented else 1,
            },
            "splits": {
                "train": {"cases": [{"case_key": "1/2", "patient_id": "1", "case_id": "2"}]},
                "test": {"cases": []},
            },
        },
    )


def _write_luna_base_cache(root: Path) -> Path:
    case_dir = root / "cases" / "case_a"
    projection_dir = root / "projections" / "case_a"
    case_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "ct_norm.npy", np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8))
    for index in range(10):
        np.save(projection_dir / f"projection_{index:03d}.npy", np.full((8, 8), index, dtype=np.float32))
    base_manifest = {
        "schema": "x2cbct_synthrad2025_task2_view_cache.v1",
        "cache_version": "luna_base",
        "geometry_branch": "simulated_drr",
        "source": "luna16",
        "anatomy": "luna16",
        "dataset_group": "luna16",
        "views": ["view_10"],
        "view_counts": {"view_10": 10},
        "target_size": 8,
        "projection_size": 8,
        "target_volume_mode": "base_reference",
        "point_sampling": {**official_lattice_metadata(8), "fixed_training_points_saved": False},
        "drr_config": {
            "projection_count": 10,
            "total_angle_deg": 180.0,
            "start_angle_deg": 0.0,
            "detector_size": 8,
            "voxel_size": 8,
            "dsd_mm": 1200.0,
            "dso_mm": 800.0,
            "detector_spacing_mm": 2.0,
            "voxel_spacing_mm": 1.0,
        },
        "samples": [
            {
                "sample_id": "luna16_luna16_case_a_si_0_rot_0_view_10",
                "source": "luna16",
                "dataset_group": "luna16",
                "case_id": "case_a_si_0_rot_0",
                "source_case_id": "case_a",
                "split": "train",
                "view_key": "view_10",
                "projection_paths": [f"projections/case_a/projection_{index:03d}.npy" for index in range(10)],
                "angles_deg": [float(index) for index in range(10)],
                "processed_cbct_path": "cases/case_a/ct_norm.npy",
                "processed_shape_zyx": [8, 8, 8],
                "processed_spacing_xyz": [1.0, 1.0, 1.0],
                "setup_augmentation": {"augmentation_id": "si_0_rot_0", "source_case_id": "case_a"},
                "augmentation_id": "si_0_rot_0",
            }
        ],
    }
    return _write_json(root / "cache_manifest.json", base_manifest)


def test_real_training_derivative_writes_only_index_files(tmp_path: Path) -> None:
    base_root = tmp_path / "real_base"
    _write_real_base_cache(base_root)
    derivative = _real_derivative_manifest(tmp_path / "real_noaug.json", augmented=False)
    out_root = tmp_path / "real_noaug_train"

    manifest = derive_v1_training_cache(
        base_cache_root=base_root,
        derivative_manifest_path=derivative,
        output_root=out_root,
        views="2,5,10",
    )

    assert manifest["view_sample_counts"] == {
        "view_2": {"train": 1, "test": 0},
        "view_5": {"train": 1, "test": 0},
        "view_10": {"train": 1, "test": 0},
    }
    assert not (out_root / "cases").exists()
    assert not (out_root / "projections").exists()
    with np.load(out_root / "view_2" / "train" / "patient1_case2_view_2.npz", allow_pickle=False) as payload:
        assert payload["projection_payload_paths"].tolist() == [
            str((base_root / "projections/patient1_case2_si_0_rot_0/projection_000.npy").resolve()),
            str((base_root / "projections/patient1_case2_si_0_rot_0/projection_006.npy").resolve()),
        ]
        assert "augmentation_id" not in payload.files


def test_real_training_derivative_can_be_read_by_cached_dataset(tmp_path: Path, monkeypatch: Any) -> None:
    base_root = tmp_path / "real_base"
    _write_real_base_cache(base_root)
    derivative = _real_derivative_manifest(tmp_path / "real_aug.json", augmented=True)
    out_root = tmp_path / "real_aug_train"

    derive_v1_training_cache(
        base_cache_root=base_root,
        derivative_manifest_path=derivative,
        output_root=out_root,
        views="2",
    )

    class DummyAdapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def project_many_views(self, physical_points: np.ndarray, gantry_angles: list[float], **_kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            point_count = int(physical_points.shape[0])
            view_count = len(gantry_angles)
            return np.zeros((view_count, point_count, 2), dtype=np.float32), np.ones((view_count, point_count), dtype=bool)

    monkeypatch.setattr(
        cached_dataset_module,
        "load_calibration_report",
        lambda *_args, **_kwargs: SimpleNamespace(candidate_id="dummy", adapter_version="dummy"),
    )
    monkeypatch.setattr(cached_dataset_module, "read_rtk_scan_geometry", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cached_dataset_module, "ClinicalGeometryAdapter", DummyAdapter)

    dataset = CachedDifNetV0bDataset(out_root, split="train", view_key="view_2", num_points=4, deterministic=True)
    item = dataset[0]

    assert item["projs"].shape == (2, 1, 8, 8)
    assert item["points"].shape == (4, 3)
    assert item["proj_points"].shape == (2, 4, 2)
    assert item["p_gt"].shape == (1, 4)
    assert item["valid_mask"].shape == (2, 4)
    assert item["trace"]["target_volume_mode"] == "base_reference"
    assert item["trace"]["augmentation_id"] == "si_0_rot_0"


def test_luna16_training_derivative_writes_manifest_only(tmp_path: Path) -> None:
    base_root = tmp_path / "luna_base"
    _write_luna_base_cache(base_root)
    derivative = _write_json(
        tmp_path / "luna_noaug.json",
        {
            "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
            "name": "luna16_noaug",
            "dataset": "luna16",
            "split_policy": "official_train_test",
            "augmentation": {"enabled": False, "translation_mm": [0], "rotation_deg": [0], "combination_count_per_source_case": 1},
            "splits": {"train": {"cases": [{"case_key": "subset0_case_a", "seriesuid": "case_a", "subset": "subset0"}]}, "test": {"cases": []}},
        },
    )

    manifest = derive_v1_training_cache(
        base_cache_root=base_root,
        derivative_manifest_path=derivative,
        output_root=tmp_path / "luna_train",
        views="2,5,10",
    )

    assert manifest["counts"]["samples"] == 3
    assert not (tmp_path / "luna_train" / "cases").exists()
    assert not (tmp_path / "luna_train" / "projections").exists()
    view2 = [item for item in manifest["samples"] if item["view_key"] == "view_2"][0]
    assert view2["projection_paths"] == [
        str((base_root / "projections/case_a/projection_000.npy").resolve()),
        str((base_root / "projections/case_a/projection_006.npy").resolve()),
    ]


def test_luna16_training_derivative_can_be_read_by_cached_dataset(tmp_path: Path) -> None:
    base_root = tmp_path / "luna_base"
    _write_luna_base_cache(base_root)
    derivative = _write_json(
        tmp_path / "luna_noaug.json",
        {
            "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
            "name": "luna16_noaug",
            "dataset": "luna16",
            "split_policy": "official_train_test",
            "augmentation": {"enabled": False, "translation_mm": [0], "rotation_deg": [0], "combination_count_per_source_case": 1},
            "splits": {"train": {"cases": [{"case_key": "subset0_case_a", "seriesuid": "case_a", "subset": "subset0"}]}, "test": {"cases": []}},
        },
    )
    out_root = tmp_path / "luna_train"

    manifest = derive_v1_training_cache(
        base_cache_root=base_root,
        derivative_manifest_path=derivative,
        output_root=out_root,
        views="2",
    )
    dataset = CachedDifNetV0bDataset(out_root, split="train", view_key="view_2", num_points=4, deterministic=True)
    item = dataset[0]

    assert {sample["split"] for sample in manifest["samples"]} == {"train"}
    assert item["projs"].shape == (2, 1, 8, 8)
    assert item["points"].shape == (4, 3)
    assert item["proj_points"].shape == (2, 4, 2)
    assert item["p_gt"].shape == (1, 4)
    assert item["valid_mask"].dtype == bool
    assert item["trace"]["source"] == "luna16"
    assert item["trace"]["target_volume_mode"] == "base_reference"

    viewer_dir = tmp_path / "luna_viewer"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_difnet_v0b_sample.py"),
            "--cache-root",
            str(out_root),
            "--split",
            "train",
            "--view-count",
            "2",
            "--index",
            "0",
            "--slice-stride",
            "4",
            "--output-dir",
            str(viewer_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    viewer_data = json.loads((viewer_dir / "viewer_data.json").read_text(encoding="utf-8"))
    assert viewer_data["schema"] == "x2cbct_difnet_v0b_interactive_viewer.v1"
    assert viewer_data["split"] == "train"
    assert viewer_data["view_key"] == "view_2"
