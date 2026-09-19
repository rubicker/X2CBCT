import json
from pathlib import Path

import numpy as np

from x2cbct_difnet.cache import resolve_cache_path
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, _scan_geometry_from_case_manifest


def test_resolve_cache_path_accepts_manifest_forward_slashes() -> None:
    cache_root = Path("/cache")
    resolved = resolve_cache_path(cache_root, "view_5/train/sample.npz")

    assert resolved == cache_root.resolve() / "view_5" / "train" / "sample.npz"


def test_resolve_cache_path_accepts_windows_style_relative_paths_on_linux() -> None:
    cache_root = Path("/cache")
    resolved = resolve_cache_path(cache_root, r"view_5\train\sample.npz")

    assert resolved == cache_root.resolve() / "view_5" / "train" / "sample.npz"


def test_resolve_cache_path_remaps_migrated_windows_project_path(tmp_path: Path) -> None:
    cache_root = tmp_path / "X2CBCT" / "data_cache" / "difnet_v1" / "self_projection1208" / "train"
    old_path = "Q:/old/X2CBCT/data_cache/difnet_v1/self_projection1208/base/cases/case_manifest.json"

    resolved = resolve_cache_path(cache_root, old_path)

    assert resolved == tmp_path / "X2CBCT" / "data_cache" / "difnet_v1" / "self_projection1208" / "base" / "cases" / "case_manifest.json"


def test_resolve_cache_path_remaps_migrated_windows_project_runs_path(tmp_path: Path) -> None:
    cache_root = tmp_path / "X2CBCT" / "data_cache" / "difnet_v1" / "self_projection1208" / "train"
    old_path = "Q:/old/X2CBCT/runs/audits/rtk_coordinate_candidates/calibration_report.json"

    resolved = resolve_cache_path(cache_root, old_path)

    assert resolved == tmp_path / "X2CBCT" / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"


def test_scan_geometry_uses_cached_projection_geometry_without_raw_case_path(tmp_path: Path) -> None:
    cache_root = tmp_path / "X2CBCT" / "data_cache" / "cache"
    case_manifest = {
        "raw_case_path": "Q:/old/dataset_list/patient0_case0",
        "projection_geometry": {
            "source_to_isocenter_distance": 1000.0,
            "source_to_detector_distance": 1536.0,
        },
    }

    geometry = _scan_geometry_from_case_manifest(cache_root, case_manifest)

    assert geometry.source_to_isocenter_distance == 1000.0
    assert geometry.source_to_detector_distance == 1536.0


def test_synthrad_dataset_remaps_migrated_windows_payload_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "X2CBCT"
    cache_root = project_root / "data_cache" / "difnet_v1" / "luna16" / "train_noaug_256"
    payload_root = project_root / "data_cache" / "difnet_v1" / "luna16" / "base_3x3_256" / "luna16"
    case_dir = payload_root / "cases" / "case0"
    projection_dir = payload_root / "projections" / "case0"
    case_dir.mkdir(parents=True)
    projection_dir.mkdir(parents=True)
    np.save(case_dir / "ct_norm.npy", np.zeros((8, 8, 8), dtype=np.float32))
    np.save(projection_dir / "projection_000.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(projection_dir / "projection_001.npy", np.zeros((8, 8), dtype=np.float32))
    old_prefix = "D:/code/X2CBCT/data_cache/difnet_v1/luna16/base_3x3_256/luna16"
    manifest = {
        "schema": "x2cbct_synthrad2025_task2_view_cache.v1",
        "geometry_branch": "simulated_drr",
        "cache_version": "pytest",
        "source": "pytest",
        "anatomy": "luna16",
        "target_size": 8,
        "projection_size": 8,
        "views": {"view_2": [0.0, 90.0]},
        "point_sampling": {"name": "official_interleaved_lattice", "fixed_training_points_saved": False},
        "samples": [
            {
                "sample_id": "case0_view_2",
                "case_id": "case0",
                "split": "train",
                "view_key": "view_2",
                "angles_deg": [0.0, 90.0],
                "projection_paths": [
                    f"{old_prefix}/projections/case0/projection_000.npy",
                    f"{old_prefix}/projections/case0/projection_001.npy",
                ],
                "processed_cbct_path": f"{old_prefix}/cases/case0/ct_norm.npy",
                "processed_shape_zyx": [8, 8, 8],
                "processed_spacing_xyz": [1.0, 1.0, 1.0],
            }
        ],
    }
    cache_root.mkdir(parents=True)
    (cache_root / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = CachedDifNetV0bDataset(cache_root, split="train", view_key="view_2", num_points=4, deterministic=True)
    item = dataset[0]

    assert item["projs"].shape == (2, 1, 8, 8)
    assert item["p_gt"].shape == (1, 4)
