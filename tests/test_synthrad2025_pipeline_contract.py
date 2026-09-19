from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from x2cbct_difnet.official_drr import normalize_c2rv_projection, normalize_official_difnet_projection
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset
from tools.build_ct_only_view_cache import _infer_target_size as infer_ct_only_view_target_size
from tools.build_luna16_difnet_dataset import _default_output_root, build_parser as build_luna16_arg_parser, resolve_size_config
from tools.build_synthrad2025_view_cache import _infer_target_size as infer_synthrad_view_target_size
from tools.view_raw_volume_sample import _display_window as _raw_volume_display_window
from tools.serve_luna16_batch_viewer import VIEWER_HTML, build_catalog, render_slice_png


def _write_mha(path: Path, value: float) -> None:
    array = np.full((8, 9, 10), value, dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.2, 1.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _write_mask(path: Path) -> None:
    array = np.zeros((8, 9, 10), dtype=np.uint8)
    array[:, 2:7, 3:8] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.2, 1.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _write_luna_mhd(path: Path, shape_zyx: tuple[int, int, int], spacing_xyz: tuple[float, float, float], value: float) -> None:
    array = np.full(shape_zyx, value, dtype=np.int16)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _write_luna_splits(path: Path, *, train: list[str], eval_cases: list[str] | None = None, test: list[str] | None = None) -> None:
    path.write_text(
        json.dumps({"train": train, "eval": eval_cases or [], "test": test or []}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_synthrad2025_registry_preprocess_and_drr_smoke(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_mha(raw_root / "Task2" / "chest" / "case_chest_001" / "cbct.mha", 10.0)
    _write_mask(raw_root / "Task2" / "chest" / "case_chest_001" / "mask.mha")
    _write_mha(raw_root / "Task2" / "abdomen" / "case_abdomen_001" / "cbct.mha", 20.0)
    _write_mask(raw_root / "Task2" / "abdomen" / "case_abdomen_001" / "mask.mha")
    _write_mha(raw_root / "Task2" / "head_neck" / "case_head_001" / "cbct.mha", 30.0)
    _write_mha(raw_root / "Task2" / "chest" / "case_chest_001" / "ct.mha", 40.0)

    registry_path = tmp_path / "registry.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_synthrad2025_registry.py"),
            "--raw-root",
            str(raw_root),
            "--output",
            str(registry_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["ct_used"] is False
    assert {case["anatomy"] for case in registry["cases"]} == {"chest", "abdomen"}
    assert all("ct" not in Path(case["cbct_path"]).name.lower().replace("cbct", "") for case in registry["cases"])
    assert all("mask_path" in case for case in registry["cases"])

    preprocess_root = tmp_path / "preprocessed"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "preprocess_synthrad2025_cbct.py"),
            "--registry",
            str(registry_path),
            "--output-root",
            str(preprocess_root),
            "--target-size",
            "16",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    preprocess_report = json.loads((preprocess_root / "preprocess_report.json").read_text(encoding="utf-8"))
    assert preprocess_report["processed_count"] == 2
    assert preprocess_report["geometry_branch"] == "simulated_drr"
    assert preprocess_report["resample_spacing_xyz"] == [0.8, 0.8, 0.8]
    assert preprocess_report["preprocess_policy"] == "spacing_aware_resample_then_center_crop_pad"
    assert preprocess_report["cbct_normalization"]["mode"] == "percentile"
    assert set(preprocess_report["cbct_normalization"]["profiles"]) == {"abdomen", "chest"}
    first = np.load(preprocess_report["processed_cases"][0]["processed_cbct_path"])
    assert first.shape == (16, 16, 16)
    assert np.isfinite(first).all()
    first_case = preprocess_report["processed_cases"][0]
    assert first_case["processed_shape_zyx"] == [16, 16, 16]
    assert first_case["processed_spacing_xyz"] == [0.8, 0.8, 0.8]
    assert first_case["preprocess"]["policy"] == "spacing_aware_resample_then_center_crop_pad"
    assert first_case["preprocess"]["raw_origin_used_for_training_geometry"] is False
    assert "hu_span" in first_case["cbct_normalization"]
    assert Path(preprocess_report["processed_cases"][0]["processed_mask_path"]).exists()

    drr_root = tmp_path / "drr"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_synthrad2025_drr.py"),
            "--preprocess-report",
            str(preprocess_root / "preprocess_report.json"),
            "--output-root",
            str(drr_root),
            "--projector",
            "fallback-ray-sum",
            "--max-cases",
            "1",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    drr_manifest = json.loads((drr_root / "drr_manifest.json").read_text(encoding="utf-8"))
    assert drr_manifest["generated_count"] == 1
    assert drr_manifest["cbct_normalization"]["profiles"]
    assert drr_manifest["cases"][0]["projector"] == "fallback-ray-sum"
    assert drr_manifest["cases"][0]["projection_value_space"] == "smoke_float16_0_1"
    assert drr_manifest["cases"][0]["drr_config"]["voxel_size"] == 16
    assert drr_manifest["cases"][0]["drr_config"]["voxel_spacing_mm"] == 0.8
    assert drr_manifest["cases"][0]["projection_geometry"]["voxel_spacing_mm"] == 0.8
    assert len(drr_manifest["cases"][0]["projection_paths"]) == 50
    projection = np.load(drr_manifest["cases"][0]["projection_paths"][0])
    assert projection.shape == (16, 16)
    assert np.isfinite(projection).all()

    view_cache_root = tmp_path / "view_cache"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_synthrad2025_view_cache.py"),
            "--drr-manifest",
            str(drr_root / "drr_manifest.json"),
            "--output-root",
            str(view_cache_root),
            "--views",
            "2,5,10",
            "--target-size",
            "16",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    view_manifest = json.loads((view_cache_root / "abdomen" / "cache_manifest.json").read_text(encoding="utf-8"))
    assert view_manifest["schema"] == "x2cbct_synthrad2025_task2_view_cache.v1"
    assert view_manifest["geometry_branch"] == "simulated_drr"
    assert view_manifest["test_is_validation"] is True
    assert view_manifest["views"] == ["view_2", "view_5", "view_10"]
    assert {sample["view_count"] for sample in view_manifest["samples"]} == {2, 5, 10}
    assert view_manifest["point_sampling"]["name"] == "official_interleaved_lattice"
    assert view_manifest["drr_config"]["detector_size"] == 16
    assert view_manifest["drr_config"]["dso_mm"] == 1000.0
    assert view_manifest["cbct_normalization"]["profile_id"].startswith("abdomen_")

    sample = view_manifest["samples"][0]
    dataset = CachedDifNetV0bDataset(
        view_cache_root / view_manifest["anatomy"],
        split=sample["split"],
        view_key=sample["view_key"],
        num_points=8,
        deterministic=True,
    )
    item = dataset[0]

    assert item["projs"].shape == (sample["view_count"], 1, 16, 16)
    assert item["points"].shape == (8, 3)
    assert item["proj_points"].shape == (sample["view_count"], 8, 2)
    assert item["p_gt"].shape == (1, 8)
    assert item["valid_mask"].dtype == bool
    assert np.isfinite(item["proj_points"]).all()
    assert item["trace"]["source"] == "synthrad2025_task2"
    assert item["trace"]["anatomy"] == view_manifest["anatomy"]
    assert item["trace"]["geometry_branch"] == "simulated_drr"
    assert item["trace"]["processed_spacing_xyz"] == [0.8, 0.8, 0.8]
    assert item["trace"]["drr_config"]["voxel_spacing_mm"] == 0.8

    train_root = tmp_path / "train_synthrad"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "train_difnet_v0b.py"),
            "--cache-root",
            str(view_cache_root / view_manifest["anatomy"]),
            "--view-count",
            str(sample["view_count"]),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-points",
            "8",
            "--steps-per-epoch",
            "1",
            "--full-test-interval",
            "0",
            "--skip-final-validation",
            "--mid-channels",
            "4",
            "--device",
            "cpu",
            "--loss-valid-policy",
            "any",
            "--output-dir",
            str(train_root),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    assert not (train_root / "checkpoints").exists()
    assert (train_root / "config_resolved.json").exists()
    assert (train_root / "validation_plan.json").exists()
    assert (train_root / "metrics_final.json").exists()


def test_luna16_scan_writes_patient_split_and_profile(tmp_path: Path) -> None:
    root = tmp_path / "LUNA16"
    for index, value in enumerate((-900.0, -700.0, 120.0), start=1):
        uid = f"1.2.826.0.1.{index:03d}"
        _write_luna_mhd(root / "subset0" / "subset0" / f"{uid}.mhd", (4, 6, 6), (0.7, 0.7, 2.5), value)
        _write_luna_mhd(root / "seg-lungs-LUNA16" / f"{uid}.mhd", (4, 6, 6), (0.7, 0.7, 2.5), 1.0)
    bad_uid = "1.2.826.0.1.999"
    _write_luna_mhd(root / "subset0" / "subset0" / f"{bad_uid}.mhd", (4, 5, 6), (0.7, 0.7, 2.5), -500.0)
    _write_luna_mhd(root / "seg-lungs-LUNA16" / f"{bad_uid}.mhd", (4, 5, 6), (0.7, 0.7, 2.5), 1.0)

    report_path = tmp_path / "luna_report.json"
    registry_path = tmp_path / "luna_registry.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "scan_luna16_dataset_profile.py"),
            "--root",
            str(root),
            "--output",
            str(report_path),
            "--registry-output",
            str(registry_path),
            "--expected-inplane",
            "6,6",
            "--train-ratio",
            "0.67",
            "--split-policy",
            "patient",
            "--hu-sample-voxels-per-case",
            "1000",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert report["included_count"] == 3
    assert report["excluded_count"] == 1
    assert report["exclusion_summary"] == {"unexpected_inplane_shape": 1}
    assert report["split"]["patient_overlap"] == []
    assert registry["split"]["train_case_count"] + registry["split"]["test_case_count"] == 3
    assert registry["hu_window"]["profiles"]["luna16"]["hu_span"] > 0

    official_report_path = tmp_path / "luna_report_official.json"
    official_registry_path = tmp_path / "luna_registry_official.json"
    splits_path = tmp_path / "luna_splits.json"
    _write_luna_splits(
        splits_path,
        train=["subset0_1.2.826.0.1.001"],
        eval_cases=["subset0_1.2.826.0.1.002"],
        test=["subset0_1.2.826.0.1.003"],
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "scan_luna16_dataset_profile.py"),
            "--root",
            str(root),
            "--output",
            str(official_report_path),
            "--registry-output",
            str(official_registry_path),
            "--expected-inplane",
            "6,6",
            "--split-policy",
            "c2rv",
            "--splits-json",
            str(splits_path),
            "--metadata-only",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    official_registry = json.loads(official_registry_path.read_text(encoding="utf-8"))
    assert official_registry["split"]["policy"] == "luna16_c2rv_deepsparse_splits"
    assert official_registry["split"]["train_case_count"] == 1
    assert official_registry["split"]["eval_case_count"] == 1
    assert official_registry["split"]["test_case_count"] == 1
    assert {case["split"] for case in official_registry["cases"]} == {"train", "eval", "test"}

    rejected = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "scan_luna16_dataset_profile.py"),
            "--root",
            str(root),
            "--output",
            str(tmp_path / "luna_report_mixed.json"),
            "--registry-output",
            str(tmp_path / "luna_registry_mixed.json"),
            "--expected-inplane",
            "6,6",
            "--split-policy",
            "mixed",
            "--metadata-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "mixed" in rejected.stderr


def test_ct_only_preprocess_resamples_anisotropic_registry_to_canonical_cube(tmp_path: Path) -> None:
    root = tmp_path / "LUNA16"
    uid = "1.2.826.0.1.777"
    ct_path = root / "subset0" / "subset0" / f"{uid}.mhd"
    _write_luna_mhd(ct_path, (4, 6, 6), (0.7, 0.7, 2.5), -700.0)
    registry_path = tmp_path / "ct_only_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "x2cbct_ct_only_test_registry.v1",
                "raw_root": str(root),
                "cases": [
                    {
                        "seriesuid": uid,
                        "split": "train",
                        "ct_path": str(ct_path),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "ct_only_preprocessed"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "preprocess_ct_only_volumes.py"),
            "--registry",
            str(registry_path),
            "--output-root",
            str(output_root),
            "--dataset-name",
            "luna16",
            "--target-shape",
            "8,8,8",
            "--resample-spacing",
            "1.6,1.6,1.6",
            "--hu-sample-voxels-per-case",
            "1000",
            "--drop-hu-after-normalization",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    report = json.loads((output_root / "preprocess_report.json").read_text(encoding="utf-8"))
    case = report["processed_cases"][0]
    volume = np.load(case["processed_cbct_path"])

    assert report["schema"] == "x2cbct_ct_only_preprocess_report.v1"
    assert report["geometry_branch"] == "simulated_drr"
    assert report["config"]["resample_spacing_xyz"] == [1.6, 1.6, 1.6]
    assert volume.shape == (8, 8, 8)
    assert case["processed_shape_zyx"] == [8, 8, 8]
    assert case["processed_size_xyz"] == [8, 8, 8]
    assert case["processed_spacing_xyz"] == [1.6, 1.6, 1.6]
    assert case["processed_origin_xyz"] == [0.0, 0.0, 0.0]
    assert case["preprocess"]["policy"] == "spacing_aware_resample_then_center_crop_pad"
    assert case["preprocess"]["canonical_center_is_isocenter"] is True
    assert case["preprocess"]["raw_origin_used_for_training_geometry"] is False
    assert case["preprocess"]["crop_pad"]["input_shape_zyx"] == [6, 3, 3]
    assert case["processed_hu_retained"] is False
    assert not Path(case["processed_hu_path"]).exists()
    assert report["cbct_normalization"]["profiles"]["luna16"]["source"]["preprocess_policy"] == "spacing_aware_resample_then_center_crop_pad"

    drr_root = tmp_path / "ct_only_drr"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_synthrad2025_drr.py"),
            "--preprocess-report",
            str(output_root / "preprocess_report.json"),
            "--output-root",
            str(drr_root),
            "--projector",
            "fallback-ray-sum",
            "--max-cases",
            "1",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    view_cache_root = tmp_path / "ct_only_view_cache"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_ct_only_view_cache.py"),
            "--drr-manifest",
            str(drr_root / "drr_manifest.json"),
            "--output-root",
            str(view_cache_root),
            "--dataset-name",
            "luna16",
            "--views",
            "2,5,10",
            "--asset-mode",
            "copy",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    view_manifest = json.loads((view_cache_root / "luna16" / "cache_manifest.json").read_text(encoding="utf-8"))
    assert view_manifest["schema"] == "x2cbct_synthrad2025_task2_view_cache.v1"
    assert view_manifest["source"] == "luna16"
    assert view_manifest["dataset_group"] == "luna16"
    assert view_manifest["geometry_branch"] == "simulated_drr"
    assert view_manifest["drr_config"]["voxel_size"] == 8
    assert view_manifest["drr_config"]["detector_size"] == 8
    assert view_manifest["projection_size"] == 8
    assert view_manifest["drr_config"]["voxel_spacing_mm"] == 1.6
    assert view_manifest["asset_mode"] == "copy"
    assert view_manifest["self_contained_assets"] is True
    assert {sample["view_count"] for sample in view_manifest["samples"]} == {2, 5, 10}
    sample = view_manifest["samples"][0]
    cache_prefix = str((view_cache_root / "luna16").resolve())
    assert str(Path(sample["processed_cbct_path"]).resolve()).startswith(cache_prefix)
    assert all(str(Path(path).resolve()).startswith(cache_prefix) for path in sample["projection_paths"])

    shutil.rmtree(output_root)
    shutil.rmtree(drr_root)

    dataset = CachedDifNetV0bDataset(
        view_cache_root / "luna16",
        split="train",
        view_key="view_2",
        num_points=4,
        deterministic=True,
    )
    item = dataset[0]
    assert item["trace"]["source"] == "luna16"
    assert item["trace"]["geometry_branch"] == "simulated_drr"
    assert item["trace"]["processed_spacing_xyz"] == [1.6, 1.6, 1.6]
    assert item["projs"].shape == (2, 1, 8, 8)
    assert item["points"].shape == (4, 3)
    assert np.isfinite(item["proj_points"]).all()


def test_luna16_one_command_builder_writes_self_contained_cache_and_cleans_workdir(tmp_path: Path) -> None:
    root = tmp_path / "LUNA16"
    for index, value in enumerate((-900.0, -700.0), start=1):
        uid = f"1.2.826.0.1.{index:03d}"
        _write_luna_mhd(root / "subset0" / "subset0" / f"{uid}.mhd", (4, 6, 6), (0.7, 0.7, 2.5), value)
        _write_luna_mhd(root / "seg-lungs-LUNA16" / f"{uid}.mhd", (4, 6, 6), (0.7, 0.7, 2.5), 1.0)

    output_root = tmp_path / "luna16_final_cache"
    work_root = tmp_path / "luna16_work"
    splits_path = tmp_path / "luna_splits.json"
    _write_luna_splits(
        splits_path,
        train=["subset0_1.2.826.0.1.001"],
        test=["subset0_1.2.826.0.1.002"],
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_luna16_difnet_dataset.py"),
            "--root",
            str(root),
            "--output-root",
            str(output_root),
            "--work-root",
            str(work_root),
            "--registry-output",
            str(tmp_path / "luna_registry.json"),
            "--profile-output",
            str(tmp_path / "luna_profile.json"),
            "--splits-json",
            str(splits_path),
            "--expected-inplane",
            "6,6",
            "--target-shape",
            "8,8,8",
            "--resample-spacing",
            "1.6,1.6,1.6",
            "--projector",
            "fallback-ray-sum",
            "--drr-projection-count",
            "10",
            "--views",
            "2",
            "--max-cases",
            "1",
            "--hu-sample-voxels-per-case",
            "1000",
            "--hu-window-split",
            "all",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    manifest = json.loads((output_root / "luna16" / "cache_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "build_summary.json").read_text(encoding="utf-8"))
    command_log = [json.loads(line) for line in (output_root / "build_commands.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = manifest["samples"][0]
    cache_prefix = str((output_root / "luna16").resolve())
    assert manifest["asset_mode"] == "copy"
    assert manifest["self_contained_assets"] is True
    assert manifest["target_size"] == 8
    assert manifest["projection_size"] == 8
    assert manifest["drr_config"]["detector_size"] == 8
    assert manifest["drr_config"]["projection_count"] == 10
    assert manifest["drr_config"]["detector_spacing_mm"] == 2.4
    assert manifest["drr_config"]["dso_mm"] == 800.0
    assert manifest["drr_config"]["dsd_mm"] == 1200.0
    assert manifest["projection_value_space"] == "c2rv_uint8_0_255"
    assert summary["target_shape"] == "8,8,8"
    assert summary["size_label"] == "vol8_proj8"
    assert summary["volume_size"] == 8
    assert summary["projection_size"] == 8
    assert summary["detector_size"] == 8
    assert summary["drr_projection_count"] == 10
    assert summary["detector_spacing_mm"] == 2.4
    assert summary["split_policy"] == "c2rv"
    assert summary["hu_window"] == [-1000.0, 1024.0]
    assert summary["command_log"] == str((output_root / "build_commands.jsonl").resolve())
    assert [item["stage"] for item in command_log] == ["scan_registry", "preprocess_ct", "generate_drr", "build_view_cache"]
    scan_command = next(item["command"] for item in command_log if item["stage"] == "scan_registry")
    preprocess_command = next(item["command"] for item in command_log if item["stage"] == "preprocess_ct")
    drr_command = next(item["command"] for item in command_log if item["stage"] == "generate_drr")
    assert "--hu-percentiles" not in scan_command
    assert "--hu-percentiles" not in preprocess_command
    assert "--projection-count" in drr_command
    assert drr_command[drr_command.index("--projection-count") + 1] == "10"
    assert "--progress-interval" in preprocess_command
    assert "--progress-interval" in drr_command
    assert str(Path(sample["processed_cbct_path"]).resolve()).startswith(cache_prefix)
    assert all(str(Path(path).resolve()).startswith(cache_prefix) for path in sample["projection_paths"])
    assert sample["projection_loader_scale"] is not None
    assert not work_root.exists()
    assert (output_root / "build_summary.json").exists()

    dataset = CachedDifNetV0bDataset(output_root / "luna16", split=sample["split"], view_key="view_2", num_points=4, deterministic=True)
    item = dataset[0]
    assert item["trace"]["source"] == "luna16"
    assert item["projs"].shape == (2, 1, 8, 8)
    assert item["points"].shape == (4, 3)
    assert item["trace"]["detector_spacing_yx"] == [2.4, 2.4]

    viewer_dir = tmp_path / "luna_viewer"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_difnet_v0b_sample.py"),
            "--cache-root",
            str(output_root / "luna16"),
            "--split",
            sample["split"],
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
    assert viewer_data["sample_id"] == sample["sample_id"]
    assert viewer_data["geometry_branch"] == "simulated_drr"
    assert viewer_data["cbct"]["processed_spacing_xyz"] == [1.6, 1.6, 1.6]
    assert len(viewer_data["projections"]) == 2
    axial_center = next(
        item
        for item in viewer_data["cbct"]["planes"]["axial"]["slices"]
        if item["index"] == viewer_data["cbct"]["planes"]["axial"]["center_index"]
    )
    axial_png = cv2.imread(str(viewer_dir / axial_center["png"]), cv2.IMREAD_UNCHANGED)
    assert axial_png is not None
    assert int(axial_png.max()) > 0

    batch_viewer_dir = tmp_path / "luna_batch_viewer"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "export_difnet_v0b_sample_viewers.py"),
            "--cache-root",
            str(output_root / "luna16"),
            "--split",
            "all",
            "--view-count",
            "2",
            "--max-samples",
            "1",
            "--slice-stride",
            "4",
            "--output-root",
            str(batch_viewer_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    batch_index = json.loads((batch_viewer_dir / "viewer_batch_index.json").read_text(encoding="utf-8"))
    assert batch_index["viewer_count"] == 1
    assert batch_index["rows"][0]["sample_id"] == sample["sample_id"]
    viewer_relative_path = batch_index["rows"][0]["viewer_relative_path"]
    assert len(viewer_relative_path) < 120
    assert (batch_viewer_dir / viewer_relative_path).exists()


def test_luna16_one_command_builder_defaults_to_c2rv_deepsparse_256_cache_names() -> None:
    parser = build_luna16_arg_parser()
    args = parser.parse_args([])
    size_config = resolve_size_config(args)

    assert size_config.target_shape_text == "256,256,256"
    assert size_config.resample_spacing_text == "1.6,1.6,1.6"
    assert size_config.detector_size == 256
    assert size_config.detector_spacing_mm == 2.4
    assert size_config.dso_mm == 800.0
    assert size_config.dsd_mm == 1200.0
    assert size_config.label == "vol256_proj256"
    assert _default_output_root(args).name == "difnet_cache_c2rvds_1p6mm_vol256_proj256_det2p4mm"


def test_luna16_one_command_builder_supports_c2rv_deepsparse_128_cache_names() -> None:
    parser = build_luna16_arg_parser()
    args = parser.parse_args(["--size-preset", "128"])
    size_config = resolve_size_config(args)

    assert size_config.target_shape_text == "128,128,128"
    assert size_config.resample_spacing_text == "3.2,3.2,3.2"
    assert size_config.detector_size == 128
    assert size_config.detector_spacing_mm == 4.8
    assert size_config.label == "vol128_proj128"
    assert _default_output_root(args).name == "difnet_cache_c2rvds_3p2mm_vol128_proj128_det4p8mm"


def test_luna16_one_command_builder_pairs_explicit_target_shape_with_projection_size() -> None:
    parser = build_luna16_arg_parser()
    args = parser.parse_args(["--target-shape", "8,8,8", "--resample-spacing", "1.6,1.6,1.6"])
    size_config = resolve_size_config(args)

    assert size_config.target_shape_text == "8,8,8"
    assert size_config.detector_size == 8
    assert size_config.detector_spacing_mm == 2.4
    assert size_config.label == "vol8_proj8"


def test_luna16_one_command_builder_rejects_mismatched_volume_and_projection_size() -> None:
    parser = build_luna16_arg_parser()
    args = parser.parse_args(["--target-shape", "128,128,128", "--detector-size", "256"])

    try:
        resolve_size_config(args)
    except ValueError as exc:
        assert "volume size and projection detector size must match" in str(exc)
    else:
        raise AssertionError("expected mismatched LUNA16 size config to fail")


def test_view_cache_target_size_inference_prefers_case_level_drr_config() -> None:
    manifest = {
        "drr_config": {"voxel_size": 256},
        "cases": [
            {
                "drr_config": {"voxel_size": 128},
                "processed_shape_zyx": [128, 128, 128],
            }
        ],
    }

    assert infer_ct_only_view_target_size(manifest) == 128
    assert infer_synthrad_view_target_size(manifest) == 128


def test_raw_volume_viewer_exports_interactive_html(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw" / "Task2" / "TH" / "2THC014" / "cbct.mha"
    _write_mha(raw_path, 25.0)
    output_dir = tmp_path / "raw_viewer"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_raw_volume_sample.py"),
            "--volume-path",
            str(raw_path),
            "--slice-stride",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    html = (output_dir / "index.html").read_text(encoding="utf-8")
    viewer_data = json.loads((output_dir / "viewer_data.json").read_text(encoding="utf-8"))

    assert "Raw Volume Interactive Viewer" in html
    assert "axialSlider" in html
    assert "viewer_data.js" in html
    assert viewer_data["schema"] == "x2cbct_raw_volume_viewer.v1"
    assert viewer_data["case_id"] == "2THC014"
    assert viewer_data["summary"]["shape_zyx"] == [8, 9, 10]
    assert viewer_data["image"]["spacing_xyz"] == [1.0, 1.2, 1.5]
    assert viewer_data["planes"]["axial"]["slices"]
    assert (output_dir / viewer_data["planes"]["axial"]["slices"][0]["png"]).exists()
    assert (output_dir / "viewer_summary.json").exists()


def test_raw_volume_viewer_exports_spacing_corrected_pngs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw" / "Task2" / "TH" / "2THC014" / "cbct.mha"
    _write_mha(raw_path, 25.0)
    output_dir = tmp_path / "raw_viewer_spacing"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_raw_volume_sample.py"),
            "--volume-path",
            str(raw_path),
            "--slice-stride",
            "8",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    viewer_data = json.loads((output_dir / "viewer_data.json").read_text(encoding="utf-8"))
    axial_png = output_dir / viewer_data["planes"]["axial"]["slices"][0]["png"]
    coronal_png = output_dir / viewer_data["planes"]["coronal"]["slices"][0]["png"]
    sagittal_png = output_dir / viewer_data["planes"]["sagittal"]["slices"][0]["png"]

    assert cv2.imread(str(axial_png), cv2.IMREAD_GRAYSCALE).shape == (11, 10)
    assert cv2.imread(str(coronal_png), cv2.IMREAD_GRAYSCALE).shape == (12, 10)
    assert cv2.imread(str(sagittal_png), cv2.IMREAD_GRAYSCALE).shape == (10, 9)
    assert viewer_data["planes"]["coronal"]["row_spacing_mm"] == 1.5
    assert viewer_data["planes"]["coronal"]["column_spacing_mm"] == 1.0


def test_luna16_batch_viewer_renders_fake_registry_slice(tmp_path: Path) -> None:
    root = tmp_path / "LUNA16"
    uid = "1.2.826.0.1.42"
    ct_path = root / "subset0" / "subset0" / f"{uid}.mhd"
    mask_path = root / "seg-lungs-LUNA16" / f"{uid}.mhd"
    _write_luna_mhd(ct_path, (6, 8, 10), (0.7, 0.7, 2.5), -800.0)
    _write_luna_mhd(mask_path, (6, 8, 10), (0.7, 0.7, 2.5), 1.0)

    registry_path = tmp_path / "luna_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "x2cbct_luna16_patient_split_registry.v1",
                "hu_window": {
                    "profiles": {
                        "luna16": {"hu_min": -1200.0, "hu_max": 600.0}
                    }
                },
                "cases": [
                    {
                        "patient_id": uid,
                        "seriesuid": uid,
                        "subset": "subset0",
                        "split": "train",
                        "ct_path": str(ct_path),
                        "mask_path": str(mask_path),
                        "relative_ct_path": f"subset0/subset0/{uid}.mhd",
                        "shape_zyx": [6, 8, 10],
                        "spacing_xyz": [0.7, 0.7, 2.5],
                        "compatibility_reasons": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = build_catalog(registry_path)
    png, meta = render_slice_png(catalog, 0, "coronal", 3, window="profile", show_mask=True, max_edge=128)

    assert catalog["case_count"] == 1
    assert catalog["hu_window_profile"] == {"low": -1200.0, "high": 600.0}
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert meta["slice_index"] == 3
    assert meta["window"]["name"] == "profile"
    assert "wheel" in VIEWER_HTML
    assert "localStorage" in VIEWER_HTML


def test_raw_volume_viewer_uses_hu_like_default_window() -> None:
    volume = np.full((8, 9, 10), -1024.0, dtype=np.float32)
    volume[:, 2:7, 3:8] = -50.0
    volume[4, 4, 4] = 620.0

    window = _raw_volume_display_window(
        volume,
        low=None,
        high=None,
        percentile_low=0.5,
        percentile_high=99.5,
    )

    assert window["mode"] == "auto_hu_like"
    assert window["low"] == -1024.0
    assert window["high"] == 620.0


def test_official_difnet_projection_normalization_matches_uint8_contract() -> None:
    raw = np.array([0.0, 0.05, 0.10, 0.20], dtype=np.float32)
    normalized = normalize_official_difnet_projection(raw)
    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [0, 127, 255, 255]


def test_c2rv_projection_normalization_uses_case_max_and_loader_scale() -> None:
    raw = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    normalized, meta = normalize_c2rv_projection(raw, loader_denominator=0.2)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [0, 127, 255]
    assert meta["projection_max"] == 1.0
    assert meta["projection_loader_scale"] == pytest.approx(51.0)
    loaded = normalized.astype(np.float32) / meta["projection_loader_scale"]
    assert loaded.tolist() == pytest.approx([0.0, 127.0 / 51.0, 5.0])
