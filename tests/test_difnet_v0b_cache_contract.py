from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import (
    CacheBuildOptions,
    PROJECTION_CORRECTION_PROFILE_SCHEMA,
    PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
    _load_projection_correction_profile,
    build_difnet_v0b_cache,
    _patient_dark_to_bright_correction_profile,
)
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest, samples_for
from x2cbct_difnet.metrics import metric_dict
from x2cbct_difnet.point_blocks import official_lattice_metadata, sample_official_lattice_indices
from x2cbct_difnet.setup_augmentation import setup_augmentation_config, transform_volume_setup, volume_background_value
from x2cbct_difnet.synthrad2025 import synthrad_view_indices
from x2cbct_difnet.split_policy import CASE_MIXED_POLICY, PATIENT_LEVEL_POLICY, assign_train_test_split
from x2cbct_difnet.volume_grid import (
    axis_convention,
    physical_fov_mm_xyz,
    resized_spacing_preserve_fov_xyz,
    shape_zyx_to_size_xyz,
)
from tools.test_difnet_v0b import (
    DEFAULT_TEST_EVAL_GRID_SIZE,
    DEFAULT_TEST_VIEWER_MAX_CASES,
    DEFAULT_TEST_VIEWER_SLICE_STRIDE,
    DEFAULT_TEST_VISUALIZATION_MAX_CASES,
    _patient_round_robin_samples,
    _shard_samples,
)
from tools.derive_real_difnet_cache import derive_real_difnet_cache
from tools.train_difnet_v0b import (
    DEFAULT_VALIDATION_EVAL_GRID_SIZE,
    _learning_rate_for_step,
    _masked_mse,
    _validation_train_device_context,
    _validation_kind,
)
from tools.view_difnet_v0b_prediction import _display_window, _to_panel


def test_samples_for_error_lists_available_split_view_combinations() -> None:
    manifest = {
        "samples": [
            {"split": "eval", "view_key": "view_2"},
            {"split": "eval", "view_key": "view_5"},
            {"split": "eval", "view_key": "view_5"},
        ]
    }

    with pytest.raises(ValueError) as exc_info:
        samples_for(manifest, "test", "view_5")

    message = str(exc_info.value)
    assert "no samples for split=test, view_key=view_5" in message
    assert "split=eval, view_key=view_2, count=1" in message
    assert "split=eval, view_key=view_5, count=2" in message
    assert "use split=eval for view_key=view_5" in message


def test_shard_samples_partitions_without_overlap() -> None:
    samples = [{"sample_id": f"s{index}"} for index in range(7)]

    shards = [_shard_samples(samples, 3, index) for index in range(3)]

    assert [[sample["sample_id"] for sample in shard] for shard in shards] == [
        ["s0", "s3", "s6"],
        ["s1", "s4"],
        ["s2", "s5"],
    ]
    merged = sorted(sample["sample_id"] for shard in shards for sample in shard)
    assert merged == [f"s{index}" for index in range(7)]


def test_training_validation_runs_every_50_epochs_and_at_final_epoch() -> None:
    assert _validation_kind(49, 400, 50, False) is None
    assert _validation_kind(50, 400, 50, False) == "full"
    assert _validation_kind(100, 400, 50, False) == "full"
    assert _validation_kind(375, 375, 50, False) == "full_final"
    assert _validation_kind(375, 375, 50, True) is None


def test_constant_then_cosine_learning_rate_uses_optimizer_steps() -> None:
    base_lr = 1e-3
    min_lr = 1e-5
    total_steps = 12_000
    warmup_steps = 8_000

    assert _learning_rate_for_step(0, total_steps, warmup_steps, base_lr, min_lr) == pytest.approx(base_lr)
    assert _learning_rate_for_step(7_999, total_steps, warmup_steps, base_lr, min_lr) == pytest.approx(base_lr)
    assert _learning_rate_for_step(8_000, total_steps, warmup_steps, base_lr, min_lr) < base_lr
    assert _learning_rate_for_step(11_999, total_steps, warmup_steps, base_lr, min_lr) == pytest.approx(min_lr)


def test_validation_train_device_context_offloads_and_restores_when_cuda_is_available(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    events_path = tmp_path / "events.jsonl"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        with _validation_train_device_context(
            model=model,
            optimizer=optimizer,
            device=device,
            policy="offload-to-cpu",
            events_path=events_path,
            epoch=50,
        ):
            assert next(model.parameters()).device.type == "cpu"
            assert all(
                value.device.type == "cpu"
                for state in optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            )

        assert next(model.parameters()).device.type == "cuda"
        assert all(
            value.device.type == "cuda"
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        events = events_path.read_text(encoding="utf-8")
        assert '"event": "validation_train_device_offload_start"' in events
        assert '"event": "validation_train_device_restore_end"' in events
    else:
        with _validation_train_device_context(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            policy="offload-to-cpu",
            events_path=events_path,
            epoch=50,
        ):
            assert next(model.parameters()).device.type == "cpu"
        assert not events_path.exists()


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cache_root = tmp_path_factory.mktemp("difnet-v0b-cache") / "v0b-cache-test"
    try:
        build_difnet_v0b_cache(
            CacheBuildOptions(
                project_root=PROJECT_ROOT,
                output_root=cache_root,
                views=("view_2",),
                projection_size=32,
                target_size=32,
                max_train_cases=2,
                max_test_cases=2,
                overwrite=True,
                command="pytest tiny cache",
                hu_sample_voxels_per_case=1000,
            )
        )
    except RuntimeError as exc:
        pytest.skip(f"self_projection1208 cache fixture is unavailable in this checkout: {exc}")
    return cache_root


def test_cache_manifest_records_v0b_contract(tiny_cache: Path) -> None:
    manifest = json.loads((tiny_cache / "cache_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "x2cbct_difnet_v0b_cache.v2"
    assert manifest["geometry_branch"] == "real_projection"
    assert manifest["axis_convention"] == axis_convention()
    assert manifest["projection_normalization"]["max"] == 8.6
    assert manifest["cbct_normalization"]["mode"] == "fixed"
    assert manifest["cbct_normalization"]["profile_id"] == "self_projection1208_fixed_hu_-290_2060"
    assert manifest["cbct_normalization"]["hu_min"] == pytest.approx(-290.0)
    assert manifest["cbct_normalization"]["hu_max"] == pytest.approx(2060.0)
    assert manifest["cbct_normalization"]["hu_span"] == pytest.approx(2350.0)
    assert manifest["cbct_normalization_estimation"]["fixed_hu_window"] == [-290.0, 2060.0]
    assert manifest["cbct_normalization_estimation"]["case_stats"] == []
    assert manifest["split"]["patient_overlap"] == []
    assert manifest["split"]["has_validation_split"] is False
    assert manifest["point_sampling"]["fixed_training_points_saved"] is False
    assert manifest["point_sampling"]["name"] == "official_interleaved_lattice"
    assert manifest["target_resize"]["size"] == 32
    assert manifest["view_sample_counts"]["view_2"] == {"train": 2, "test": 2}


def test_cache_builder_can_write_case_mixed_split_manifest(tmp_path: Path) -> None:
    cache_root = tmp_path / "difnet-v0b-mixed-cache"
    manifest = build_difnet_v0b_cache(
        CacheBuildOptions(
            project_root=PROJECT_ROOT,
            output_root=cache_root,
            views=("view_2",),
            projection_size=32,
            target_size=32,
            max_train_cases=2,
            max_test_cases=2,
            overwrite=True,
            dry_run=True,
            command="pytest mixed split cache",
            hu_window_mode="legacy",
            split_policy="mixed",
            split_train_ratio=0.5,
            split_seed=0,
        )
    )
    if not manifest["samples"]:
        pytest.skip("self_projection1208 cache fixture is unavailable in this checkout")

    assert manifest["split_manifest"] is None
    assert manifest["split"]["policy"] == CASE_MIXED_POLICY
    assert manifest["split"]["allow_patient_overlap"] is True
    assert manifest["split"]["case_overlap_count"] == 0
    assert manifest["view_sample_counts"]["view_2"] == {"train": 2, "test": 2}


def test_cache_builder_expands_setup_augmentation_inside_each_split(tmp_path: Path) -> None:
    cache_root = tmp_path / "difnet-v1-aug-cache"
    try:
        manifest = build_difnet_v0b_cache(
            CacheBuildOptions(
                project_root=PROJECT_ROOT,
                output_root=cache_root,
                views=("view_2",),
                projection_size=32,
                target_size=32,
                max_train_cases=1,
                max_test_cases=1,
                overwrite=True,
                dry_run=True,
                command="pytest setup augmentation cache",
                hu_window_mode="legacy",
                setup_augmentation=setup_augmentation_config(
                    enabled=True,
                    translation_mm=["-4", "0", "4"],
                    rotation_deg=["-4", "0", "4"],
                ),
            )
        )
    except RuntimeError as exc:
        pytest.skip(f"self_projection1208 cache fixture is unavailable in this checkout: {exc}")

    assert manifest["setup_augmentation"]["enabled"] is True
    assert manifest["setup_augmentation"]["combination_count_per_source_case"] == 9
    assert manifest["view_sample_counts"]["view_2"] == {"train": 9, "test": 9}
    assert all("setup_augmentation" in sample for sample in manifest["samples"])
    assert {sample["setup_augmentation"]["augmentation_id"] for sample in manifest["samples"]} == {
        "si_m4_rot_m4",
        "si_m4_rot_0",
        "si_m4_rot_p4",
        "si_0_rot_m4",
        "si_0_rot_0",
        "si_0_rot_p4",
        "si_p4_rot_m4",
        "si_p4_rot_0",
        "si_p4_rot_p4",
    }


def _write_source_sample(
    root: Path,
    *,
    source_case: str,
    patient_id: str,
    case_id: str,
    split: str,
    augmentation_id: str,
) -> dict[str, object]:
    case_key = f"{source_case}_{augmentation_id}"
    case_dir = root / "cases" / case_key
    projection_dir = root / "projections" / case_key
    sample_dir = root / "view_2" / split
    case_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "cbct_norm.npy", np.zeros((4, 4, 4), dtype=np.float16))
    np.save(projection_dir / "projection_000.npy", np.zeros((4, 4), dtype=np.float16))
    (case_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_key": case_key,
                "patient_id": patient_id,
                "case_id": case_id,
                "raw_case_path": str(PROJECT_ROOT / "dataset_list" / "self_projection1208" / patient_id / case_id),
                "cbct_payload_path": f"cases/{case_key}/cbct_norm.npy",
                "processed_shape_zyx": [4, 4, 4],
                "processed_spacing_xyz": [1.0, 1.0, 1.0],
                "processed_origin_xyz": [0.0, 0.0, 0.0],
                "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "projection_geometry": {"type": "real_projection", "rtk_geometry_path": "Geometry"},
                "point_blocks": official_lattice_metadata(4),
            }
        ),
        encoding="utf-8",
    )
    sample_id = f"{case_key}_view_2"
    np.savez_compressed(
        sample_dir / f"{sample_id}.npz",
        sample_id=np.array(sample_id),
        split=np.array(split),
        view_key=np.array("view_2"),
        case_key=np.array(case_key),
        patient_id=np.array(patient_id),
        case_id=np.array(case_id),
        case_manifest_path=np.array(f"cases/{case_key}/case_manifest.json"),
        cbct_payload_path=np.array(f"cases/{case_key}/cbct_norm.npy"),
        projection_payload_paths=np.array([f"projections/{case_key}/projection_000.npy"]),
        projection_files=np.array(["projection_000.npy"]),
        target_angles_deg=np.array([0.0], dtype=np.float32),
        gantry_angles_deg=np.array([0.0], dtype=np.float32),
        angular_residuals_deg=np.array([0.0], dtype=np.float32),
        detector_shape=np.array([4, 4], dtype=np.int32),
        detector_spacing_yx=np.array([1.0, 1.0], dtype=np.float32),
        projection_size=np.array(4, dtype=np.int32),
        augmentation_id=np.array(augmentation_id),
        translation_mm=np.array(0, dtype=np.int32),
        rotation_deg=np.array(0, dtype=np.int32),
        source_patient_id=np.array(patient_id),
        source_case_id=np.array(case_id),
        source_case_key=np.array(source_case),
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "view_key": "view_2",
        "geometry_branch": "real_projection",
        "patient_id": patient_id,
        "case_id": case_id,
        "sample_path": f"view_2/{split}/{sample_id}.npz",
        "case_key": case_key,
        "case_manifest_path": f"cases/{case_key}/case_manifest.json",
        "cbct_payload_path": f"cases/{case_key}/cbct_norm.npy",
        "projection_payload_paths": [f"projections/{case_key}/projection_000.npy"],
        "selected_projections": [{"payload_path": f"projections/{case_key}/projection_000.npy"}],
        "setup_augmentation": {
            "augmentation_id": augmentation_id,
            "translation_mm": 0,
            "rotation_deg": 0,
            "source_patient_id": int(patient_id),
            "source_case_id": int(case_id),
            "source_case_key": source_case,
        },
    }


def _write_source_augmented_cache(root: Path) -> None:
    samples = []
    for source_case, patient_id, case_id, split in (
        ("patient1_case0", "1", "0", "train"),
        ("patient1_case1", "1", "1", "test"),
    ):
        for augmentation_id in ("si_0_rot_0", "si_p4_rot_0"):
            samples.append(
                _write_source_sample(
                    root,
                    source_case=source_case,
                    patient_id=patient_id,
                    case_id=case_id,
                    split=split,
                    augmentation_id=augmentation_id,
                )
            )
    manifest = {
        "schema": "x2cbct_difnet_v0b_cache.v2",
        "geometry_branch": "real_projection",
        "axis_convention": axis_convention(),
        "cache_version": root.name,
        "calibration_report": {
            "path": str(PROJECT_ROOT / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"),
            "status": "pass",
            "candidate_id": "test",
            "adapter_version": "test",
        },
        "views": {"view_2": [0]},
        "target_resize": {"size": 4},
        "projection_resize": {"size": 4},
        "projection_normalization": {"max": 8.6},
        "cbct_normalization": {"mode": "fixed", "profile_id": "test"},
        "point_sampling": {
            **official_lattice_metadata(4),
            "fixed_training_points_saved": False,
        },
        "split": {
            "policy": CASE_MIXED_POLICY,
            "split_unit": "case",
            "seed": 0,
            "train_ratio": 0.5,
            "patient_overlap": ["1"],
            "allow_patient_overlap": True,
            "case_overlap_count": 0,
            "has_validation_split": False,
        },
        "setup_augmentation": {
            "enabled": True,
            "translation_mm": [0, 4],
            "rotation_deg": [0],
            "combination_count_per_source_case": 2,
        },
        "samples": samples,
        "failed_cases": [],
    }
    (root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_derive_real_cache_base_uses_source_case_split_and_absolute_assets(tmp_path: Path) -> None:
    source_root = tmp_path / "source_augmented"
    _write_source_augmented_cache(source_root)
    output_root = tmp_path / "derived_mixed_base"

    manifest = derive_real_difnet_cache(
        source_cache_root=source_root,
        output_root=output_root,
        split_policy="mixed",
        augmentation_selection="base",
        overwrite=True,
    )

    assert manifest["setup_augmentation"]["enabled"] is False
    assert manifest["view_sample_counts"]["view_2"] == {"train": 1, "test": 1}
    assert {sample["sample_id"] for sample in manifest["samples"]} == {
        "patient1_case0_view_2",
        "patient1_case1_view_2",
    }
    train_sample = output_root / "view_2" / "train" / "patient1_case0_view_2.npz"
    with np.load(train_sample, allow_pickle=False) as raw:
        assert str(raw["split"].item()) == "train"
        assert str(raw["case_key"].item()) == "patient1_case0"
        assert "augmentation_id" not in raw.files
        assert Path(str(raw["cbct_payload_path"].item())).is_absolute()


def test_derive_real_cache_patient_augmented_keeps_source_case_inside_one_split(tmp_path: Path) -> None:
    source_root = tmp_path / "source_augmented"
    _write_source_augmented_cache(source_root)
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(
        json.dumps(
            {
                "protocol": "test_no_val",
                "seed": 123,
                "test_ratio": 0.5,
                "train": {"cases": [{"patient_id": "1", "case_id": "0"}]},
                "test": {"cases": [{"patient_id": "1", "case_id": "1"}]},
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "derived_patient_aug"

    manifest = derive_real_difnet_cache(
        source_cache_root=source_root,
        output_root=output_root,
        split_policy="patient",
        augmentation_selection="all",
        patient_split_manifest=split_manifest,
        overwrite=True,
    )

    splits_by_source: dict[str, set[str]] = {}
    for sample in manifest["samples"]:
        source = sample["setup_augmentation"]["source_case_key"]
        splits_by_source.setdefault(source, set()).add(sample["split"])
    assert splits_by_source == {"patient1_case0": {"train"}, "patient1_case1": {"test"}}
    assert manifest["view_sample_counts"]["view_2"] == {"train": 2, "test": 2}
    assert manifest["split"]["policy"] == PATIENT_LEVEL_POLICY
    assert manifest["split"]["case_overlap_count"] == 0


def test_patient_internal_projection_correction_uses_one_offset_for_dark_cases() -> None:
    profile = _patient_dark_to_bright_correction_profile(
        [
            {"patient_id": 24, "case_id": 8, "case_key": "patient24_case8", "norm_mean": 0.30},
            {"patient_id": 24, "case_id": 9, "case_key": "patient24_case9", "norm_mean": 0.34},
            {"patient_id": 24, "case_id": 2, "case_key": "patient24_case2", "norm_mean": 0.56},
            {"patient_id": 24, "case_id": 3, "case_key": "patient24_case3", "norm_mean": 0.58},
            {"patient_id": 27, "case_id": 6, "case_key": "patient27_case6", "norm_mean": 0.25},
            {"patient_id": 27, "case_id": 0, "case_key": "patient27_case0", "norm_mean": 0.50},
        ]
    )

    assert profile["mode"] == PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN
    assert profile["case_offset_table"]["patient24_case8"]["delta_norm"] == pytest.approx(0.57 - 0.32)
    assert profile["case_offset_table"]["patient24_case9"]["delta_norm"] == pytest.approx(0.57 - 0.32)
    assert profile["case_offset_table"]["patient24_case8"]["delta_raw"] == pytest.approx((0.57 - 0.32) * 8.6)
    assert profile["case_offset_table"]["patient27_case6"]["delta_norm"] == pytest.approx(0.25)


def test_projection_correction_profile_loader_validates_contract(tmp_path: Path) -> None:
    profile_path = tmp_path / "projection_correction.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": PROJECTION_CORRECTION_PROFILE_SCHEMA,
                "mode": PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
                "source_view_key": "view_10",
                "source_projection_size": 256,
                "case_offset_table": {"patient11_case4": {"delta_norm": 0.2, "delta_raw": 1.72}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profile = _load_projection_correction_profile(
        profile_path,
        expected_mode=PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
        projection_size=256,
        view_key="view_10",
    )

    assert profile["profile_source"] == "external_profile_file"
    assert profile["case_offset_table"]["patient11_case4"]["delta_raw"] == pytest.approx(1.72)

    with pytest.raises(ValueError, match="projection-size mismatch"):
        _load_projection_correction_profile(
            profile_path,
            expected_mode=PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
            projection_size=128,
            view_key="view_10",
        )


def test_volume_grid_axis_mapping_keeps_array_zyx_and_spacing_xyz_separate() -> None:
    assert shape_zyx_to_size_xyz((4, 5, 6)) == (6, 5, 4)
    assert physical_fov_mm_xyz((6, 5, 4), (0.7, 0.8, 2.5)) == pytest.approx((3.5, 3.2, 7.5))
    assert resized_spacing_preserve_fov_xyz((4, 5, 6), (0.7, 0.8, 2.5), (8, 8, 8)) == pytest.approx(
        (0.5, 0.457142857, 1.071428571)
    )


def test_metric_dict_uses_manifest_hu_span() -> None:
    metrics = metric_dict(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.1], dtype=np.float32),
        normalization={"hu_min": -100.0, "hu_max": 900.0, "hu_span": 1000.0},
    )

    assert metrics["mae_hu"] == pytest.approx(100.0)


def test_metric_dict_uses_skimage_3d_ssim() -> None:
    z, y, x = np.meshgrid(
        np.linspace(0.0, 1.0, 8, dtype=np.float32),
        np.linspace(0.0, 1.0, 8, dtype=np.float32),
        np.linspace(0.0, 1.0, 8, dtype=np.float32),
        indexing="ij",
    )
    target = ((z * 0.5) + (y * 0.3) + (x * 0.2)).astype(np.float32)
    pred = np.clip(target + (0.04 * np.sin(2.0 * np.pi * x)).astype(np.float32), 0.0, 1.0)

    metrics = metric_dict(pred, target)

    assert metrics["ssim"] == pytest.approx(structural_similarity(target, pred, data_range=1.0))


def test_synthrad_base_reference_target_samples_match_materialized_setup(tmp_path: Path) -> None:
    cache_root = tmp_path / "synthrad_base_reference"
    cache_root.mkdir(parents=True)
    z, y, x = np.meshgrid(
        np.arange(8, dtype=np.float32),
        np.arange(8, dtype=np.float32),
        np.arange(8, dtype=np.float32),
        indexing="ij",
    )
    base_volume = z * 100.0 + y * 10.0 + x
    base_path = cache_root / "base_ct_norm.npy"
    np.save(base_path, base_volume.astype(np.float32))
    projection_paths = []
    for index in range(2):
        projection = np.full((8, 8), float(index + 1), dtype=np.float32)
        path = cache_root / f"projection_{index:03d}.npy"
        np.save(path, projection)
        projection_paths.append(str(path))
    fill_value = volume_background_value(base_volume)
    setup = {
        "augmentation_id": "si_p2_rot_p6",
        "translation_mm": 2,
        "rotation_deg": 6,
        "source_case_id": "case0",
        "volume_fill_value": fill_value,
    }
    manifest = {
        "schema": "x2cbct_synthrad2025_task2_view_cache.v1",
        "geometry_branch": "simulated_drr",
        "axis_convention": axis_convention(),
        "source": "luna16",
        "anatomy": "luna16",
        "dataset_group": "luna16",
        "cache_version": "unit",
        "target_volume_mode": "base_reference",
        "target_size": 8,
        "projection_size": 8,
        "views": {"view_2": [0, 6]},
        "view_counts": {"view_2": 2},
        "point_sampling": official_lattice_metadata(8),
        "drr_config": {
            "projection_count": 10,
            "total_angle_deg": 180.0,
            "start_angle_deg": 0.0,
            "detector_size": 8,
            "voxel_size": 8,
            "dsd_mm": 1200.0,
            "dso_mm": 800.0,
            "detector_spacing_mm": 1.0,
            "voxel_spacing_mm": 1.0,
        },
        "samples": [
            {
                "sample_id": "luna16_case0_si_p2_rot_p6_view_2",
                "source": "luna16",
                "anatomy": "luna16",
                "dataset_group": "luna16",
                "case_id": "case0_si_p2_rot_p6",
                "split": "train",
                "view_key": "view_2",
                "view_count": 2,
                "view_indices": [0, 6],
                "angles_deg": [0.0, 108.0],
                "projection_paths": projection_paths,
                "processed_cbct_path": str(base_path),
                "processed_shape_zyx": [8, 8, 8],
                "processed_spacing_xyz": [1.0, 1.0, 1.0],
                "processed_origin_xyz": [0.0, 0.0, 0.0],
                "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "target_volume_mode": "base_reference",
                "setup_augmentation": setup,
            }
        ],
    }
    (cache_root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset = CachedDifNetV0bDataset(
        cache_root,
        split="train",
        view_key="view_2",
        num_points=8,
        deterministic=True,
        seed=99,
    )

    item = dataset[0]
    indices, _ = sample_official_lattice_indices(8, np.random.default_rng(99), grid_size=8)
    materialized = transform_volume_setup(
        base_volume,
        translation_mm=2,
        rotation_deg=6,
        spacing_z_mm=1.0,
        fill_value=fill_value,
    )

    assert item["trace"]["target_volume_mode"] == "base_reference"
    assert item["p_gt"].reshape(-1) == pytest.approx(
        materialized[indices[:, 0], indices[:, 1], indices[:, 2]],
        abs=2e-4,
    )


def test_synthrad_view_subsets_come_from_fixed_view10_indices() -> None:
    assert synthrad_view_indices(10, "view_2") == [0, 6]
    assert synthrad_view_indices(10, "view_5") == [0, 2, 4, 6, 8]
    assert synthrad_view_indices(10, "view_10") == list(range(10))


def test_official_lattice_point_blocks_cover_full_grid() -> None:
    metadata = official_lattice_metadata(grid_size=256)
    rng = np.random.default_rng(20260525)
    indices, trace = sample_official_lattice_indices(4096, rng, grid_size=256, block_index=0)

    assert metadata["block_count"] == 64
    assert metadata["points_per_block"] == 64 * 64 * 64
    assert metadata["blocks"][0]["offset_zyx"] == [0, 0, 0]
    assert metadata["blocks"][-1]["offset_zyx"] == [3, 3, 3]
    assert indices.shape == (4096, 3)
    assert indices.min() >= 0
    assert indices.max() <= 252
    assert set(np.unique(indices % 4)) == {0}
    assert trace["point_sampling"] == "official_interleaved_lattice"


def test_cache_layout_has_shared_case_payload_and_light_view_samples(tiny_cache: Path) -> None:
    manifest = json.loads((tiny_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    sample_path = tiny_cache / manifest["samples"][0]["sample_path"]

    assert sample_path.exists()
    assert (tiny_cache / "cases").exists()
    assert (tiny_cache / "projections").exists()
    with np.load(sample_path, allow_pickle=False) as sample:
        assert "projection_payload_paths" in sample.files
        assert "cbct_payload_path" in sample.files
        assert "points" not in sample.files
        assert "p_gt" not in sample.files
    case_manifest = json.loads((tiny_cache / manifest["samples"][0]["case_manifest_path"]).read_text(encoding="utf-8"))
    sample_manifest = manifest["samples"][0]
    assert case_manifest["geometry_branch"] == "real_projection"
    assert case_manifest["processed_shape_zyx"] == [32, 32, 32]
    assert case_manifest["projection_geometry"]["type"] == "real_projection"
    assert sample_manifest["projection_geometry"]["type"] == "real_projection"


def test_cached_dataset_returns_dynamic_difnet_item(tiny_cache: Path) -> None:
    dataset = CachedDifNetV0bDataset(
        tiny_cache,
        split="train",
        view_key="view_2",
        num_points=12,
        deterministic=True,
    )

    first = dataset[0]
    second = dataset[0]

    assert first["projs"].shape == (2, 1, 32, 32)
    assert first["points"].shape == (12, 3)
    assert first["proj_points"].shape == (2, 12, 2)
    assert first["p_gt"].shape == (1, 12)
    assert first["valid_mask"].dtype == bool
    assert first["view_angles_deg"].shape == (2,)
    assert np.allclose(first["view_angles_deg"], first["trace"]["gantry_angles_deg"])
    assert np.isfinite(first["proj_points"]).all()
    assert np.allclose(first["points"], second["points"])
    assert first["trace"]["candidate_id"]
    assert first["trace"]["geometry_branch"] == "real_projection"
    assert first["trace"]["processed_shape_zyx"] == [32, 32, 32]
    assert len(first["trace"]["processed_spacing_xyz"]) == 3
    assert first["trace"]["point_sampling"]["point_sampling"] == "official_interleaved_lattice"


def test_training_loss_can_mask_invalid_projection_points() -> None:
    pred = torch.tensor([[[1.0, 2.0, 3.0]]])
    gt = torch.zeros_like(pred)
    valid_mask = torch.tensor([[[True, True, False], [True, False, False]]])

    any_loss, any_ratio = _masked_mse(pred, gt, valid_mask, "any")
    all_loss, all_ratio = _masked_mse(pred, gt, valid_mask, "all")
    full_loss, full_ratio = _masked_mse(pred, gt, valid_mask, "none")

    assert any_loss.item() == pytest.approx(2.5)
    assert any_ratio == pytest.approx(2 / 3)
    assert all_loss.item() == pytest.approx(1.0)
    assert all_ratio == pytest.approx(1 / 3)
    assert full_loss.item() == pytest.approx(14 / 3)
    assert np.isnan(full_ratio)


def test_patient_round_robin_sample_selection_spreads_patients() -> None:
    samples = [
        {"patient_id": "1", "sample_id": "patient1_case0"},
        {"patient_id": "1", "sample_id": "patient1_case1"},
        {"patient_id": "2", "sample_id": "patient2_case0"},
        {"patient_id": "3", "sample_id": "patient3_case0"},
    ]

    selected = _patient_round_robin_samples(samples, 4)

    assert [item["sample_id"] for item in selected] == [
        "patient1_case0",
        "patient2_case0",
        "patient3_case0",
        "patient1_case1",
    ]


def test_split_policy_can_mix_cases_within_the_same_patient() -> None:
    rows = [
        {"patient_id": "p1", "case_id": "case0"},
        {"patient_id": "p1", "case_id": "case1"},
        {"patient_id": "p2", "case_id": "case0"},
        {"patient_id": "p3", "case_id": "case0"},
    ]

    patient_split = assign_train_test_split(rows, policy=PATIENT_LEVEL_POLICY, train_ratio=0.5, seed=0)
    mixed_split = assign_train_test_split(rows, policy="mixed", train_ratio=0.5, seed=0)

    assert patient_split.metadata["policy"] == PATIENT_LEVEL_POLICY
    assert patient_split.metadata["patient_overlap"] == []
    assert mixed_split.metadata["policy"] == CASE_MIXED_POLICY
    assert mixed_split.metadata["allow_patient_overlap"] is True
    assert mixed_split.metadata["patient_overlap"] == ["p1"]
    assert mixed_split.metadata["case_overlap_count"] == 0
    assert mixed_split.metadata["train_case_count"] == 2
    assert mixed_split.metadata["test_case_count"] == 2


def test_local_cache_manifest_accepts_case_mixed_patient_overlap(tmp_path: Path) -> None:
    cache_root = tmp_path / "mixed_cache"
    cache_root.mkdir()
    (cache_root / "cache_manifest.json").write_text(
        json.dumps(
            {
                "schema": "x2cbct_difnet_v0b_cache.v2",
                "geometry_branch": "real_projection",
                "axis_convention": axis_convention(),
                "split": {
                    "policy": CASE_MIXED_POLICY,
                    "allow_patient_overlap": True,
                    "patient_overlap": ["p1"],
                    "case_overlap_count": 0,
                },
                "calibration_report": {"status": "pass"},
                "point_sampling": {
                    "fixed_training_points_saved": False,
                    "name": "official_interleaved_lattice",
                },
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_cache_manifest(cache_root)

    assert manifest["split"]["policy"] == CASE_MIXED_POLICY
    assert manifest["split"]["patient_overlap"] == ["p1"]


def test_default_prediction_export_grid_is_128() -> None:
    assert DEFAULT_VALIDATION_EVAL_GRID_SIZE == 128
    assert DEFAULT_TEST_EVAL_GRID_SIZE == 128
    assert DEFAULT_TEST_VISUALIZATION_MAX_CASES == 50
    assert DEFAULT_TEST_VIEWER_MAX_CASES == 50
    assert DEFAULT_TEST_VIEWER_SLICE_STRIDE == 4


def test_prediction_viewer_auto_windows_dark_normalized_volumes() -> None:
    array = np.zeros((8, 8), dtype=np.float32)
    array[2:6, 2:6] = 0.08
    window = _display_window(array)
    panel = _to_panel(array, display_window=window)

    assert 0.07 <= window["high"] <= 0.081
    assert int(panel.max()) == 255


def test_sample_viewer_exports_interactive_html(tiny_cache: Path, tmp_path: Path) -> None:
    manifest = json.loads((tiny_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    sample_path = tiny_cache / manifest["samples"][0]["sample_path"]
    output_dir = tmp_path / "viewer"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_difnet_v0b_sample.py"),
            "--cache-root",
            str(tiny_cache),
            "--sample-path",
            str(sample_path),
            "--slice-stride",
            "16",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    html = (output_dir / "index.html").read_text(encoding="utf-8")
    viewer_data = json.loads((output_dir / "viewer_data.json").read_text(encoding="utf-8"))

    assert "DIF-Net V0b Interactive Viewer" in html
    assert "projectionSlider" in html
    assert "axialSlider" in html
    assert "viewer_data.js" in html
    assert viewer_data["schema"] == "x2cbct_difnet_v0b_interactive_viewer.v1"
    assert viewer_data["projections"]
    assert viewer_data["geometry_branch"] == "real_projection"
    assert viewer_data["cbct"]["display_policy"] == "spacing_corrected_by_processed_spacing_xyz"
    assert viewer_data["cbct"]["processed_spacing_xyz"]
    assert viewer_data["cbct"]["planes"]["axial"]["slices"]
    assert (output_dir / "viewer_data.js").exists()


def test_cache_build_watcher_exports_partial_cache_dashboard(tmp_path: Path) -> None:
    cache_root = tmp_path / "partial_cache"
    case_dir = cache_root / "cases" / "patient1_case0"
    projection_dir = cache_root / "projections"
    sample_dir = cache_root / "view_2" / "train"
    case_dir.mkdir(parents=True)
    projection_dir.mkdir(parents=True)
    sample_dir.mkdir(parents=True)
    np.save(case_dir / "cbct_norm.npy", np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8))
    np.save(projection_dir / "patient1_case0_proj0.npy", np.eye(8, dtype=np.float32))
    np.savez_compressed(
        sample_dir / "patient1_case0_view_2.npz",
        sample_id=np.array("patient1_case0_view_2"),
        split=np.array("train"),
        view_key=np.array("view_2"),
        case_key=np.array("patient1_case0"),
        cbct_payload_path=np.array("cases/patient1_case0/cbct_norm.npy"),
        projection_payload_paths=np.array(["projections/patient1_case0_proj0.npy"]),
    )
    output_dir = tmp_path / "watch"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "watch_difnet_v0b_cache_build.py"),
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(output_dir),
            "--view-count",
            "2",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    status = json.loads((output_dir / "watch_status.json").read_text(encoding="utf-8"))
    assert status["schema"] == "x2cbct_difnet_v0b_cache_build_watch.v1"
    assert status["manifest_exists"] is False
    assert status["counts"] == {"case_payloads": 1, "projection_payloads": 1, "samples": 1}
    assert (output_dir / "index.html").exists()
    assert list((output_dir / "assets").glob("*.png"))


def test_batch_sample_viewer_uses_case_named_output_dirs(tiny_cache: Path, tmp_path: Path) -> None:
    output_root = tmp_path / "batch_viewers"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "export_difnet_v0b_sample_viewers.py"),
            "--cache-root",
            str(tiny_cache),
            "--split",
            "all",
            "--view-count",
            "2",
            "--max-samples",
            "1",
            "--slice-stride",
            "16",
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    batch_index = json.loads((output_root / "viewer_batch_index.json").read_text(encoding="utf-8"))
    train_row = next(row for row in batch_index["rows"] if row["split"] == "train")
    test_row = next(row for row in batch_index["rows"] if row["split"] == "test")

    assert (output_root / "index.html").exists()
    assert batch_index["schema"] == "x2cbct_difnet_v0b_sample_viewer_batch.v1"
    assert batch_index["split"] == "all"
    assert batch_index["split_counts"] == {"train": 1, "test": 1}
    assert [row["display_order"] for row in batch_index["rows"]] == [0, 1]
    for row in (train_row, test_row):
        assert row["sample_id"].startswith("patient")
        assert row["patient_id"] != ""
        assert row["case_id"] != ""
        assert f"patient{row['patient_id']}" in row["viewer_dir"]
        assert f"case{row['case_id']}" in row["viewer_dir"]
        assert row["viewer_relative_path"].startswith(f"{row['split']}/")
        assert (Path(row["viewer_dir"]) / "index.html").exists()
        viewer_html = (Path(row["viewer_dir"]) / "index.html").read_text(encoding="utf-8")
        viewer_data = json.loads((Path(row["viewer_dir"]) / "viewer_data.json").read_text(encoding="utf-8"))
        assert "sampleNav" in viewer_html
        assert viewer_data["navigation"]["index"]["href"].endswith("index.html")
        assert viewer_data["navigation"]["total"] == 2
    assert json.loads((Path(train_row["viewer_dir"]) / "viewer_data.json").read_text(encoding="utf-8"))["navigation"][
        "next"
    ]["split"] == "test"
    assert json.loads((Path(test_row["viewer_dir"]) / "viewer_data.json").read_text(encoding="utf-8"))["navigation"][
        "previous"
    ]["split"] == "train"


def test_train_test_visualization_smoke_outputs_artifacts(tiny_cache: Path, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_run = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "train_difnet_v0b.py"),
            "--cache-root",
            str(tiny_cache),
            "--view-count",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-points",
            "8",
            "--steps-per-epoch",
            "2",
            "--validation-steps",
            "1",
            "--full-test-interval",
            "1",
            "--validation-eval-grid-size",
            "8",
            "--validation-chunk-size",
            "256",
            "--validation-train-device-policy",
            "offload-to-cpu",
            "--viewer-slice-stride",
            "8",
            "--mid-channels",
            "4",
            "--output-dir",
            str(train_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    checkpoint = train_dir / "checkpoints" / "epoch_0001.pt"
    config = json.loads((train_dir / "config_resolved.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((train_dir / "metrics_final.json").read_text(encoding="utf-8"))
    validation_plan = json.loads((train_dir / "validation_plan.json").read_text(encoding="utf-8"))
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert checkpoint.exists()
    assert not (train_dir / "checkpoints" / "latest.pt").exists()
    assert not (train_dir / "checkpoints" / "best_by_mae_norm.pt").exists()
    assert (train_dir / "checkpoints" / "best_by_psnr.pt").exists()
    assert (train_dir / "validation_plan.json").exists()
    assert (train_dir / "metrics_final.json").exists()
    assert (train_dir / "events.jsonl").exists()
    assert config["requested_device"] == "auto"
    assert config["device"] == expected_device
    assert validation_plan["blocking"] is True
    assert config["loss_valid_policy"] == "any"
    assert config["validation_train_device_policy"] == "offload-to-cpu"
    assert validation_plan["full_test"]["interval_epochs"] == 1
    assert validation_plan["train_device_policy"] == "offload-to-cpu"
    assert validation_plan["non_finite_metrics_are_execution_failures"] is True
    assert final_metrics["best_by_psnr"]["checkpoint"].endswith("epoch_0001.pt")
    saved_checkpoint = torch.load(checkpoint, map_location="cpu")
    assert saved_checkpoint["optimizer_step"] == 2
    assert "scheduler_state" in saved_checkpoint
    assert train_run.stdout.count("[train]") == 1
    assert "step=0001" not in train_run.stdout
    events = (train_dir / "events.jsonl").read_text(encoding="utf-8")
    if expected_device == "cuda":
        assert '"event": "validation_train_device_offload_start"' in events
        assert '"event": "validation_train_device_restore_end"' in events
    validation_dir = train_dir / "validation" / "epoch_0001_full_final"
    validation_metrics = json.loads((validation_dir / "test" / "metrics_summary.json").read_text(encoding="utf-8"))
    assert validation_metrics["case_count"] == 2
    assert (validation_dir / "test" / "prediction_manifest.json").exists()
    assert (validation_dir / "test" / "metrics_summary.json").exists()
    assert (validation_dir / "comparisons" / "visualization_summary.json").exists()
    assert (validation_dir / "viewer" / "index.html").exists()
    assert (validation_dir / "viewer" / "viewer_data.json").exists()

    test_dir = tmp_path / "test"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "test_difnet_v0b.py"),
            "--cache-root",
            str(tiny_cache),
            "--view-count",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--eval-grid-size",
            "8",
            "--chunk-size",
            "256",
            "--max-test-cases",
            "1",
            "--mid-channels",
            "4",
            "--output-dir",
            str(test_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    assert (test_dir / "prediction_manifest.json").exists()
    assert (test_dir / "metrics_summary.json").exists()
    assert (test_dir / "comparisons" / "visualization_summary.json").exists()
    assert (test_dir / "viewer" / "index.html").exists()
    assert (test_dir / "viewer" / "viewer_data.json").exists()
    prediction_manifest = json.loads((test_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction_manifest["geometry_branch"] == "real_projection"
    assert prediction_manifest["predictions"][0]["processed_spacing_xyz"]
    assert prediction_manifest["predictions"][0]["geometry_branch"] == "real_projection"

    viz_dir = tmp_path / "viz"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "visualize_difnet_v0b_predictions.py"),
            "--prediction-manifest",
            str(test_dir / "prediction_manifest.json"),
            "--output-dir",
            str(viz_dir),
            "--max-cases",
            "1",
            "--slice-mode",
            "axial",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    viz_summary = json.loads((viz_dir / "visualization_summary.json").read_text(encoding="utf-8"))
    assert viz_summary["display_policy"] == "spacing_corrected_by_processed_spacing_xyz"
    assert viz_summary["records"][0]["processed_spacing_xyz"]
    assert list(viz_dir.glob("*/*.png"))

    viewer_dir = tmp_path / "prediction_viewer"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "view_difnet_v0b_prediction.py"),
            "--prediction-manifest",
            str(test_dir / "prediction_manifest.json"),
            "--output-dir",
            str(viewer_dir),
            "--slice-stride",
            "8",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    viewer_data = json.loads((viewer_dir / "viewer_data.json").read_text(encoding="utf-8"))
    viewer_html = (viewer_dir / "index.html").read_text(encoding="utf-8")
    assert viewer_data["schema"] == "x2cbct_difnet_v0b_prediction_viewer.v1"
    assert viewer_data["display_policy"] == "spacing_corrected_by_processed_spacing_xyz"
    assert viewer_data["cases"][0]["processed_spacing_xyz"]
    assert set(viewer_data["cases"][0]["volumes"]) == {"generated", "target"}
    assert "volumeSelect" not in viewer_html
    assert "error" not in viewer_data["cases"][0]["volumes"]
    assert (viewer_dir / "index.html").exists()
    assert (viewer_dir / "viewer_data.js").exists()
    asset_pngs = list((viewer_dir / "assets").glob("*/*/*.png"))
    assert asset_pngs
    assert not list((viewer_dir / "assets").glob("*/error/*.png"))
    assert next((viewer_dir / "assets").glob("*/generated/axial_*.png"))
    target_png = next((viewer_dir / "assets").glob("*/target/axial_*.png"))
    image = cv2.imread(str(target_png), cv2.IMREAD_COLOR)
    yellow_label_pixels = ((image[:, :, 1] > 180) & (image[:, :, 2] > 180) & (image[:, :, 0] < 80)).sum()
    assert yellow_label_pixels == 0


def test_train_without_test_epoch_keeps_no_checkpoint(tmp_path: Path) -> None:
    cache_root = tmp_path / "synthrad_cache"
    case_dir = cache_root / "cases" / "case0"
    projection_dir = cache_root / "projections" / "case0"
    case_dir.mkdir(parents=True)
    projection_dir.mkdir(parents=True)
    cbct_path = case_dir / "ct_norm.npy"
    projection_paths = [projection_dir / f"projection_{index:03d}.npy" for index in range(2)]
    np.save(cbct_path, np.zeros((32, 32, 32), dtype=np.float32))
    for path in projection_paths:
        np.save(path, np.zeros((32, 32), dtype=np.float32))
    samples = []
    for split in ("train", "test"):
        samples.append(
            {
                "sample_id": f"{split}_case0_view_2",
                "case_id": f"{split}_case0",
                "split": split,
                "view_key": "view_2",
                "view_count": 2,
                "angles_deg": [0.0, 90.0],
                "projection_paths": [str(path.resolve()) for path in projection_paths],
                "processed_cbct_path": str(cbct_path.resolve()),
                "processed_shape_zyx": [32, 32, 32],
                "processed_spacing_xyz": [1.0, 1.0, 1.0],
                "anatomy": "pytest",
                "source": "simulated_drr",
                "projector": "pytest",
                "projection_value_space": "normalized",
                "projection_loader_scale": 1.0,
            }
        )
    manifest = {
        "schema": "x2cbct_synthrad2025_task2_view_cache.v1",
        "source": "pytest",
        "anatomy": "pytest",
        "cache_version": "pytest_synthrad_cache",
        "geometry_branch": "simulated_drr",
        "projector": "pytest",
        "projection_value_space": "normalized",
        "projection_loader_scale": 1.0,
        "axis_convention": axis_convention(),
        "target_size": 32,
        "projection_size": 32,
        "views": {"view_2": [0.0, 90.0]},
        "view_counts": {"view_2": 2},
        "point_sampling": {
            **official_lattice_metadata(32),
            "fixed_training_points_saved": False,
        },
        "drr_config": {
            "projection_count": 2,
            "total_angle_deg": 180.0,
            "start_angle_deg": 0.0,
            "detector_size": 32,
            "voxel_size": 32,
            "dsd_mm": 1200.0,
            "dso_mm": 800.0,
            "detector_spacing_mm": 2.0,
            "voxel_spacing_mm": 1.0,
        },
        "samples": samples,
    }
    (cache_root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_dir = tmp_path / "train_non_centennial"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "train_difnet_v0b.py"),
            "--cache-root",
            str(cache_root),
            "--view-count",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--num-points",
            "8",
            "--steps-per-epoch",
            "1",
            "--validation-steps",
            "1",
            "--full-test-interval",
            "0",
            "--skip-final-validation",
            "--mid-channels",
            "4",
            "--device",
            "cpu",
            "--output-dir",
            str(train_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    checkpoint_dir = train_dir / "checkpoints"
    assert not (checkpoint_dir / "latest.pt").exists()
    assert not (checkpoint_dir / "epoch_0001.pt").exists()
    assert not (checkpoint_dir / "epoch_0002.pt").exists()
    assert not (train_dir / "validation").exists()
    final_metrics = json.loads((train_dir / "metrics_final.json").read_text(encoding="utf-8"))
    assert final_metrics["checkpoint"] is None
    assert final_metrics["validation_records"] == []


def test_train_refuses_to_reuse_non_empty_output_dir(tiny_cache: Path, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "old_metrics.json").write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "train_difnet_v0b.py"),
            "--cache-root",
            str(tiny_cache),
            "--view-count",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-points",
            "8",
            "--steps-per-epoch",
            "1",
            "--mid-channels",
            "4",
            "--output-dir",
            str(train_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "output dir already exists and is not empty" in completed.stderr


def test_train_resume_from_epoch_checkpoint_appends_to_existing_output_dir(tiny_cache: Path, tmp_path: Path) -> None:
    train_dir = tmp_path / "resume_train"
    base_args = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_difnet_v0b.py"),
        "--cache-root",
        str(tiny_cache),
        "--view-count",
        "2",
        "--batch-size",
        "1",
        "--num-points",
        "8",
        "--steps-per-epoch",
        "1",
        "--validation-steps",
        "1",
        "--full-test-interval",
        "1",
        "--mid-channels",
        "4",
        "--device",
        "cpu",
        "--output-dir",
        str(train_dir),
    ]

    subprocess.run([*base_args, "--epochs", "1"], check=True, cwd=PROJECT_ROOT)

    checkpoint = train_dir / "checkpoints" / "epoch_0001.pt"
    resumed = subprocess.run(
        [
            *base_args,
            "--epochs",
            "2",
            "--resume-checkpoint",
            str(checkpoint),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert "resume_loaded" in resumed.stdout
    assert torch.load(train_dir / "checkpoints" / "epoch_0002.pt", map_location="cpu")["epoch"] == 2
    assert torch.load(checkpoint, map_location="cpu")["epoch"] == 1
    events = (train_dir / "events.jsonl").read_text(encoding="utf-8")
    assert events.count('"event": "resume_loaded"') == 1
    assert events.count('"event": "validation_end"') == 2
    train_rows = (train_dir / "train_log.csv").read_text(encoding="utf-8").splitlines()
    assert len(train_rows) == 3
    final_metrics = json.loads((train_dir / "metrics_final.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in final_metrics["validation_records"]] == [1, 2]
