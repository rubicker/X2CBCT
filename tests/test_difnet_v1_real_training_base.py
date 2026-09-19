from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import CacheBuildOptions, build_difnet_v0b_cache
from x2cbct_difnet.setup_augmentation import setup_augmentation_config


def test_real_training_base_cache_uses_training_union_manifest(tmp_path: Path) -> None:
    fixed_manifest_path = PROJECT_ROOT / "data_manifest" / "fixed_view_sampling_v0b.json"
    if not fixed_manifest_path.exists():
        pytest.skip("self_projection1208 fixed-view manifest is unavailable")
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8-sig"))
    if not fixed_manifest.get("records"):
        pytest.skip("self_projection1208 fixed-view manifest has no records")
    first = fixed_manifest["records"][0]
    training_base_manifest = tmp_path / "self_projection1208_training_base_cases.json"
    training_base_manifest.write_text(
        json.dumps(
            {
                "schema": "test_training_base_manifest",
                "cases": [
                    {
                        "case_key": f"{first['patient_id']}/{first['case_id']}",
                        "patient_id": str(first["patient_id"]),
                        "case_id": str(first["case_id"]),
                        "case_path": str(first["case_path"]),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        manifest = build_difnet_v0b_cache(
            CacheBuildOptions(
                project_root=PROJECT_ROOT,
                output_root=tmp_path / "real-training-base-cache",
                views=("view_10",),
                projection_size=32,
                target_size=32,
                max_train_cases=1,
                max_test_cases=0,
                dry_run=True,
                overwrite=True,
                command="pytest real training base cache",
                hu_window_mode="legacy",
                setup_augmentation=setup_augmentation_config(
                    enabled=True,
                    translation_mm=["0"],
                    rotation_deg=["0"],
                ),
                target_volume_mode="base_reference",
                training_base_manifest_path=training_base_manifest,
                progress_interval=0,
            )
        )
    except RuntimeError as exc:
        pytest.skip(f"self_projection1208 data files are unavailable in this checkout: {exc}")

    assert manifest["split"]["policy"] == "training_base_manifest"
    assert manifest["split"]["train_case_count"] == 1
    assert manifest["split"]["test_case_count"] == 0
    assert manifest["view_sample_counts"]["view_10"] == {"train": 1, "test": 0}
    assert manifest["target_volume_mode"] == "base_reference"
    assert manifest["setup_augmentation"]["combination_count_per_source_case"] == 1
    assert manifest["split_manifest"]["path"] == str(training_base_manifest.resolve())
    assert {sample["split"] for sample in manifest["samples"]} == {"train"}
    assert {sample["view_key"] for sample in manifest["samples"]} == {"view_10"}


def test_real_independent_test_cache_uses_derivative_manifest_test_split(tmp_path: Path) -> None:
    fixed_manifest_path = PROJECT_ROOT / "data_manifest" / "fixed_view_sampling_v0b.json"
    if not fixed_manifest_path.exists():
        pytest.skip("self_projection1208 fixed-view manifest is unavailable")
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8-sig"))
    if not fixed_manifest.get("records"):
        pytest.skip("self_projection1208 fixed-view manifest has no records")
    first = fixed_manifest["records"][0]
    derivative_manifest = tmp_path / "self_projection1208_patient_noaug.json"
    derivative_manifest.write_text(
        json.dumps(
            {
                "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
                "name": "self_projection1208_patient_noaug",
                "dataset": "self_projection1208",
                "split_policy": "patient",
                "augmentation": {"enabled": False, "translation_mm": [0], "rotation_deg": [0]},
                "splits": {
                    "train": {"cases": []},
                    "test": {
                        "cases": [
                            {
                                "case_key": f"{first['patient_id']}/{first['case_id']}",
                                "patient_id": str(first["patient_id"]),
                                "case_id": str(first["case_id"]),
                                "case_path": str(first["case_path"]),
                            }
                        ]
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        manifest = build_difnet_v0b_cache(
            CacheBuildOptions(
                project_root=PROJECT_ROOT,
                output_root=tmp_path / "real-independent-test-cache",
                views=("view_10",),
                projection_size=32,
                target_size=32,
                max_train_cases=0,
                max_test_cases=1,
                dry_run=True,
                overwrite=True,
                command="pytest real independent test cache",
                hu_window_mode="legacy",
                setup_augmentation=setup_augmentation_config(enabled=False),
                target_volume_mode="full",
                records_manifest_path=derivative_manifest,
                records_manifest_split="test",
                records_output_split="test",
                progress_interval=0,
            )
        )
    except RuntimeError as exc:
        pytest.skip(f"self_projection1208 data files are unavailable in this checkout: {exc}")

    assert manifest["split"]["policy"] == "patient"
    assert manifest["split"]["train_case_count"] == 0
    assert manifest["split"]["test_case_count"] == 1
    assert manifest["target_volume_mode"] == "full"
    assert manifest["view_sample_counts"]["view_10"] == {"train": 0, "test": 1}
    assert manifest["split_manifest"]["path"] == str(derivative_manifest.resolve())
    assert {sample["split"] for sample in manifest["samples"]} == {"test"}
