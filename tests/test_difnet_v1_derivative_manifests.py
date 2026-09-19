from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.derivative_manifests import build_derivative_manifest_set


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fixed_case(patient_id: str, case_id: str, root: Path) -> dict[str, Any]:
    case_path = root / "dataset_list" / "self_projection1208" / patient_id / case_id
    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "case_path": str(case_path),
        "selected": {
            "view_10": [
                {
                    "target_angle_deg": angle,
                    "ordinal": index,
                    "file": f"{index}_{angle}.nii.gz",
                    "gantry_angle_deg": float(angle),
                    "angular_residual_deg": 0.0,
                }
                for index, angle in enumerate((200, 215, 230, 245, 260, 275, 290, 305, 320, 335))
            ],
            "view_5": [],
            "view_2": [],
        },
    }


def _case_ref(patient_id: str, case_id: str, root: Path) -> dict[str, str]:
    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "case_path": str(root / "dataset_list" / "self_projection1208" / patient_id / case_id),
    }


def test_builds_six_derivative_manifests_and_training_base_unions(tmp_path: Path) -> None:
    fixed_manifest = _write_json(
        tmp_path / "data_manifest" / "fixed_view_sampling_v0b.json",
        {
            "protocol": "fixture_fixed_view_sampling",
            "included_case_count": 4,
            "excluded_case_count": 0,
            "records": [
                _fixed_case("p1", "c0", tmp_path),
                _fixed_case("p1", "c1", tmp_path),
                _fixed_case("p2", "c0", tmp_path),
                _fixed_case("p3", "c0", tmp_path),
            ],
        },
    )
    patient_split_manifest = _write_json(
        tmp_path / "data_manifest" / "splits_v0b_train_test.json",
        {
            "protocol": "fixture_patient_split",
            "seed": 17,
            "test_ratio": 0.5,
            "train": {
                "patient_count": 1,
                "case_count": 2,
                "patients": ["p1"],
                "cases": [_case_ref("p1", "c0", tmp_path), _case_ref("p1", "c1", tmp_path)],
            },
            "eval_to_test": {
                "case_count": 1,
                "cases": [_case_ref("p1", "c1", tmp_path)],
            },
            "test": {
                "patient_count": 2,
                "case_count": 2,
                "patients": ["p2", "p3"],
                "cases": [_case_ref("p2", "c0", tmp_path), _case_ref("p3", "c0", tmp_path)],
            },
        },
    )
    luna16_registry = _write_json(
        tmp_path / "data_manifest" / "luna16_c2rv_split_registry.json",
        {
            "schema": "fixture_luna16_registry",
            "cases": [
                {"seriesuid": "luna_train", "patient_id": "luna_train", "ct_path": "train.mhd", "split": "train"},
                {
                    "seriesuid": "luna_eval",
                    "patient_id": "luna_eval",
                    "ct_path": "eval.mhd",
                    "split": "eval",
                    "upstream_split": "eval",
                    "split_source": "upstream_splits.json",
                },
                {"seriesuid": "luna_test", "patient_id": "luna_test", "ct_path": "test.mhd", "split": "test"},
            ],
        },
    )

    index = build_derivative_manifest_set(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        fixed_view_manifest_path=fixed_manifest,
        patient_split_manifest_path=patient_split_manifest,
        luna16_registry_path=luna16_registry,
        mixed_train_ratio=0.5,
        mixed_seed=3,
        command="pytest manifest builder",
    )

    manifest_paths = {item["name"]: Path(item["path"]) for item in index["derivative_manifests"]}
    assert set(manifest_paths) == {
        "self_projection1208_patient_noaug",
        "self_projection1208_mixed_noaug",
        "self_projection1208_patient_aug",
        "self_projection1208_mixed_aug",
        "luna16_noaug",
        "luna16_aug",
    }

    patient_noaug = json.loads(manifest_paths["self_projection1208_patient_noaug"].read_text(encoding="utf-8"))
    assert patient_noaug["dataset"] == "self_projection1208"
    assert patient_noaug["split_policy"] == "patient"
    assert patient_noaug["augmentation"]["enabled"] is False
    assert [item["case_key"] for item in patient_noaug["splits"]["train"]["cases"]] == ["p1/c0"]
    assert "p1/c1" in {item["case_key"] for item in patient_noaug["splits"]["test"]["cases"]}
    assert patient_noaug["split"]["eval_to_test_cases"] == ["p1/c1"]
    assert set(patient_noaug["splits"]) == {"train", "test"}

    mixed_noaug = json.loads(manifest_paths["self_projection1208_mixed_noaug"].read_text(encoding="utf-8"))
    assert "p1/c1" not in {item["case_key"] for item in mixed_noaug["splits"]["train"]["cases"]}
    assert "p1/c1" in {item["case_key"] for item in mixed_noaug["splits"]["test"]["cases"]}

    patient_aug = json.loads(manifest_paths["self_projection1208_patient_aug"].read_text(encoding="utf-8"))
    assert patient_aug["augmentation"]["translation_mm"] == [-10, 0, 10]
    assert patient_aug["augmentation"]["rotation_deg"] == [-10, 0, 10]
    assert patient_aug["augmentation"]["combination_count_per_source_case"] == 9

    real_base = json.loads(Path(index["training_base_manifests"]["self_projection1208"]).read_text(encoding="utf-8"))
    real_base_keys = {item["case_key"] for item in real_base["cases"]}
    for name in (
        "self_projection1208_patient_noaug",
        "self_projection1208_mixed_noaug",
        "self_projection1208_patient_aug",
        "self_projection1208_mixed_aug",
    ):
        manifest = json.loads(manifest_paths[name].read_text(encoding="utf-8"))
        train_keys = {item["case_key"] for item in manifest["splits"]["train"]["cases"]}
        assert train_keys <= real_base_keys

    luna_noaug = json.loads(manifest_paths["luna16_noaug"].read_text(encoding="utf-8"))
    assert luna_noaug["dataset"] == "luna16"
    assert luna_noaug["split_policy"] == "official_train_test"
    assert {item["case_key"] for item in luna_noaug["splits"]["train"]["cases"]} == {"luna_train"}
    assert {item["case_key"] for item in luna_noaug["splits"]["test"]["cases"]} == {"luna_eval", "luna_test"}
    assert set(luna_noaug["splits"]) == {"train", "test"}
    serialized_luna = json.dumps(luna_noaug, ensure_ascii=False)
    assert "upstream_split" not in serialized_luna
    assert "split_source" not in serialized_luna


def test_luna16_manifests_reject_unexpected_split(tmp_path: Path) -> None:
    fixed_manifest = _write_json(
        tmp_path / "data_manifest" / "fixed_view_sampling_v0b.json",
        {
            "protocol": "fixture_fixed_view_sampling",
            "included_case_count": 1,
            "excluded_case_count": 0,
            "records": [_fixed_case("p1", "c0", tmp_path)],
        },
    )
    patient_split_manifest = _write_json(
        tmp_path / "data_manifest" / "splits_v0b_train_test.json",
        {
            "protocol": "fixture_patient_split",
            "train": {"patient_count": 1, "case_count": 1, "patients": ["p1"], "cases": [_case_ref("p1", "c0", tmp_path)]},
            "test": {"patient_count": 0, "case_count": 0, "patients": [], "cases": []},
        },
    )
    luna16_registry = _write_json(
        tmp_path / "data_manifest" / "luna16_c2rv_split_registry.json",
        {"schema": "fixture_luna16_registry", "cases": [{"seriesuid": "bad", "split": "dev"}]},
    )

    try:
        build_derivative_manifest_set(
            project_root=tmp_path,
            output_root=tmp_path / "out",
            fixed_view_manifest_path=fixed_manifest,
            patient_split_manifest_path=patient_split_manifest,
            luna16_registry_path=luna16_registry,
        )
    except ValueError as exc:
        assert "unsupported LUNA16 split" in str(exc)
    else:
        raise AssertionError("unexpected LUNA16 split should fail")


def test_real_manifests_can_reject_stale_fixed_view_paths(tmp_path: Path) -> None:
    fixed_manifest = _write_json(
        tmp_path / "data_manifest" / "fixed_view_sampling_v0b.json",
        {
            "protocol": "fixture_fixed_view_sampling",
            "included_case_count": 1,
            "excluded_case_count": 0,
            "records": [_fixed_case("p1", "c0", tmp_path)],
        },
    )
    patient_split_manifest = _write_json(
        tmp_path / "data_manifest" / "splits_v0b_train_test.json",
        {
            "protocol": "fixture_patient_split",
            "train": {"patient_count": 1, "case_count": 1, "patients": ["p1"], "cases": [_case_ref("p1", "c0", tmp_path)]},
            "test": {"patient_count": 0, "case_count": 0, "patients": [], "cases": []},
        },
    )
    luna16_registry = _write_json(
        tmp_path / "data_manifest" / "luna16_c2rv_split_registry.json",
        {"schema": "fixture_luna16_registry", "cases": [{"seriesuid": "luna_train", "split": "train"}]},
    )

    try:
        build_derivative_manifest_set(
            project_root=tmp_path,
            output_root=tmp_path / "out",
            fixed_view_manifest_path=fixed_manifest,
            patient_split_manifest_path=patient_split_manifest,
            luna16_registry_path=luna16_registry,
            validate_case_paths=True,
        )
    except RuntimeError as exc:
        assert "fixed-view manifest references" in str(exc)
        assert "Regenerate fixed_view_sampling_v0b.json" in str(exc)
    else:
        raise AssertionError("stale fixed-view paths should fail when validation is enabled")
