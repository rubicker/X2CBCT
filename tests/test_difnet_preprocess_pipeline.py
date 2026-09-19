from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import x2cbct_difnet.dataset_bundle as dataset_bundle
from x2cbct_difnet.preprocess_pipeline import (
    BASE_AUGMENTATION_ID,
    DATASET_ID,
    PipelineStage,
    ValidationResult,
    build_pipeline_stages,
    describe_pipeline,
    pipeline_layout,
    run_pipeline,
    validate_cache,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _create_required_inputs(root: Path) -> None:
    (root / "dataset_list" / "self_projection1208").mkdir(parents=True)
    (root / "dataset_list" / "LUNA16").mkdir(parents=True)
    for name in (
        "fixed_view_sampling_v0b.json",
        "splits_v0b_train_test_eval_to_test_20260613.json",
        "luna16_c2rv_split_registry.json",
        "self_projection1208_projection_correction_profile.json",
    ):
        _write_json(root / "data_manifest" / name, {})
    _write_json(root / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json", {"status": "pass"})


def test_fixed_plan_contains_complete_3x3_matrix(tmp_path: Path) -> None:
    stages = build_pipeline_stages(tmp_path, python_executable="python")
    layout = pipeline_layout(tmp_path)
    plan = describe_pipeline(stages, layout)
    names = [stage["name"] for stage in plan["stages"]]

    assert plan["dataset_id"] == DATASET_ID
    assert plan["translation_mm"] == [-10, 0, 10]
    assert plan["rotation_deg"] == [-10, 0, 10]
    assert plan["views"] == ["view_2", "view_5", "view_10"]
    assert plan["size"] == 256
    assert len(stages) == 15
    assert names == [
        "manifests",
        "real_base",
        "real_train_patient_noaug",
        "real_train_mixed_noaug",
        "real_train_patient_aug",
        "real_train_mixed_aug",
        "real_test_patient_noaug",
        "real_test_mixed_noaug",
        "real_test_patient_aug",
        "real_test_mixed_aug",
        "luna16_base",
        "luna16_train_noaug",
        "luna16_train_aug",
        "luna16_test_noaug",
        "luna16_test_aug",
    ]
    command_text = "\n".join(" ".join(stage.command) for stage in stages)
    assert "--translation-mm=-10,0,10" in command_text
    assert "--rotation-deg=-10,0,10" in command_text
    finalized = {stage.name for stage in stages if stage.finalize_commands}
    assert finalized == {"real_test_patient_aug", "real_test_mixed_aug", "luna16_test_aug"}


def test_pipeline_skips_complete_and_backs_up_for_force_rebuild(tmp_path: Path) -> None:
    _create_required_inputs(tmp_path)
    layout = pipeline_layout(tmp_path)
    output = tmp_path / "data_cache" / "custom_stage"
    marker = output / "complete.json"
    calls = []

    def validator() -> ValidationResult:
        return ValidationResult(marker.is_file(), "complete" if marker.is_file() else "missing")

    def runner(command: list[str], **_: object) -> None:
        calls.append(command)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")

    stage = PipelineStage(
        name="fixture",
        output_root=output,
        command=("fixture-build",),
        validator=validator,
    )
    first = run_pipeline((stage,), layout, runner=runner)
    second = run_pipeline((stage,), layout, runner=runner)
    third = run_pipeline((stage,), layout, force_rebuild=True, runner=runner)

    assert first["stages"][0]["status"] == "built"
    assert second["stages"][0]["status"] == "skipped_complete"
    assert third["stages"][0]["status"] == "rebuilt"
    assert len(calls) == 2
    backups = list(layout.backup_root.rglob("complete.json"))
    assert len(backups) == 1


def test_augmented_test_filter_removes_base_and_refreshes_counts(tmp_path: Path) -> None:
    cache_root = tmp_path / "test_aug"
    derivative = _write_json(
        tmp_path / "manifest.json",
        {
            "splits": {
                "test": {
                    "case_count": 1,
                    "cases": [{"patient_id": "1", "case_id": "1", "case_key": "1/1"}],
                }
            },
            "augmentation": {
                "enabled": True,
                "translation_mm": [-10, 0, 10],
                "rotation_deg": [-10, 0, 10],
                "combination_count_per_source_case": 9,
            },
        },
    )
    samples = []
    for view in ("view_2", "view_5", "view_10"):
        for shift in (-10, 0, 10):
            for rotation in (-10, 0, 10):
                shift_label = f"m{abs(shift)}" if shift < 0 else ("0" if shift == 0 else f"p{shift}")
                rotation_label = f"m{abs(rotation)}" if rotation < 0 else ("0" if rotation == 0 else f"p{rotation}")
                augmentation_id = f"si_{shift_label}_rot_{rotation_label}"
                samples.append(
                    {
                        "sample_id": f"{view}_{augmentation_id}",
                        "case_id": f"case_{augmentation_id}",
                        "patient_id": "1",
                        "split": "test",
                        "view_key": view,
                        "setup_augmentation": {
                            "augmentation_id": augmentation_id,
                            "source_case_key": "patient1_case1",
                        },
                    }
                )
    _write_json(
        cache_root / "cache_manifest.json",
        {
            "schema": "x2cbct_difnet_v0b_cache.v2",
            "projection_resize": {"size": 256},
            "target_resize": {"size": 256},
            "target_volume_mode": "full",
            "setup_augmentation": {
                "enabled": True,
                "translation_mm": [-10, 0, 10],
                "rotation_deg": [-10, 0, 10],
                "combination_count_per_source_case": 9,
            },
            "view_sample_counts": {view: {"train": 0, "test": 9} for view in ("view_2", "view_5", "view_10")},
            "samples": samples,
            "failed_cases": [],
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "remove_cache_augmentation_id.py"),
            "--cache-root",
            str(cache_root),
            "--augmentation-id",
            BASE_AUGMENTATION_ID,
            "--backup-parent",
            str(tmp_path / "backups"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    validation = validate_cache(
        cache_root,
        dataset="self_projection1208",
        split="test",
        views=("view_2", "view_5", "view_10"),
        augmented=True,
        filtered=True,
        target_volume_mode="full",
        source_manifest=derivative,
    )
    manifest = json.loads((cache_root / "cache_manifest.json").read_text(encoding="utf-8"))

    assert result["removed_sample_count"] == 3
    assert validation.ok, validation.reason
    assert all(counts["test"] == 8 for counts in manifest["view_sample_counts"].values())
    assert len(manifest["samples"]) == 24
    assert not (cache_root / "filter_backups").exists()


def test_incremental_bundle_scope_and_sidecars(tmp_path: Path, monkeypatch) -> None:
    manifest_root, cache_root = dataset_bundle.bundle_sources(tmp_path)
    _write_json(manifest_root / "index.json", {"ok": True})
    _write_json(cache_root / "pipeline_summary.json", {"ok": True})
    monkeypatch.setattr(dataset_bundle, "validate_bundle_sources", lambda _: None)
    calls = []

    def runner(command: list[str], **_: object) -> None:
        calls.append(command)
        Path(command[2]).write_bytes(b"fixture archive")

    result = dataset_bundle.create_dataset_bundle(
        tmp_path,
        tmp_path / "packages",
        runner=runner,
    )
    package_root = Path(result["package_root"])

    assert calls
    assert f"data_manifest/{DATASET_ID}" in calls[0]
    assert f"data_cache/{DATASET_ID}" in calls[0]
    assert (package_root / "bundle_manifest.json").is_file()
    assert (package_root / "files.json").is_file()
    assert (package_root / "SHA256SUMS.txt").is_file()
