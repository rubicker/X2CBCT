from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


DATASET_ID = "difnet_v1_si10_rot10_3x3_256"
TRANSLATION_MM = (-10, 0, 10)
ROTATION_DEG = (-10, 0, 10)
VIEW_KEYS = ("view_2", "view_5", "view_10")
VIEW_ARG = "2,5,10"
SIZE = 256
BASE_AUGMENTATION_ID = "si_0_rot_0"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str


@dataclass(frozen=True)
class PipelineLayout:
    project_root: Path
    manifest_root: Path
    cache_root: Path
    real_root: Path
    luna_root: Path
    backup_root: Path
    summary_path: Path


Validator = Callable[[], ValidationResult]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    output_root: Path
    command: tuple[str, ...]
    validator: Validator
    pre_finalize_validator: Validator | None = None
    finalize_commands: tuple[tuple[str, ...], ...] = ()
    auxiliary_paths: tuple[Path, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


def pipeline_layout(project_root: Path) -> PipelineLayout:
    root = Path(project_root).resolve()
    cache_root = root / "data_cache" / DATASET_ID
    return PipelineLayout(
        project_root=root,
        manifest_root=root / "data_manifest" / DATASET_ID,
        cache_root=cache_root,
        real_root=cache_root / "self_projection1208",
        luna_root=cache_root / "luna16",
        backup_root=root / "data_cache" / "_pipeline_backups" / DATASET_ID,
        summary_path=cache_root / "pipeline_summary.json",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result(ok: bool, reason: str) -> ValidationResult:
    return ValidationResult(ok=ok, reason=reason)


def _augmentation_matches(payload: dict, *, augmented: bool, filtered: bool = False) -> bool:
    config = payload if "enabled" in payload else payload.get("setup_augmentation") or payload.get("augmentation") or {}
    if bool(config.get("enabled")) != augmented:
        return False
    if augmented:
        if tuple(config.get("translation_mm", [])) != TRANSLATION_MM:
            return False
        if tuple(config.get("rotation_deg", [])) != ROTATION_DEG:
            return False
        if int(config.get("combination_count_per_source_case", 0)) != 9:
            return False
        if filtered and BASE_AUGMENTATION_ID not in config.get("excluded_augmentation_ids", []):
            return False
    return True


def _sample_augmentation_id(sample: dict) -> str:
    setup = sample.get("setup_augmentation")
    if isinstance(setup, dict) and setup.get("augmentation_id"):
        return str(setup["augmentation_id"])
    return str(sample.get("augmentation_id") or BASE_AUGMENTATION_ID)


def validate_manifest_set(manifest_root: Path) -> ValidationResult:
    index_path = manifest_root / "index.json"
    if not index_path.is_file():
        return _result(False, "missing index.json")
    expected = {
        "self_projection1208_patient_noaug",
        "self_projection1208_mixed_noaug",
        "self_projection1208_patient_aug",
        "self_projection1208_mixed_aug",
        "luna16_noaug",
        "luna16_aug",
    }
    try:
        index = _read_json(index_path)
        project_root = manifest_root.resolve().parents[1]
        expected_inputs = {
            "fixed_view_manifest": project_root / "data_manifest" / "fixed_view_sampling_v0b.json",
            "patient_split_manifest": project_root / "data_manifest" / "splits_v0b_train_test_eval_to_test_20260613.json",
            "luna16_registry": project_root / "data_manifest" / "luna16_c2rv_split_registry.json",
        }
        for key, expected_path in expected_inputs.items():
            actual_path = Path(index["input_files"][key]["path"]).resolve()
            if actual_path != expected_path.resolve():
                return _result(False, f"input path mismatch: {key}")
        names = {str(item["name"]) for item in index.get("derivative_manifests", [])}
        if names != expected:
            return _result(False, f"unexpected derivative manifests: {sorted(names)}")
        for name in expected:
            payload = _read_json(manifest_root / f"{name}.json")
            augmented = name.endswith("_aug")
            if not _augmentation_matches(payload.get("augmentation", {}), augmented=augmented):
                return _result(False, f"augmentation mismatch: {name}")
            if set(payload.get("splits", {})) != {"train", "test"}:
                return _result(False, f"split mismatch: {name}")
        for name in ("self_projection1208_training_base_cases", "luna16_training_base_cases"):
            payload = _read_json(manifest_root / f"{name}.json")
            if int(payload.get("case_count", 0)) <= 0:
                return _result(False, f"empty training base manifest: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result(False, f"invalid manifest set: {exc}")
    return _result(True, "manifest set complete")


def _size_matches(manifest: dict) -> bool:
    target_size = manifest.get("target_size")
    projection_size = manifest.get("projection_size")
    if target_size is None:
        target_size = (manifest.get("target_resize") or {}).get("size")
    if projection_size is None:
        projection_size = (manifest.get("projection_resize") or {}).get("size")
    return int(target_size or 0) == SIZE and int(projection_size or 0) == SIZE


def _expected_source_keys(source: dict, *, split: str, dataset: str) -> set[str]:
    rows = source.get("cases") if "case_count" in source else source["splits"][split]["cases"]
    keys = set()
    for row in rows:
        if dataset == "self_projection1208":
            patient_id = row.get("patient_id")
            case_id = row.get("case_id")
            keys.add(f"patient{patient_id}_case{case_id}")
        else:
            keys.add(str(row.get("seriesuid") or row.get("source_case_id") or row.get("patient_id") or row["case_key"]))
    return keys


def _sample_source_key(sample: dict, *, dataset: str) -> str:
    setup = sample.get("setup_augmentation") if isinstance(sample.get("setup_augmentation"), dict) else {}
    if dataset == "self_projection1208":
        return str(
            setup.get("source_case_key")
            or sample.get("source_case_key")
            or f"patient{sample.get('patient_id')}_case{sample.get('case_id')}"
        )
    return str(
        setup.get("source_case_id")
        or sample.get("source_case_id")
        or sample.get("patient_id")
        or sample.get("case_id")
    )


def _resolve_reference(cache_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (cache_root / path).resolve()


def _representative_paths(cache_root: Path, samples: list[dict]) -> list[Path]:
    selected = [samples[0], samples[len(samples) // 2], samples[-1]]
    paths = []
    for sample in selected:
        values = [
            sample.get("sample_path"),
            sample.get("case_manifest_path"),
            sample.get("cbct_payload_path"),
            sample.get("processed_cbct_path"),
        ]
        values.extend(list(sample.get("projection_payload_paths") or [])[:1])
        values.extend(list(sample.get("projection_paths") or [])[:1])
        projections = sample.get("selected_projections") or []
        if projections and isinstance(projections[0], dict):
            values.append(projections[0].get("payload_path"))
        for value in values:
            path = _resolve_reference(cache_root, value)
            if path is not None:
                paths.append(path)
    return paths


def validate_cache(
    cache_root: Path,
    *,
    dataset: str,
    split: str,
    views: Sequence[str],
    augmented: bool,
    filtered: bool,
    target_volume_mode: str,
    source_manifest: Path,
) -> ValidationResult:
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        return _result(False, "missing cache_manifest.json")
    try:
        manifest = _read_json(manifest_path)
        source = _read_json(source_manifest)
        source_count = int(
            source.get("case_count")
            if "case_count" in source
            else source["splits"][split]["case_count"]
        )
        combinations = 8 if filtered else (9 if augmented else 1)
        expected_per_view = source_count * combinations
        expected_source_keys = _expected_source_keys(source, split=split, dataset=dataset)
        samples = list(manifest.get("samples", []))
        if not samples:
            return _result(False, "cache has no samples")
        if not _size_matches(manifest):
            return _result(False, "cache size is not paired 256")
        if str(manifest.get("target_volume_mode")) != target_volume_mode:
            return _result(False, "target volume mode mismatch")
        if not _augmentation_matches(manifest, augmented=augmented, filtered=filtered):
            return _result(False, "cache augmentation mismatch")
        if manifest.get("failed_cases"):
            return _result(False, "cache contains failed cases")
        missing = (manifest.get("derived_training_cache") or {}).get("missing_source_cases", [])
        if missing:
            return _result(False, "derived cache contains missing source cases")
        actual_views = {str(sample.get("view_key")) for sample in samples}
        if actual_views != set(views):
            return _result(False, f"view mismatch: {sorted(actual_views)}")
        if {str(sample.get("split")) for sample in samples} != {split}:
            return _result(False, "sample split mismatch")
        actual_source_keys = {_sample_source_key(sample, dataset=dataset) for sample in samples}
        if actual_source_keys != expected_source_keys:
            return _result(False, "source case set does not match fixed manifest")
        for view in views:
            count = sum(1 for sample in samples if sample.get("view_key") == view)
            if count != expected_per_view:
                return _result(False, f"{view} count {count}, expected {expected_per_view}")
        augmentation_ids = {_sample_augmentation_id(sample) for sample in samples}
        if augmented and filtered and BASE_AUGMENTATION_ID in augmentation_ids:
            return _result(False, "filtered test cache still contains base augmentation")
        if augmented and not filtered and BASE_AUGMENTATION_ID not in augmentation_ids:
            return _result(False, "training cache lost base augmentation")
        missing_paths = [path for path in _representative_paths(cache_root, samples) if not path.exists()]
        if missing_paths:
            return _result(False, f"referenced asset is missing: {missing_paths[0]}")
        if dataset == "self_projection1208" and not str(manifest.get("schema", "")).startswith("x2cbct_difnet"):
            return _result(False, "unexpected real cache schema")
        if dataset == "luna16" and not str(manifest.get("schema", "")).startswith("x2cbct_synthrad2025"):
            return _result(False, "unexpected LUNA16 cache schema")
        if dataset == "luna16" and str(manifest.get("projector")) != "tigre":
            return _result(False, "LUNA16 cache was not generated with TIGRE")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result(False, f"invalid cache: {exc}")
    return _result(True, "cache complete")


def _tool_command(python: str, root: Path, tool: str, *args: str) -> tuple[str, ...]:
    return (python, str(root / "tools" / tool), *args)


def build_pipeline_stages(
    project_root: Path,
    *,
    python_executable: str | None = None,
    progress_interval: int = 25,
) -> tuple[PipelineStage, ...]:
    layout = pipeline_layout(project_root)
    root = layout.project_root
    python = python_executable or sys.executable
    manifests = layout.manifest_root
    real = layout.real_root
    luna = layout.luna_root
    split_manifest = root / "data_manifest" / "splits_v0b_train_test_eval_to_test_20260613.json"
    profile = root / "data_manifest" / "self_projection1208_projection_correction_profile.json"
    common_real = (
        "--projection-size", "256", "--target-size", "256",
        "--hu-window-mode", "fixed", "--hu-fixed-min", "-290", "--hu-fixed-max", "2060",
        "--projection-correction-mode", "self_projection1208_patient_dark_to_bright_mean",
        "--projection-correction-view", "10", "--projection-correction-profile", str(profile),
        "--progress-interval", str(progress_interval),
    )
    stages: list[PipelineStage] = []

    stages.append(
        PipelineStage(
            name="manifests",
            output_root=manifests,
            command=_tool_command(
                python, root, "build_difnet_v1_derivative_manifests.py",
                "--output-root", str(manifests),
                "--fixed-view-manifest", str(root / "data_manifest" / "fixed_view_sampling_v0b.json"),
                "--patient-split-manifest", str(split_manifest),
                "--luna16-registry", str(root / "data_manifest" / "luna16_c2rv_split_registry.json"),
                "--translation-mm=-10,0,10", "--rotation-deg=-10,0,10",
            ),
            validator=lambda: validate_manifest_set(manifests),
            metadata={"kind": "manifest_set"},
        )
    )

    real_base = real / "base_3x3_256"
    real_base_manifest = manifests / "self_projection1208_training_base_cases.json"
    stages.append(
        PipelineStage(
            name="real_base",
            output_root=real_base,
            command=_tool_command(
                python, root, "build_real_training_base_preprocess.py",
                "--output-root", str(real_base),
                "--training-base-manifest", str(real_base_manifest),
                *common_real,
                "--translation-mm=-10,0,10", "--rotation-deg=-10,0,10",
            ),
            validator=lambda: validate_cache(
                real_base,
                dataset="self_projection1208", split="train", views=("view_10",),
                augmented=True, filtered=False, target_volume_mode="base_reference",
                source_manifest=real_base_manifest,
            ),
            metadata={"kind": "training_base", "dataset": "self_projection1208"},
        )
    )

    real_derivatives = (
        ("real_train_patient_noaug", "self_projection1208_patient_noaug", "train_patient_noaug_256", False),
        ("real_train_mixed_noaug", "self_projection1208_mixed_noaug", "train_mixed_noaug_256", False),
        ("real_train_patient_aug", "self_projection1208_patient_aug", "train_patient_aug3x3_256", True),
        ("real_train_mixed_aug", "self_projection1208_mixed_aug", "train_mixed_aug3x3_256", True),
    )
    for stage_name, manifest_name, output_name, augmented in real_derivatives:
        derivative = manifests / f"{manifest_name}.json"
        output = real / output_name
        stages.append(
            PipelineStage(
                name=stage_name,
                output_root=output,
                command=_tool_command(
                    python, root, "derive_difnet_v1_training_cache.py",
                    "--base-cache-root", str(real_base),
                    "--derivative-manifest", str(derivative),
                    "--output-root", str(output),
                    "--views", VIEW_ARG,
                ),
                validator=lambda output=output, derivative=derivative, augmented=augmented: validate_cache(
                    output,
                    dataset="self_projection1208", split="train", views=VIEW_KEYS,
                    augmented=augmented, filtered=False, target_volume_mode="base_reference",
                    source_manifest=derivative,
                ),
                metadata={"kind": "training_derivative", "dataset": "self_projection1208"},
            )
        )

    real_tests = (
        ("real_test_patient_noaug", "self_projection1208_patient_noaug", "test_patient_noaug_256", False),
        ("real_test_mixed_noaug", "self_projection1208_mixed_noaug", "test_mixed_noaug_256", False),
        ("real_test_patient_aug", "self_projection1208_patient_aug", "test_patient_aug3x3_256", True),
        ("real_test_mixed_aug", "self_projection1208_mixed_aug", "test_mixed_aug3x3_256", True),
    )
    for stage_name, manifest_name, output_name, augmented in real_tests:
        derivative = manifests / f"{manifest_name}.json"
        output = real / output_name
        finalize = ()
        if augmented:
            finalize = (
                _tool_command(
                    python, root, "remove_cache_augmentation_id.py",
                    "--cache-root", str(output),
                    "--augmentation-id", BASE_AUGMENTATION_ID,
                    "--backup-parent", str(layout.backup_root / "filtered_samples" / stage_name),
                ),
            )
        stages.append(
            PipelineStage(
                name=stage_name,
                output_root=output,
                command=_tool_command(
                    python, root, "build_real_independent_test_cache.py",
                    "--derivative-manifest", str(derivative),
                    "--output-root", str(output),
                    "--views", VIEW_ARG,
                    *common_real,
                ),
                validator=lambda output=output, derivative=derivative, augmented=augmented: validate_cache(
                    output,
                    dataset="self_projection1208", split="test", views=VIEW_KEYS,
                    augmented=augmented, filtered=augmented, target_volume_mode="full",
                    source_manifest=derivative,
                ),
                pre_finalize_validator=(
                    (lambda output=output, derivative=derivative: validate_cache(
                        output,
                        dataset="self_projection1208", split="test", views=VIEW_KEYS,
                        augmented=True, filtered=False, target_volume_mode="full",
                        source_manifest=derivative,
                    ))
                    if augmented else None
                ),
                finalize_commands=finalize,
                metadata={"kind": "independent_test", "dataset": "self_projection1208"},
            )
        )

    luna_base_outer = luna / "base_3x3_256"
    luna_base_cache = luna_base_outer / "luna16"
    luna_base_manifest = manifests / "luna16_training_base_cases.json"
    luna_base_work = luna_base_outer.parent / f"_{luna_base_outer.name}_build_tmp"
    stages.append(
        PipelineStage(
            name="luna16_base",
            output_root=luna_base_outer,
            command=_tool_command(
                python, root, "build_luna16_training_base_preprocess.py",
                "--output-root", str(luna_base_outer),
                "--training-base-manifest", str(luna_base_manifest),
                "--root", str(root / "dataset_list" / "LUNA16"),
                "--size-preset", "256", "--projector", "tigre",
                "--translation-mm=-10,0,10", "--rotation-deg=-10,0,10",
                "--progress-interval", str(progress_interval),
            ),
            validator=lambda: validate_cache(
                luna_base_cache,
                dataset="luna16", split="train", views=("view_10",),
                augmented=True, filtered=False, target_volume_mode="base_reference",
                source_manifest=luna_base_manifest,
            ),
            auxiliary_paths=(luna_base_work,),
            metadata={"kind": "training_base", "dataset": "luna16"},
        )
    )

    luna_derivatives = (
        ("luna16_train_noaug", "luna16_noaug", "train_noaug_256", False),
        ("luna16_train_aug", "luna16_aug", "train_aug3x3_256", True),
    )
    for stage_name, manifest_name, output_name, augmented in luna_derivatives:
        derivative = manifests / f"{manifest_name}.json"
        output = luna / output_name
        stages.append(
            PipelineStage(
                name=stage_name,
                output_root=output,
                command=_tool_command(
                    python, root, "derive_difnet_v1_training_cache.py",
                    "--base-cache-root", str(luna_base_cache),
                    "--derivative-manifest", str(derivative),
                    "--output-root", str(output),
                    "--views", VIEW_ARG,
                ),
                validator=lambda output=output, derivative=derivative, augmented=augmented: validate_cache(
                    output,
                    dataset="luna16", split="train", views=VIEW_KEYS,
                    augmented=augmented, filtered=False, target_volume_mode="base_reference",
                    source_manifest=derivative,
                ),
                metadata={"kind": "training_derivative", "dataset": "luna16"},
            )
        )

    luna_tests = (
        ("luna16_test_noaug", "luna16_noaug", "test_noaug_256", False),
        ("luna16_test_aug", "luna16_aug", "test_aug3x3_256", True),
    )
    for stage_name, manifest_name, output_name, augmented in luna_tests:
        derivative = manifests / f"{manifest_name}.json"
        output_outer = luna / output_name
        output_cache = output_outer / "luna16"
        work_root = output_outer.parent / f"_{output_outer.name}_build_tmp"
        finalize = ()
        if augmented:
            finalize = (
                _tool_command(
                    python, root, "remove_cache_augmentation_id.py",
                    "--cache-root", str(output_cache),
                    "--augmentation-id", BASE_AUGMENTATION_ID,
                    "--backup-parent", str(layout.backup_root / "filtered_samples" / stage_name),
                ),
            )
        stages.append(
            PipelineStage(
                name=stage_name,
                output_root=output_outer,
                command=_tool_command(
                    python, root, "build_luna16_independent_test_cache.py",
                    "--derivative-manifest", str(derivative),
                    "--output-root", str(output_outer),
                    "--root", str(root / "dataset_list" / "LUNA16"),
                    "--size-preset", "256", "--projector", "tigre",
                    "--progress-interval", str(progress_interval),
                ),
                validator=lambda output_cache=output_cache, derivative=derivative, augmented=augmented: validate_cache(
                    output_cache,
                    dataset="luna16", split="test", views=VIEW_KEYS,
                    augmented=augmented, filtered=augmented, target_volume_mode="full",
                    source_manifest=derivative,
                ),
                pre_finalize_validator=(
                    (lambda output_cache=output_cache, derivative=derivative: validate_cache(
                        output_cache,
                        dataset="luna16", split="test", views=VIEW_KEYS,
                        augmented=True, filtered=False, target_volume_mode="full",
                        source_manifest=derivative,
                    ))
                    if augmented else None
                ),
                finalize_commands=finalize,
                auxiliary_paths=(work_root,),
                metadata={"kind": "independent_test", "dataset": "luna16"},
            )
        )
    return tuple(stages)


def required_pipeline_inputs(project_root: Path) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    return (
        root / "dataset_list" / "self_projection1208",
        root / "dataset_list" / "LUNA16",
        root / "data_manifest" / "fixed_view_sampling_v0b.json",
        root / "data_manifest" / "splits_v0b_train_test_eval_to_test_20260613.json",
        root / "data_manifest" / "luna16_c2rv_split_registry.json",
        root / "data_manifest" / "self_projection1208_projection_correction_profile.json",
        root / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json",
    )


def validate_pipeline_inputs(project_root: Path) -> None:
    missing = [path for path in required_pipeline_inputs(project_root) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required pipeline inputs:\n{formatted}")


def _backup_existing(path: Path, *, layout: PipelineLayout, stage_name: str, stamp: str) -> Path | None:
    if not path.exists():
        return None
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(layout.project_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to move outside project: {resolved}") from exc
    destination = layout.backup_root / stamp / stage_name / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"backup destination already exists: {destination}")
    shutil.move(str(resolved), str(destination))
    return destination


def _write_summary(layout: PipelineLayout, payload: dict) -> None:
    layout.summary_path.parent.mkdir(parents=True, exist_ok=True)
    layout.summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def describe_pipeline(stages: Sequence[PipelineStage], layout: PipelineLayout) -> dict:
    return {
        "schema": "x2cbct_difnet_preprocess_plan.v1",
        "dataset_id": DATASET_ID,
        "manifest_root": str(layout.manifest_root),
        "cache_root": str(layout.cache_root),
        "translation_mm": list(TRANSLATION_MM),
        "rotation_deg": list(ROTATION_DEG),
        "views": list(VIEW_KEYS),
        "size": SIZE,
        "stages": [
            {
                "name": stage.name,
                "output_root": str(stage.output_root),
                "command": list(stage.command),
                "finalize_commands": [list(command) for command in stage.finalize_commands],
                **stage.metadata,
            }
            for stage in stages
        ],
    }


def run_pipeline(
    stages: Sequence[PipelineStage],
    layout: PipelineLayout,
    *,
    force_rebuild: bool = False,
    runner: Callable[..., object] = subprocess.run,
) -> dict:
    validate_pipeline_inputs(layout.project_root)
    started_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "schema": "x2cbct_difnet_preprocess_run.v1",
        "dataset_id": DATASET_ID,
        "started_at": started_at,
        "force_rebuild": bool(force_rebuild),
        "stages": [],
    }
    for stage in stages:
        event = {"name": stage.name, "output_root": str(stage.output_root)}
        current = stage.validator()
        if current.ok and not force_rebuild:
            event.update({"status": "skipped_complete", "validation": current.reason})
            summary["stages"].append(event)
            _write_summary(layout, summary)
            print(json.dumps({"event": "stage_skip", **event}, ensure_ascii=False), flush=True)
            continue
        try:
            if (
                not force_rebuild
                and stage.pre_finalize_validator is not None
                and stage.pre_finalize_validator().ok
            ):
                for command in stage.finalize_commands:
                    runner(list(command), cwd=layout.project_root, check=True)
                final = stage.validator()
                if not final.ok:
                    raise RuntimeError(f"stage finalization failed validation: {stage.name}: {final.reason}")
                event.update({"status": "finalized_existing", "validation": final.reason})
            else:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backups = []
                for path in (stage.output_root, *stage.auxiliary_paths):
                    backup = _backup_existing(path, layout=layout, stage_name=stage.name, stamp=stamp)
                    if backup is not None:
                        backups.append(str(backup))
                print(json.dumps({"event": "stage_start", "stage": stage.name}, ensure_ascii=False), flush=True)
                runner(list(stage.command), cwd=layout.project_root, check=True)
                for command in stage.finalize_commands:
                    runner(list(command), cwd=layout.project_root, check=True)
                final = stage.validator()
                if not final.ok:
                    raise RuntimeError(f"stage failed validation: {stage.name}: {final.reason}")
                event.update({"status": "rebuilt" if backups else "built", "backups": backups, "validation": final.reason})
            summary["stages"].append(event)
            _write_summary(layout, summary)
            print(json.dumps({"event": "stage_done", **event}, ensure_ascii=False), flush=True)
        except Exception as exc:
            event.update({"status": "failed", "error": str(exc)})
            summary["stages"].append(event)
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary["ok"] = False
            _write_summary(layout, summary)
            raise
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["ok"] = True
    _write_summary(layout, summary)
    return summary
