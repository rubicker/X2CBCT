from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .split_policy import (
    CASE_MIXED_POLICY,
    PATIENT_LEVEL_POLICY,
    assign_train_test_split,
    summarize_train_test_split,
)
from .normalization import SUPPORTED_VIEW_KEYS


SCHEMA = "x2cbct_difnet_v1_derivative_manifest_set.v1"
DERIVATIVE_SCHEMA = "x2cbct_difnet_v1_derivative_manifest.v1"
TRAINING_BASE_SCHEMA = "x2cbct_difnet_v1_training_base_manifest.v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_manifest(project_root: Path, name: str) -> Path:
    return project_root / "data_manifest" / name


def _case_key(patient_id: str, case_id: str) -> str:
    return f"{patient_id}/{case_id}"


def _sort_real_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cases, key=lambda item: (str(item["patient_id"]), str(item["case_id"])))


def _projection_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_angle_deg": float(raw["target_angle_deg"]),
        "ordinal": int(raw["ordinal"]),
        "file": str(raw["file"]),
        "gantry_angle_deg": float(raw["gantry_angle_deg"]),
        "angular_residual_deg": float(raw["angular_residual_deg"]),
    }


def _real_case_from_fixed(raw: dict[str, Any]) -> dict[str, Any]:
    patient_id = str(raw["patient_id"])
    case_id = str(raw["case_id"])
    selected = {
        str(view_key): [_projection_row(item) for item in projections]
        for view_key, projections in raw.get("selected", {}).items()
        if view_key in SUPPORTED_VIEW_KEYS
    }
    return {
        "case_key": _case_key(patient_id, case_id),
        "patient_id": patient_id,
        "case_id": case_id,
        "case_path": str(raw["case_path"]),
        "projection_sets": selected,
    }


def _index_fixed_cases(fixed_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in fixed_manifest.get("records", []):
        case = _real_case_from_fixed(raw)
        indexed[case["case_key"]] = case
    return indexed


def _validate_fixed_case_paths(fixed_manifest: dict[str, Any]) -> None:
    missing: list[str] = []
    for raw in fixed_manifest.get("records", []):
        case_path = Path(str(raw["case_path"]))
        patient_id = str(raw["patient_id"])
        case_id = str(raw["case_id"])
        case_key = f"{patient_id}/{case_id}"
        required = [
            ("case_dir", case_path),
            ("CBCT.nii.gz", case_path / "CBCT.nii.gz"),
            ("Geometry", case_path / "Geometry"),
        ]
        for projections in raw.get("selected", {}).values():
            for projection in projections:
                required.append((str(projection["file"]), case_path / str(projection["file"])))
        for label, path in required:
            if not path.exists():
                missing.append(f"{case_key}: {label} -> {path}")
                break
    if missing:
        preview = "; ".join(missing[:5])
        raise RuntimeError(
            f"fixed-view manifest references {len(missing)} cases with missing local files. "
            f"First missing: {preview}. "
            "Regenerate fixed_view_sampling_v0b.json and splits_v0b_train_test.json from the current dataset root, "
            "or restore the older dataset snapshot that the manifests were built from."
        )


def _real_cases_from_split_group(group: dict[str, Any], indexed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in group.get("cases", []):
        key = _case_key(str(item["patient_id"]), str(item["case_id"]))
        if key not in indexed:
            raise KeyError(f"case {key} is missing from fixed-view manifest")
        cases.append(indexed[key])
    return _sort_real_cases(cases)


def _real_case_key_from_row(row: dict[str, Any]) -> str:
    return _case_key(str(row["patient_id"]), str(row["case_id"]))


def _real_eval_to_test_keys(split_manifest: dict[str, Any]) -> set[str]:
    group = split_manifest.get("eval_to_test", {})
    return {_real_case_key_from_row(item) for item in group.get("cases", [])}


def _apply_real_eval_to_test(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    eval_to_test_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not eval_to_test_keys:
        return train, test, []
    all_keys = {_real_case_key_from_row(item) for item in [*train, *test]}
    missing = sorted(eval_to_test_keys - all_keys)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"eval_to_test cases are missing from real split inputs: {preview}")
    moved = [item for item in train if _real_case_key_from_row(item) in eval_to_test_keys]
    kept_train = [item for item in train if _real_case_key_from_row(item) not in eval_to_test_keys]
    kept_test_keys = {_real_case_key_from_row(item) for item in test}
    merged_test = [
        *test,
        *(item for item in moved if _real_case_key_from_row(item) not in kept_test_keys),
    ]
    moved_keys = sorted(_real_case_key_from_row(item) for item in moved)
    return _sort_real_cases(kept_train), _sort_real_cases(merged_test), moved_keys


def _add_eval_to_test_metadata(split: dict[str, Any], eval_to_test_keys: set[str], moved_keys: list[str]) -> dict[str, Any]:
    if not eval_to_test_keys:
        return split
    out = dict(split)
    out["eval_to_test_case_count"] = len(eval_to_test_keys)
    out["eval_to_test_cases"] = sorted(eval_to_test_keys)
    out["moved_from_generated_train_case_count"] = len(moved_keys)
    out["moved_from_generated_train_cases"] = moved_keys
    return out


def _split_counts(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "train_case_count": len(train),
        "test_case_count": len(test),
        "train_patient_count": len({str(item.get("patient_id", item.get("case_key"))) for item in train}),
        "test_patient_count": len({str(item.get("patient_id", item.get("case_key"))) for item in test}),
    }


def _split_section(cases: list[dict[str, Any]]) -> dict[str, Any]:
    patients = sorted({str(item.get("patient_id", item.get("case_key"))) for item in cases})
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "patients": patients,
        "cases": cases,
    }


def _augmentation(enabled: bool, translation_mm: tuple[int, ...], rotation_deg: tuple[int, ...]) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "mode": "none",
            "translation_mm": [0],
            "rotation_deg": [0],
            "combination_count_per_source_case": 1,
        }
    return {
        "enabled": True,
        "mode": "train_time_reverse_mapping",
        "translation_mm": list(translation_mm),
        "rotation_deg": list(rotation_deg),
        "combination_count_per_source_case": len(translation_mm) * len(rotation_deg),
    }


def _derivative_manifest(
    *,
    name: str,
    dataset: str,
    split_policy: str,
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    augmentation: dict[str, Any],
    split: dict[str, Any],
    generated_at: str,
    command: str | None,
) -> dict[str, Any]:
    return {
        "schema": DERIVATIVE_SCHEMA,
        "generated_at": generated_at,
        "name": name,
        "dataset": dataset,
        "split_policy": split_policy,
        "command": command,
        "augmentation": augmentation,
        "split": split,
        "counts": _split_counts(train, test),
        "splits": {
            "train": _split_section(train),
            "test": _split_section(test),
        },
    }


def _training_base_manifest(
    *,
    name: str,
    dataset: str,
    cases: list[dict[str, Any]],
    derivative_names: list[str],
    generated_at: str,
    command: str | None,
) -> dict[str, Any]:
    return {
        "schema": TRAINING_BASE_SCHEMA,
        "generated_at": generated_at,
        "name": name,
        "dataset": dataset,
        "command": command,
        "rule": "union_of_all_training_splits",
        "derivative_manifests": derivative_names,
        "case_count": len(cases),
        "patient_count": len({str(item.get("patient_id", item.get("case_key"))) for item in cases}),
        "cases": cases,
    }


def _luna_case_key(row: dict[str, Any]) -> str:
    for key in ("split_case_key", "seriesuid", "patient_id", "case_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError("LUNA16 case is missing split_case_key/seriesuid/patient_id/case_id")


def _luna_case(row: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "patient_id",
        "seriesuid",
        "subset",
        "ct_path",
        "relative_ct_path",
        "raw_path",
        "relative_raw_path",
        "mask_path",
        "relative_mask_path",
        "dim_size_xyz",
        "shape_zyx",
        "spacing_xyz",
        "element_type",
        "anatomical_orientation",
        "transform_is_identity",
    )
    out = {"case_key": _luna_case_key(row)}
    for key in keep_keys:
        if key in row:
            out[key] = row[key]
    return out


def _luna_train_test_cases(registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in registry.get("cases", []):
        split = str(row.get("split", "")).strip().lower()
        case = _luna_case(row)
        if split == "train":
            train.append(case)
        elif split in {"eval", "test"}:
            test.append(case)
        else:
            raise ValueError(f"unsupported LUNA16 split: {split!r}")
    train.sort(key=lambda item: str(item["case_key"]))
    test.sort(key=lambda item: str(item["case_key"]))
    return train, test


def _unique_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for case in cases:
        indexed[str(case["case_key"])] = case
    return sorted(indexed.values(), key=lambda item: str(item["case_key"]))


def _write_named_manifest(output_root: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = output_root / f"{name}.json"
    _write_json(payload, path)
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "case_counts": payload.get("counts") or {"case_count": payload.get("case_count")},
    }


def build_derivative_manifest_set(
    *,
    project_root: Path,
    output_root: Path,
    fixed_view_manifest_path: Path | None = None,
    patient_split_manifest_path: Path | None = None,
    luna16_registry_path: Path | None = None,
    mixed_train_ratio: float = 0.8,
    mixed_seed: int = 20260517,
    translation_mm: tuple[int, ...] = (-10, 0, 10),
    rotation_deg: tuple[int, ...] = (-10, 0, 10),
    command: str | None = None,
    overwrite: bool = False,
    validate_case_paths: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"manifest output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    fixed_view_manifest_path = fixed_view_manifest_path or _default_data_manifest(project_root, "fixed_view_sampling_v0b.json")
    patient_split_manifest_path = patient_split_manifest_path or _default_data_manifest(project_root, "splits_v0b_train_test.json")
    luna16_registry_path = luna16_registry_path or _default_data_manifest(project_root, "luna16_c2rv_split_registry.json")
    fixed_manifest = _read_json(Path(fixed_view_manifest_path))
    patient_split_manifest = _read_json(Path(patient_split_manifest_path))
    luna16_registry = _read_json(Path(luna16_registry_path))
    if validate_case_paths:
        _validate_fixed_case_paths(fixed_manifest)
    generated_at = datetime.now().isoformat(timespec="seconds")

    indexed_fixed = _index_fixed_cases(fixed_manifest)
    real_eval_to_test_keys = _real_eval_to_test_keys(patient_split_manifest)
    patient_train = _real_cases_from_split_group(patient_split_manifest["train"], indexed_fixed)
    patient_test = _real_cases_from_split_group(patient_split_manifest["test"], indexed_fixed)
    patient_train, patient_test, patient_moved_keys = _apply_real_eval_to_test(
        patient_train,
        patient_test,
        real_eval_to_test_keys,
    )
    all_real_cases = _sort_real_cases(indexed_fixed.values())
    mixed = assign_train_test_split(
        all_real_cases,
        policy=CASE_MIXED_POLICY,
        train_ratio=mixed_train_ratio,
        seed=mixed_seed,
    )
    mixed_train, mixed_test, mixed_moved_keys = _apply_real_eval_to_test(
        _sort_real_cases(mixed.train),
        _sort_real_cases(mixed.test),
        real_eval_to_test_keys,
    )
    patient_train_ratio = 1.0 - float(patient_split_manifest.get("test_ratio", 0.2))
    patient_seed = int(patient_split_manifest.get("seed", mixed_seed))
    patient_split = _add_eval_to_test_metadata(
        summarize_train_test_split(
            patient_train,
            patient_test,
            policy=PATIENT_LEVEL_POLICY,
            train_ratio=patient_train_ratio,
            seed=patient_seed,
        ),
        real_eval_to_test_keys,
        patient_moved_keys,
    )
    mixed_split = _add_eval_to_test_metadata(
        summarize_train_test_split(
            mixed_train,
            mixed_test,
            policy=CASE_MIXED_POLICY,
            train_ratio=mixed_train_ratio,
            seed=mixed_seed,
        ),
        real_eval_to_test_keys,
        mixed_moved_keys,
    )
    luna_train, luna_test = _luna_train_test_cases(luna16_registry)
    luna_split = {
        "policy": "official_train_test",
        "split_unit": "case",
        "allow_patient_overlap": False,
        "patient_overlap": sorted(
            {str(item.get("patient_id", item["case_key"])) for item in luna_train}
            & {str(item.get("patient_id", item["case_key"])) for item in luna_test}
        ),
        "case_overlap_count": len({item["case_key"] for item in luna_train} & {item["case_key"] for item in luna_test}),
    }

    derivative_specs = [
        ("self_projection1208_patient_noaug", "self_projection1208", "patient", patient_train, patient_test, False, patient_split),
        ("self_projection1208_mixed_noaug", "self_projection1208", "mixed", mixed_train, mixed_test, False, mixed_split),
        ("self_projection1208_patient_aug", "self_projection1208", "patient", patient_train, patient_test, True, patient_split),
        ("self_projection1208_mixed_aug", "self_projection1208", "mixed", mixed_train, mixed_test, True, mixed_split),
        ("luna16_noaug", "luna16", "official_train_test", luna_train, luna_test, False, luna_split),
        ("luna16_aug", "luna16", "official_train_test", luna_train, luna_test, True, luna_split),
    ]
    derivative_entries = []
    for name, dataset, split_policy, train, test, aug_enabled, split in derivative_specs:
        payload = _derivative_manifest(
            name=name,
            dataset=dataset,
            split_policy=split_policy,
            train=train,
            test=test,
            augmentation=_augmentation(aug_enabled, tuple(int(item) for item in translation_mm), tuple(int(item) for item in rotation_deg)),
            split=split,
            generated_at=generated_at,
            command=command,
        )
        derivative_entries.append(_write_named_manifest(output_root, name, payload))

    real_base_name = "self_projection1208_training_base_cases"
    luna_base_name = "luna16_training_base_cases"
    real_base = _training_base_manifest(
        name=real_base_name,
        dataset="self_projection1208",
        cases=_unique_cases([*patient_train, *mixed_train]),
        derivative_names=[
            "self_projection1208_patient_noaug",
            "self_projection1208_mixed_noaug",
            "self_projection1208_patient_aug",
            "self_projection1208_mixed_aug",
        ],
        generated_at=generated_at,
        command=command,
    )
    luna_base = _training_base_manifest(
        name=luna_base_name,
        dataset="luna16",
        cases=_unique_cases(luna_train),
        derivative_names=["luna16_noaug", "luna16_aug"],
        generated_at=generated_at,
        command=command,
    )
    real_base_entry = _write_named_manifest(output_root, real_base_name, real_base)
    luna_base_entry = _write_named_manifest(output_root, luna_base_name, luna_base)

    index = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "command": command,
        "input_files": {
            "fixed_view_manifest": {"path": str(Path(fixed_view_manifest_path).resolve()), "sha256": _sha256_file(Path(fixed_view_manifest_path))},
            "patient_split_manifest": {
                "path": str(Path(patient_split_manifest_path).resolve()),
                "sha256": _sha256_file(Path(patient_split_manifest_path)),
            },
            "luna16_registry": {"path": str(Path(luna16_registry_path).resolve()), "sha256": _sha256_file(Path(luna16_registry_path))},
        },
        "rules": {
            "preprocessing_split_decision": "all_train_test_lists_are_decided_before_real_preprocessing",
            "generated_splits": ["train", "test"],
            "luna16_eval_group": "official_eval_group_merged_into_test",
            "real_eval_to_test_case_count": len(real_eval_to_test_keys),
            "test_assets": "independent_physical_cache_required_by_later_steps",
        },
        "derivative_manifests": derivative_entries,
        "training_base_manifests": {
            "self_projection1208": real_base_entry["path"],
            "luna16": luna_base_entry["path"],
        },
    }
    _write_json(index, output_root / "index.json")
    index["index_path"] = str((output_root / "index.json").resolve())
    return index
