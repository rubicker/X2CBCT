from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cache import read_json, relative_to_cache, resolve_cache_path, write_json
from .normalization import VIEW_TARGETS
from .setup_augmentation import bin_label
from .synthrad2025 import SYNTHRAD_VIEW_COUNTS


REAL_SCHEMA = "x2cbct_difnet_v0b_cache.v2"
SYNTHRAD_SCHEMA = "x2cbct_synthrad2025_task2_view_cache.v1"
BASE_AUGMENTATION_ID = "si_0_rot_0"
VIEW_SUBSET_INDICES = {
    "view_10": list(range(10)),
    "view_8": [0, 1, 2, 4, 5, 6, 8, 9],
    "view_5": [0, 2, 4, 6, 8],
    "view_2": [0, 6],
}


def _parse_views(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    views: list[str] = []
    for item in raw:
        token = str(item).strip()
        if not token:
            continue
        if token in {"2", "5", "8", "10"}:
            token = f"view_{token}"
        if token not in VIEW_SUBSET_INDICES:
            raise ValueError(f"unsupported view: {token}")
        views.append(token)
    return tuple(dict.fromkeys(views))


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _absolute_path(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    return str(resolve_cache_path(root, value).resolve())


def _case_key_to_real_source(case_key: str) -> str:
    text = str(case_key)
    if text.startswith("patient") and "_case" in text:
        return text
    patient_id, sep, case_id = text.partition("/")
    if not sep:
        raise ValueError(f"unsupported self_projection1208 case_key: {case_key}")
    return f"patient{patient_id}_case{case_id}"


def _real_source_key(sample: dict[str, Any]) -> str:
    setup = sample.get("setup_augmentation") if isinstance(sample.get("setup_augmentation"), dict) else {}
    if setup.get("source_case_key"):
        return str(setup["source_case_key"])
    if sample.get("case_key"):
        return str(sample["case_key"])
    return f"patient{sample['patient_id']}_case{sample['case_id']}"


def _real_augmentation_id(sample: dict[str, Any]) -> str:
    setup = sample.get("setup_augmentation") if isinstance(sample.get("setup_augmentation"), dict) else {}
    return str(setup.get("augmentation_id") or sample.get("augmentation_id") or BASE_AUGMENTATION_ID)


def _luna_source_key(sample: dict[str, Any]) -> str:
    setup = sample.get("setup_augmentation") if isinstance(sample.get("setup_augmentation"), dict) else {}
    return str(setup.get("source_case_id") or sample.get("source_case_id") or sample["case_id"])


def _luna_case_key_from_derivative(item: dict[str, Any]) -> str:
    for key in ("seriesuid", "source_case_id", "patient_id"):
        if item.get(key):
            return str(item[key])
    case_key = str(item["case_key"])
    subset = str(item.get("subset") or "")
    prefix = f"{subset}_"
    if subset and case_key.startswith(prefix):
        return case_key[len(prefix) :]
    if case_key.startswith("subset") and "_" in case_key:
        maybe_subset, rest = case_key.split("_", 1)
        if maybe_subset.removeprefix("subset").isdigit():
            return rest
    return case_key


def _allowed_augmentation_ids(derivative_manifest: dict[str, Any]) -> set[str]:
    augmentation = derivative_manifest.get("augmentation", {})
    if not augmentation.get("enabled"):
        return {BASE_AUGMENTATION_ID}
    ids = set()
    for shift in augmentation.get("translation_mm", [-10, 0, 10]):
        for angle in augmentation.get("rotation_deg", [-10, 0, 10]):
            ids.add(f"si_{bin_label(int(shift))}_rot_{bin_label(int(angle))}")
    return ids


def _source_case_keys(derivative_manifest: dict[str, Any]) -> set[str]:
    dataset = str(derivative_manifest["dataset"])
    cases = derivative_manifest["splits"]["train"]["cases"]
    if dataset == "self_projection1208":
        return {_case_key_to_real_source(str(item["case_key"])) for item in cases}
    if dataset == "luna16":
        return {_luna_case_key_from_derivative(item) for item in cases}
    raise ValueError(f"unsupported derivative dataset: {dataset}")


def _subset_list(values: list[Any], view_key: str) -> list[Any]:
    indices = VIEW_SUBSET_INDICES[view_key]
    if len(values) < 10:
        raise ValueError(f"view_10 source has only {len(values)} items")
    return [values[index] for index in indices]


def _subset_array(values: np.ndarray, view_key: str) -> np.ndarray:
    indices = VIEW_SUBSET_INDICES[view_key]
    if len(values) < 10:
        raise ValueError(f"view_10 source has only {len(values)} items")
    return values[indices]


def _real_sample_id(source_key: str, augmentation_id: str, view_key: str, *, augmented: bool) -> str:
    if augmented:
        return f"{source_key}_{augmentation_id}_{view_key}"
    return f"{source_key}_{view_key}"


def _case_key_from_sample_id(sample_id: str, view_key: str) -> str:
    suffix = f"_{view_key}"
    if not str(sample_id).endswith(suffix):
        raise ValueError(f"sample_id {sample_id} does not end with {suffix}")
    return str(sample_id)[: -len(suffix)]


def _real_case_manifest_path(source_root: Path, value: str, sample_id: str, view_key: str) -> str:
    current = Path(_absolute_path(source_root, value))
    if current.exists():
        return str(current)
    case_key = _case_key_from_sample_id(sample_id, view_key)
    repaired = (source_root / "cases" / case_key / "case_manifest.json").resolve()
    return str(repaired)


def _copy_real_index_npz(
    *,
    source_npz_path: Path,
    output_npz_path: Path,
    source_root: Path,
    source_key: str,
    view_key: str,
    sample_id: str,
    augmented: bool,
) -> None:
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_npz_path, allow_pickle=False) as raw:
        payload = {key: raw[key] for key in raw.files}
        payload["sample_id"] = np.array(sample_id)
        payload["split"] = np.array("train")
        payload["view_key"] = np.array(view_key)
        payload["case_key"] = np.array(source_key)
        payload["case_manifest_path"] = np.array(
            _real_case_manifest_path(source_root, str(raw["case_manifest_path"].item()), str(raw["sample_id"].item()), "view_10")
        )
        payload["cbct_payload_path"] = np.array(_absolute_path(source_root, str(raw["cbct_payload_path"].item())))
        payload["projection_payload_paths"] = np.array(
            [_absolute_path(source_root, str(item)) for item in _subset_list([str(item) for item in raw["projection_payload_paths"].tolist()], view_key)]
        )
        for key in ("projection_files", "target_angles_deg", "gantry_angles_deg", "angular_residuals_deg"):
            if key in payload:
                payload[key] = _subset_array(payload[key], view_key)
        if not augmented:
            for key in (
                "augmentation_id",
                "translation_mm",
                "rotation_deg",
                "source_patient_id",
                "source_case_id",
                "source_case_key",
                "volume_fill_value",
                "target_sampling_fill_value",
            ):
                payload.pop(key, None)
        np.savez_compressed(output_npz_path, **payload)


def _real_manifest_sample(
    source_sample: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    output_npz_path: Path,
    source_key: str,
    view_key: str,
    sample_id: str,
    augmented: bool,
) -> dict[str, Any]:
    sample = dict(source_sample)
    sample["sample_id"] = sample_id
    sample["split"] = "train"
    sample["view_key"] = view_key
    sample["case_key"] = source_key
    sample["sample_path"] = relative_to_cache(output_npz_path, output_root)
    sample["case_manifest_path"] = _real_case_manifest_path(
        source_root,
        str(source_sample["case_manifest_path"]),
        str(source_sample["sample_id"]),
        str(source_sample["view_key"]),
    )
    sample["cbct_payload_path"] = _absolute_path(source_root, str(source_sample["cbct_payload_path"]))
    sample["projection_payload_paths"] = [
        _absolute_path(source_root, str(item))
        for item in _subset_list([str(item) for item in source_sample["projection_payload_paths"]], view_key)
    ]
    selected = []
    for item in _subset_list(list(source_sample.get("selected_projections", [])), view_key):
        copied = dict(item)
        for key in ("payload_path", "metadata_path"):
            if copied.get(key):
                copied[key] = _absolute_path(source_root, str(copied[key]))
        selected.append(copied)
    sample["selected_projections"] = selected
    if selected:
        sample["projection_geometry"] = {
            **dict(sample.get("projection_geometry", {})),
            "gantry_angles_deg": [float(item["gantry_angle_deg"]) for item in selected],
            "selected_projection_files": [str(item["file"]) for item in selected],
        }
    if not augmented:
        sample.pop("setup_augmentation", None)
    return sample


def _split_summary(samples: list[dict[str, Any]], *, policy: str, dataset: str) -> dict[str, Any]:
    source_keys = sorted(
        {
            _real_source_key(sample) if dataset == "self_projection1208" else _luna_source_key(sample)
            for sample in samples
        }
    )
    patients = sorted({str(sample.get("patient_id", sample.get("case_id"))) for sample in samples})
    return {
        "policy": policy,
        "split_unit": "precomputed_training_case",
        "train_patient_count": len(patients),
        "test_patient_count": 0,
        "train_case_count": len(source_keys),
        "test_case_count": 0,
        "patient_overlap": [],
        "case_overlap_count": 0,
        "case_overlap": [],
        "allow_patient_overlap": policy == "mixed",
        "train_cases": len(source_keys),
        "test_cases": 0,
        "indexed_train_cases": len(source_keys),
        "indexed_test_cases": 0,
        "has_validation_split": False,
        "test_split_role": "separate_independent_test_cache",
    }


def _view_counts(samples: list[dict[str, Any]], views: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {view: {"train": sum(1 for item in samples if item["view_key"] == view), "test": 0} for view in views}


def _missing_source_cases_error(missing: list[str], base_manifest: dict[str, Any]) -> str:
    preview = ", ".join(missing[:5])
    message = f"{len(missing)} source cases are missing from base cache: {preview}"
    excluded = base_manifest.get("local_dataset_filter", {}).get("excluded_cases", [])
    excluded_by_key = {
        f"patient{item.get('patient_id')}_case{item.get('case_id')}": item
        for item in excluded
        if item.get("patient_id") is not None and item.get("case_id") is not None
    }
    matched = [key for key in missing if key in excluded_by_key]
    if matched:
        first = excluded_by_key[matched[0]]
        missing_kinds = ", ".join(str(item.get("kind")) for item in first.get("missing", [])[:4])
        message += (
            f". {len(matched)} missing cases were excluded when the base cache was built because required local files "
            f"were absent. First excluded case: {matched[0]} missing {missing_kinds}. "
            "Restore the missing dataset files and rebuild the training-base cache, or regenerate the fixed manifests "
            "for the locally available subset."
        )
    return message


def _real_views(views: tuple[str, ...]) -> dict[str, list[float]]:
    return {view: [float(item) for item in VIEW_TARGETS[view]] for view in views}


def _derive_real_training_cache(
    *,
    base_root: Path,
    base_manifest: dict[str, Any],
    derivative_manifest: dict[str, Any],
    output_root: Path,
    views: tuple[str, ...],
    command: str,
) -> dict[str, Any]:
    source_cases = _source_case_keys(derivative_manifest)
    allowed_augmentations = _allowed_augmentation_ids(derivative_manifest)
    augmented = bool(derivative_manifest.get("augmentation", {}).get("enabled"))
    source_samples = [
        item
        for item in base_manifest["samples"]
        if item["view_key"] == "view_10"
        and item["split"] == "train"
        and _real_source_key(item) in source_cases
        and _real_augmentation_id(item) in allowed_augmentations
    ]
    derived_samples: list[dict[str, Any]] = []
    for source_sample in sorted(source_samples, key=lambda item: (_real_source_key(item), _real_augmentation_id(item))):
        source_key = _real_source_key(source_sample)
        augmentation_id = _real_augmentation_id(source_sample)
        source_npz = resolve_cache_path(base_root, str(source_sample["sample_path"]))
        for view_key in views:
            sample_id = _real_sample_id(source_key, augmentation_id, view_key, augmented=augmented)
            output_npz = output_root / view_key / "train" / f"{sample_id}.npz"
            _copy_real_index_npz(
                source_npz_path=source_npz,
                output_npz_path=output_npz,
                source_root=base_root,
                source_key=source_key,
                view_key=view_key,
                sample_id=sample_id,
                augmented=augmented,
            )
            derived_samples.append(
                _real_manifest_sample(
                    source_sample,
                    source_root=base_root,
                    output_root=output_root,
                    output_npz_path=output_npz,
                    source_key=source_key,
                    view_key=view_key,
                    sample_id=sample_id,
                    augmented=augmented,
                )
            )
    missing_cases = sorted(source_cases - {_real_source_key(item) for item in source_samples})
    manifest = dict(base_manifest)
    manifest.update(
        {
            "cache_version": output_root.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_command": command,
            "views": _real_views(views),
            "setup_augmentation": derivative_manifest.get("augmentation", {}),
            "split_manifest": {
                "path": str(Path(derivative_manifest.get("manifest_path", "") or "").resolve())
                if derivative_manifest.get("manifest_path")
                else None,
                "derivative_name": derivative_manifest["name"],
            },
            "split": _split_summary(derived_samples, policy=str(derivative_manifest["split_policy"]), dataset="self_projection1208"),
            "target_volume_mode": base_manifest.get("target_volume_mode", "base_reference"),
            "derived_training_cache": {
                "source_cache_root": str(base_root),
                "source_cache_manifest": str((base_root / "cache_manifest.json").resolve()),
                "derivative_manifest": derivative_manifest["name"],
                "asset_reference_mode": "absolute_paths",
                "writes_body_or_projection_assets": False,
                "missing_source_cases": missing_cases,
                "view_subset_indices": {key: VIEW_SUBSET_INDICES[key] for key in views},
            },
            "view_sample_counts": _view_counts(derived_samples, views),
            "samples": derived_samples,
            "failed_cases": [],
        }
    )
    return manifest


def _luna_manifest_sample(source_sample: dict[str, Any], *, base_root: Path, view_key: str) -> dict[str, Any]:
    sample = dict(source_sample)
    indices = VIEW_SUBSET_INDICES[view_key]
    projection_paths = [str(item) for item in source_sample["projection_paths"]]
    angles = [float(item) for item in source_sample["angles_deg"]]
    sample["split"] = "train"
    sample["view_key"] = view_key
    sample["view_count"] = SYNTHRAD_VIEW_COUNTS[view_key]
    sample["view_indices"] = indices
    sample["angles_deg"] = _subset_list(angles, view_key)
    sample["projection_paths"] = [_absolute_path(base_root, item) for item in _subset_list(projection_paths, view_key)]
    sample["processed_cbct_path"] = _absolute_path(base_root, str(source_sample["processed_cbct_path"]))
    if source_sample.get("processed_mask_path"):
        sample["processed_mask_path"] = _absolute_path(base_root, str(source_sample["processed_mask_path"]))
    sample["sample_id"] = f"{source_sample['source']}_{source_sample.get('dataset_group', 'luna16')}_{source_sample['case_id']}_{view_key}"
    return sample


def _derive_luna_training_cache(
    *,
    base_root: Path,
    base_manifest: dict[str, Any],
    derivative_manifest: dict[str, Any],
    views: tuple[str, ...],
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    source_cases = _source_case_keys(derivative_manifest)
    allowed_augmentations = _allowed_augmentation_ids(derivative_manifest)
    source_samples = [
        item
        for item in base_manifest["samples"]
        if item["view_key"] == "view_10"
        and item["split"] == "train"
        and _luna_source_key(item) in source_cases
        and str(item.get("augmentation_id") or BASE_AUGMENTATION_ID) in allowed_augmentations
    ]
    derived_samples = [
        _luna_manifest_sample(source_sample, base_root=base_root, view_key=view_key)
        for source_sample in sorted(source_samples, key=lambda item: (_luna_source_key(item), str(item.get("augmentation_id", ""))))
        for view_key in views
    ]
    missing_cases = sorted(source_cases - {_luna_source_key(item) for item in source_samples})
    manifest = dict(base_manifest)
    manifest.update(
        {
            "cache_version": output_root.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_command": command,
            "views": list(views),
            "view_counts": {view: SYNTHRAD_VIEW_COUNTS[view] for view in views},
            "setup_augmentation": derivative_manifest.get("augmentation", {}),
            "split_policy": str(derivative_manifest["split_policy"]),
            "test_is_validation": False,
            "target_volume_mode": base_manifest.get("target_volume_mode", "base_reference"),
            "derived_training_cache": {
                "source_cache_root": str(base_root),
                "source_cache_manifest": str((base_root / "cache_manifest.json").resolve()),
                "derivative_manifest": derivative_manifest["name"],
                "asset_reference_mode": "absolute_paths",
                "writes_body_or_projection_assets": False,
                "missing_source_cases": missing_cases,
                "view_subset_indices": {key: VIEW_SUBSET_INDICES[key] for key in views},
            },
            "counts": {
                "cases": len({_luna_source_key(item) for item in source_samples}),
                "split": {"train": len(source_samples), "test": 0},
                "samples": len(derived_samples),
            },
            "split": _split_summary(derived_samples, policy=str(derivative_manifest["split_policy"]), dataset="luna16"),
            "samples": derived_samples,
        }
    )
    return manifest


def derive_v1_training_cache(
    *,
    base_cache_root: Path,
    derivative_manifest_path: Path,
    output_root: Path,
    views: str | Iterable[str] = ("view_2", "view_5", "view_8", "view_10"),
    overwrite: bool = False,
    command: str = "",
) -> dict[str, Any]:
    base_root = Path(base_cache_root).resolve()
    out_root = Path(output_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root already exists: {out_root}")
        _safe_rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_views = _parse_views(views)
    base_manifest = read_json(base_root / "cache_manifest.json")
    derivative_manifest = read_json(derivative_manifest_path)
    derivative_manifest["manifest_path"] = str(Path(derivative_manifest_path).resolve())
    schema = str(base_manifest.get("schema"))
    if schema == REAL_SCHEMA:
        manifest = _derive_real_training_cache(
            base_root=base_root,
            base_manifest=base_manifest,
            derivative_manifest=derivative_manifest,
            output_root=out_root,
            views=selected_views,
            command=command,
        )
    elif schema == SYNTHRAD_SCHEMA:
        manifest = _derive_luna_training_cache(
            base_root=base_root,
            base_manifest=base_manifest,
            derivative_manifest=derivative_manifest,
            views=selected_views,
            output_root=out_root,
            command=command,
        )
    else:
        raise ValueError(f"unsupported base cache schema: {schema}")
    write_json(manifest, out_root / "cache_manifest.json")
    write_json(
        {
            "ok": not manifest.get("derived_training_cache", {}).get("missing_source_cases"),
            "cache_root": str(out_root),
            "base_cache_root": str(base_root),
            "derivative_manifest": str(Path(derivative_manifest_path).resolve()),
            "view_sample_counts": manifest.get("view_sample_counts"),
            "counts": manifest.get("counts"),
            "writes_body_or_projection_assets": False,
            "missing_source_cases": manifest.get("derived_training_cache", {}).get("missing_source_cases", []),
        },
        out_root / "reports" / "cache_summary.json",
    )
    missing = manifest.get("derived_training_cache", {}).get("missing_source_cases", [])
    if missing:
        raise RuntimeError(_missing_source_cases_error(missing, base_manifest))
    return manifest
