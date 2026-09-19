from __future__ import annotations

import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from .calibration import load_calibration_report
from .geometry import read_rtk_scan_geometry
from .manifest import FixedViewRecord, SelectedProjection, load_fixed_view_manifest, load_split_manifest, records_for_split
from .normalization import (
    CBCT_HU_MAX,
    CBCT_HU_MIN,
    CBCT_HU_SPAN,
    LEGACY_CBCT_NORMALIZATION,
    PROJECTION_MAX,
    PROJECTION_MIN,
    VIEW_TARGETS,
    cbct_normalization_profile,
    normalize_cbct_hu,
    normalize_projection,
)
from .normalization_profile import estimate_grouped_hu_profiles, parse_percentiles
from .point_blocks import OFFICIAL_GRID_SIZE, official_lattice_metadata
from .setup_augmentation import (
    SetupAugmentation,
    SetupAugmentationConfig,
    image_background_value,
    iter_augmentations,
    shift_image_vertical,
    transform_volume_setup,
    volume_background_value,
)
from .split_policy import PATIENT_LEVEL_POLICY, assign_train_test_split, normalize_split_policy, summarize_train_test_split
from .volume_grid import axis_convention, resized_spacing_preserve_fov_xyz, volume_grid_payload

PROJECTION_CORRECTION_NONE = "none"
PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN = (
    "self_projection1208_patient_dark_to_bright_mean"
)
PROJECTION_CORRECTION_MODES = (
    PROJECTION_CORRECTION_NONE,
    PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
)
PROJECTION_CORRECTION_PROFILE_SCHEMA = "x2cbct_self_projection1208_projection_correction_profile.v1"
SELF_PROJECTION1208_DARK_CASES = frozenset(
    {
        (0, 0),
        (1, 3),
        (1, 4),
        (3, 0),
        (3, 3),
        (3, 5),
        (3, 9),
        (7, 0),
        (7, 2),
        (7, 4),
        (7, 9),
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 6),
        (9, 10),
        (11, 1),
        (11, 3),
        (11, 4),
        (11, 5),
        (12, 1),
        (12, 4),
        (12, 6),
        (14, 0),
        (14, 1),
        (14, 6),
        (14, 10),
        (15, 3),
        (15, 5),
        (15, 8),
        (15, 14),
        (19, 1),
        (19, 5),
        (19, 6),
        (24, 8),
        (24, 9),
        (24, 12),
        (24, 18),
        (27, 6),
    }
)


@dataclass(frozen=True)
class CacheBuildOptions:
    project_root: Path
    output_root: Path
    views: tuple[str, ...] = ("view_2", "view_5", "view_10")
    projection_size: int = OFFICIAL_GRID_SIZE
    target_size: int = OFFICIAL_GRID_SIZE
    max_train_cases: int | None = None
    max_test_cases: int | None = None
    dry_run: bool = False
    overwrite: bool = False
    command: str = ""
    hu_window_mode: str = "fixed"
    hu_percentiles: tuple[float, float] = (0.5, 99.5)
    hu_fixed_min: float = CBCT_HU_MIN
    hu_fixed_max: float = CBCT_HU_MAX
    hu_sample_voxels_per_case: int = 200_000
    split_policy: str = PATIENT_LEVEL_POLICY
    split_train_ratio: float = 0.8
    split_seed: int = 20260517
    projection_correction_mode: str = PROJECTION_CORRECTION_NONE
    projection_correction_view_key: str = "view_10"
    projection_correction_profile_path: Path | None = None
    setup_augmentation: SetupAugmentationConfig = field(default_factory=SetupAugmentationConfig)
    target_volume_mode: str = "full"
    progress_interval: int = 25
    training_base_manifest_path: Path | None = None
    records_manifest_path: Path | None = None
    records_manifest_split: str = "test"
    records_output_split: str = "test"
    allow_missing_required_case_files: bool = True


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out


def default_projection_correction_profile_path(project_root: Path) -> Path:
    return Path(project_root) / "data_manifest" / "self_projection1208_projection_correction_profile.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_key(record: FixedViewRecord) -> str:
    return f"patient{record.patient_id}_case{record.case_id}"


def augmented_case_key(record: FixedViewRecord, augmentation: SetupAugmentation, *, enabled: bool) -> str:
    key = case_key(record)
    return f"{key}_{augmentation.id}" if enabled else key


def relative_to_cache(path: Path, cache_root: Path) -> str:
    return path.resolve().relative_to(cache_root.resolve()).as_posix()


def _nearest_ancestor(path: Path, name: str) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == name:
            return candidate
    return None


def _suffix_after_part(value: str, marker: str) -> Path | None:
    parts = tuple(part for part in value.replace("\\", "/").split("/") if part and part != ".")
    for index, part in enumerate(parts):
        if part == marker and index + 1 < len(parts):
            return Path(*parts[index + 1 :])
    return None


def resolve_cache_path(cache_root: Path, relative_path: str) -> Path:
    value = str(relative_path).replace("\\", "/")
    path = Path(value)
    cache_root = Path(cache_root).resolve()
    data_cache_root = _nearest_ancestor(cache_root, "data_cache")
    project_root = data_cache_root.parent if data_cache_root is not None else None
    if project_root is not None:
        project_suffix = _suffix_after_part(value, project_root.name)
        if project_suffix is not None:
            remapped = project_root / project_suffix
            if not path.is_absolute() or not path.exists():
                return remapped
        data_cache_suffix = _suffix_after_part(value, "data_cache")
        if data_cache_suffix is not None:
            remapped = data_cache_root / data_cache_suffix
            if not path.is_absolute() or not path.exists():
                return remapped
    if path.is_absolute():
        return path
    return cache_root / path


def _read_image(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, np.asarray(sitk.GetArrayFromImage(image))


def _projection_spacing_yx(image: sitk.Image) -> tuple[float, float]:
    spacing = image.GetSpacing()
    return float(spacing[1]), float(spacing[0])


def _resize_projection(array: np.ndarray, size: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(array, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"projection must be 2D after squeeze, got shape {arr.shape}")
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def _normalized_projection_mean(array: np.ndarray) -> float:
    return float(np.mean(normalize_projection(array)))


def _normalized_projection_clip_high_ratio(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float32)
    return float(np.mean(arr > PROJECTION_MAX))


def _resize_volume_to_cube(array: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"CBCT volume must be 3D, got shape {arr.shape}")
    if tuple(arr.shape) == (size, size, size):
        return arr
    factors = [size / dim for dim in arr.shape]
    return scipy.ndimage.zoom(arr, factors, order=1, prefilter=False).astype(np.float32)


def _geometry_gantry_angles(geometry_path: Path) -> list[float]:
    root = ET.parse(geometry_path).getroot()
    out: list[float] = []
    for projection in root.findall("Projection"):
        text = projection.findtext("GantryAngle")
        out.append(float(text) if text is not None else float("nan"))
    return out


def _angular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _projection_ordinal(path: Path) -> int | None:
    try:
        return int(path.stem.split("_", 1)[0])
    except ValueError:
        return None


def _nearest_projection_for_angle(record: FixedViewRecord, target_angle_deg: float) -> SelectedProjection:
    angles = _geometry_gantry_angles(record.case_path / "Geometry")
    candidates = []
    for path in record.case_path.glob("*.nii.gz"):
        if path.name == "CBCT.nii.gz":
            continue
        ordinal = _projection_ordinal(path)
        if ordinal is None or ordinal >= len(angles):
            continue
        gantry = angles[ordinal]
        if not math.isfinite(gantry):
            continue
        candidates.append(
            SelectedProjection(
                target_angle_deg=float(target_angle_deg),
                ordinal=int(ordinal),
                file=path.name,
                gantry_angle_deg=float(gantry),
                angular_residual_deg=float(_angular_distance_deg(gantry, target_angle_deg)),
            )
        )
    if not candidates:
        raise ValueError(f"no usable projection candidates under {record.case_path}")
    return min(candidates, key=lambda item: (item.angular_residual_deg, item.ordinal, item.file))


def _selected_for_augmentation(record: FixedViewRecord, view_key: str, augmentation: SetupAugmentation, *, enabled: bool) -> list[SelectedProjection]:
    if not enabled or int(augmentation.rotation_deg) == 0:
        return list(record.selected[view_key])
    return [
        _nearest_projection_for_angle(record, float(selected.target_angle_deg) + float(augmentation.rotation_deg))
        for selected in record.selected[view_key]
    ]


def _detector_pixel_shift_for_si(
    shift_mm: int,
    *,
    sid_mm: float,
    sdd_mm: float,
    detector_spacing_y_mm: float,
    detector_height: int,
    projection_size: int,
) -> dict[str, float]:
    detector_shift_mm = -float(shift_mm) * float(sdd_mm) / float(sid_mm)
    raw_pixel_shift_y = detector_shift_mm / float(detector_spacing_y_mm)
    resized_pixel_shift_y = raw_pixel_shift_y * ((float(projection_size) - 1.0) / max(float(detector_height) - 1.0, 1.0))
    return {
        "si_shift_mm": float(shift_mm),
        "sid_mm": float(sid_mm),
        "sdd_mm": float(sdd_mm),
        "magnification": float(sdd_mm) / float(sid_mm),
        "detector_shift_mm": float(detector_shift_mm),
        "raw_pixel_shift_y": float(raw_pixel_shift_y),
        "resized_pixel_shift_y": float(resized_pixel_shift_y),
    }


def _resized_spacing_xyz(image: sitk.Image, shape_zyx: tuple[int, int, int], target_size: int) -> tuple[float, float, float]:
    spacing = tuple(float(item) for item in image.GetSpacing())
    target_shape = (int(target_size), int(target_size), int(target_size))
    return resized_spacing_preserve_fov_xyz(shape_zyx, spacing, target_shape)


def _records_by_split(project_root: Path) -> dict[str, list[FixedViewRecord]]:
    return {
        "train": records_for_split(project_root, "train"),
        "test": records_for_split(project_root, "test"),
    }


def _records_from_training_base_manifest(path: Path, fixed: Any) -> list[FixedViewRecord]:
    payload = read_json(path)
    indexed = fixed.by_case()
    records: list[FixedViewRecord] = []
    missing: list[str] = []
    for item in payload.get("cases", []):
        patient_id = str(item["patient_id"])
        case_id = str(item["case_id"])
        key = (patient_id, case_id)
        record = indexed.get(key)
        if record is None:
            missing.append(f"{patient_id}/{case_id}")
            continue
        records.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} training-base cases are missing from fixed-view manifest: {preview}")
    if not records:
        raise ValueError(f"training-base manifest has no cases: {path}")
    return sorted(records, key=lambda record: (record.patient_id, record.case_id))


def _records_from_derivative_manifest(path: Path, fixed: Any, *, source_split: str) -> list[FixedViewRecord]:
    payload = read_json(path)
    indexed = fixed.by_case()
    split_payload = payload.get("splits", {}).get(source_split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"derivative manifest has no split {source_split!r}: {path}")
    records: list[FixedViewRecord] = []
    missing: list[str] = []
    for item in split_payload.get("cases", []):
        patient_id = str(item["patient_id"])
        case_id = str(item["case_id"])
        key = (patient_id, case_id)
        record = indexed.get(key)
        if record is None:
            missing.append(f"{patient_id}/{case_id}")
            continue
        records.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} derivative cases are missing from fixed-view manifest: {preview}")
    return sorted(records, key=lambda record: (record.patient_id, record.case_id))


def _training_base_split_summary(records: list[FixedViewRecord], *, manifest_path: Path) -> dict[str, Any]:
    patients = sorted({record.patient_id for record in records})
    cases = sorted(case_key(record) for record in records)
    return {
        "policy": "training_base_manifest",
        "split_unit": "precomputed_training_case",
        "seed": None,
        "train_ratio": 1.0,
        "train_patient_count": len(patients),
        "test_patient_count": 0,
        "train_case_count": len(records),
        "test_case_count": 0,
        "patient_overlap": [],
        "case_overlap_count": 0,
        "case_overlap": [],
        "allow_patient_overlap": False,
        "training_base_manifest": str(manifest_path.resolve()),
        "training_base_cases": cases,
    }


def _single_split_summary(
    records: list[FixedViewRecord],
    *,
    manifest_path: Path,
    output_split: str,
    policy: str,
) -> dict[str, Any]:
    patients = sorted({record.patient_id for record in records})
    cases = sorted(case_key(record) for record in records)
    train_count = len(records) if output_split == "train" else 0
    test_count = len(records) if output_split == "test" else 0
    return {
        "policy": policy,
        "split_unit": "precomputed_case",
        "seed": None,
        "train_ratio": None,
        "train_patient_count": len(patients) if output_split == "train" else 0,
        "test_patient_count": len(patients) if output_split == "test" else 0,
        "train_case_count": train_count,
        "test_case_count": test_count,
        "patient_overlap": [],
        "case_overlap_count": 0,
        "case_overlap": [],
        "allow_patient_overlap": policy == "mixed",
        "records_manifest": str(manifest_path.resolve()),
        "records_manifest_output_split": output_split,
        "records_manifest_cases": cases,
    }


def _missing_required_case_files(record: FixedViewRecord, views: tuple[str, ...]) -> list[dict[str, str]]:
    required = [
        ("cbct", record.case_path / "CBCT.nii.gz"),
        ("geometry", record.case_path / "Geometry"),
    ]
    for view_key in views:
        for selected in record.selected[view_key]:
            required.append((f"projection:{view_key}:{selected.file}", record.case_path / selected.file))
    missing = []
    for kind, path in required:
        if not path.exists():
            missing.append({"kind": kind, "path": str(path)})
    return missing


def _filter_available_records(
    records_by_split: dict[str, list[FixedViewRecord]],
    views: tuple[str, ...],
) -> tuple[dict[str, list[FixedViewRecord]], list[dict[str, Any]]]:
    filtered: dict[str, list[FixedViewRecord]] = {split: [] for split in records_by_split}
    excluded: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        for record in records:
            missing = _missing_required_case_files(record, views)
            if missing:
                excluded.append(
                    {
                        "split": split,
                        "patient_id": record.patient_id,
                        "case_id": record.case_id,
                        "case_path": str(record.case_path),
                        "reason": "missing_required_case_files",
                        "missing": missing,
                    }
                )
                continue
            filtered[split].append(record)
    return filtered, excluded


def _missing_required_files_error(excluded: list[dict[str, Any]]) -> str:
    first = excluded[0]
    label = f"patient{first.get('patient_id')}_case{first.get('case_id')}"
    missing_kinds = ", ".join(str(item.get("kind")) for item in first.get("missing", [])[:4])
    return (
        f"{len(excluded)} cases are missing required local files before cache build. "
        f"First excluded case: {label} missing {missing_kinds}. "
        "Restore the missing dataset files or pass an explicit allow-missing option from the wrapper script "
        "when intentionally building a local subset."
    )


def _limit_records_by_split(
    records_by_split: dict[str, list[FixedViewRecord]],
    max_train: int | None,
    max_test: int | None,
) -> dict[str, list[FixedViewRecord]]:
    limited = {split: list(records) for split, records in records_by_split.items()}
    if max_train is not None:
        limited["train"] = limited["train"][: int(max_train)]
    if max_test is not None:
        limited["test"] = limited["test"][: int(max_test)]
    return limited


def _projection_case_rows(
    records_by_split: dict[str, list[FixedViewRecord]],
    *,
    projection_size: int,
    view_key: str,
    progress_interval: int = 25,
) -> list[dict[str, Any]]:
    rows = []
    total_cases = sum(len(records) for records in records_by_split.values())
    interval = max(0, int(progress_interval))
    done_cases = 0
    start_time = time.perf_counter()
    for split, records in records_by_split.items():
        for record in records:
            means = []
            for selected in record.selected[view_key]:
                _image, raw = _read_image(record.case_path / selected.file)
                resized = _resize_projection(np.squeeze(raw), projection_size)
                means.append(_normalized_projection_mean(resized))
            rows.append(
                {
                    "split": split,
                    "patient_id": int(record.patient_id),
                    "case_id": int(record.case_id),
                    "case_key": case_key(record),
                    "norm_mean": float(np.mean(means)),
                }
            )
            done_cases += 1
            if interval > 0 and (done_cases == 1 or done_cases == total_cases or done_cases % interval == 0):
                elapsed = time.perf_counter() - start_time
                print(
                    (
                        f"[cache] projection-correction prescan "
                        f"{done_cases}/{total_cases} cases "
                        f"split={split} patient={record.patient_id} case={record.case_id} "
                        f"elapsed={elapsed:.1f}s"
                    ),
                    flush=True,
                )
    return rows


def _patient_dark_to_bright_correction_profile(
    case_rows: list[dict[str, Any]],
    *,
    dark_cases: frozenset[tuple[int, int]] = SELF_PROJECTION1208_DARK_CASES,
) -> dict[str, Any]:
    by_patient: dict[int, list[dict[str, Any]]] = {}
    for row in case_rows:
        by_patient.setdefault(int(row["patient_id"]), []).append(row)

    patients = []
    case_offset_table: dict[str, dict[str, Any]] = {}
    for patient_id, rows in sorted(by_patient.items()):
        dark_rows = [
            row for row in rows if (int(row["patient_id"]), int(row["case_id"])) in dark_cases
        ]
        if not dark_rows:
            continue
        bright_rows = [
            row for row in rows if (int(row["patient_id"]), int(row["case_id"])) not in dark_cases
        ]
        if not bright_rows:
            raise ValueError(f"patient{patient_id} has dark cases but no bright reference cases")
        dark_mean = float(np.mean([float(row["norm_mean"]) for row in dark_rows]))
        bright_mean = float(np.mean([float(row["norm_mean"]) for row in bright_rows]))
        if dark_mean <= 0.0:
            raise ValueError(f"patient{patient_id} dark mean must be positive, got {dark_mean}")
        delta_norm = float(bright_mean - dark_mean)
        delta_raw = float(delta_norm * (PROJECTION_MAX - PROJECTION_MIN))
        patients.append(
            {
                "patient_id": int(patient_id),
                "dark_case_count": len(dark_rows),
                "bright_case_count": len(bright_rows),
                "dark_mean": dark_mean,
                "bright_mean": bright_mean,
                "delta_norm": delta_norm,
                "delta_raw": delta_raw,
                "dark_cases": [row["case_key"] for row in sorted(dark_rows, key=lambda item: int(item["case_id"]))],
                "bright_cases": [
                    row["case_key"] for row in sorted(bright_rows, key=lambda item: int(item["case_id"]))
                ],
            }
        )
        for row in dark_rows:
            case_offset_table[str(row["case_key"])] = {
                "patient_id": int(patient_id),
                "case_id": int(row["case_id"]),
                "delta_norm": delta_norm,
                "delta_raw": delta_raw,
                "case_norm_mean": float(row["norm_mean"]),
                "patient_dark_mean": dark_mean,
                "patient_bright_mean": bright_mean,
            }

    return {
        "mode": PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
        "rule": "for each patient, add patient bright mean - patient dark mean to listed dark cases before fixed projection normalization",
        "dark_case_source": "Coff0xc manual review, 2026-06-01",
        "correction_domain": "normalized_projection_mean",
        "raw_delta_rule": "delta_raw = delta_norm * (projection_max - projection_min)",
        "dark_case_count_in_profile": len(case_offset_table),
        "patient_count_in_profile": len(patients),
        "patients": patients,
        "case_offset_table": case_offset_table,
    }


def _projection_correction_profile(
    mode: str,
    records_by_split: dict[str, list[FixedViewRecord]],
    *,
    projection_size: int,
    view_key: str,
    progress_interval: int = 25,
) -> dict[str, Any]:
    if mode == PROJECTION_CORRECTION_NONE:
        return {"mode": PROJECTION_CORRECTION_NONE, "rule": "no projection brightness correction"}
    if mode not in PROJECTION_CORRECTION_MODES:
        raise ValueError(f"unsupported projection_correction_mode: {mode}")
    if view_key not in VIEW_TARGETS:
        raise ValueError(f"projection_correction_view_key must be one of {tuple(VIEW_TARGETS)}, got {view_key}")
    total_cases = sum(len(records) for records in records_by_split.values())
    total_projections = sum(len(record.selected[view_key]) for records in records_by_split.values() for record in records)
    print(
        (
            f"[cache] projection-correction prescan started "
            f"mode={mode} view={view_key} cases={total_cases} projections={total_projections}"
        ),
        flush=True,
    )
    rows = _projection_case_rows(
        records_by_split,
        projection_size=projection_size,
        view_key=view_key,
        progress_interval=progress_interval,
    )
    profile = _patient_dark_to_bright_correction_profile(rows)
    profile["source_view_key"] = view_key
    profile["source_projection_size"] = int(projection_size)
    profile["source_case_count"] = len(rows)
    print(
        (
            f"[cache] projection-correction prescan finished "
            f"cases={len(rows)} dark_cases={profile['dark_case_count_in_profile']} "
            f"patients={profile['patient_count_in_profile']}"
        ),
        flush=True,
    )
    return profile


def build_self_projection1208_projection_correction_profile(
    *,
    project_root: Path,
    output_path: Path | None = None,
    projection_size: int = OFFICIAL_GRID_SIZE,
    view_key: str = "view_10",
    progress_interval: int = 25,
    allow_missing_required_case_files: bool = False,
    command: str = "",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    fixed = load_fixed_view_manifest(root)
    available_records_by_split, local_excluded_cases = _filter_available_records(
        {"all": fixed.records},
        (view_key,),
    )
    if local_excluded_cases and not allow_missing_required_case_files:
        raise RuntimeError(_missing_required_files_error(local_excluded_cases))
    profile = _projection_correction_profile(
        PROJECTION_CORRECTION_SELF_PROJECTION1208_PATIENT_DARK_TO_BRIGHT_MEAN,
        available_records_by_split,
        projection_size=projection_size,
        view_key=view_key,
        progress_interval=progress_interval,
    )
    profile.update(
        {
            "schema": PROJECTION_CORRECTION_PROFILE_SCHEMA,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_command": command,
            "fixed_view_manifest": {
                "path": str(fixed.path.resolve()),
                "sha256": sha256_file(fixed.path),
                "included_case_count": int(fixed.included_case_count),
                "excluded_case_count": int(fixed.excluded_case_count),
            },
            "local_dataset_filter": {
                "policy": "exclude_missing_required_case_files_before_profile_build",
                "view_key": view_key,
                "excluded_case_count": len(local_excluded_cases),
                "excluded_cases": local_excluded_cases,
            },
        }
    )
    out = Path(output_path) if output_path is not None else default_projection_correction_profile_path(root)
    write_json(profile, out)
    profile["profile_path"] = str(out.resolve())
    return profile


def _load_projection_correction_profile(
    path: Path,
    *,
    expected_mode: str,
    projection_size: int,
    view_key: str,
) -> dict[str, Any]:
    profile = read_json(path)
    if profile.get("schema") != PROJECTION_CORRECTION_PROFILE_SCHEMA:
        raise ValueError(f"unsupported projection correction profile schema: {profile.get('schema')!r}")
    if profile.get("mode") != expected_mode:
        raise ValueError(f"projection correction profile mode mismatch: {profile.get('mode')!r} != {expected_mode!r}")
    if str(profile.get("source_view_key")) != str(view_key):
        raise ValueError(
            f"projection correction profile view mismatch: {profile.get('source_view_key')!r} != {view_key!r}"
        )
    if int(profile.get("source_projection_size", -1)) != int(projection_size):
        raise ValueError(
            "projection correction profile projection-size mismatch: "
            f"{profile.get('source_projection_size')!r} != {int(projection_size)}"
        )
    if not isinstance(profile.get("case_offset_table"), dict):
        raise ValueError("projection correction profile is missing case_offset_table")
    profile = dict(profile)
    profile["profile_source"] = "external_profile_file"
    profile["profile_path"] = str(Path(path).resolve())
    profile["profile_sha256"] = sha256_file(path)
    return profile


def _projection_payload(
    record: FixedViewRecord,
    selected: Any,
    augmentation: SetupAugmentation,
    cache_root: Path,
    projection_size: int,
    dry_run: bool,
    projection_correction: dict[str, Any],
    *,
    setup_enabled: bool,
) -> tuple[dict[str, Any], tuple[int, int], tuple[float, float]]:
    source_key = case_key(record)
    key = augmented_case_key(record, augmentation, enabled=setup_enabled)
    raw_path = record.case_path / selected.file
    image, raw = _read_image(raw_path)
    squeezed = np.squeeze(raw)
    detector_shape = tuple(int(value) for value in squeezed.shape[-2:])
    detector_spacing = _projection_spacing_yx(image)
    stem = Path(selected.file).stem
    out_path = cache_root / "projections" / key / f"{stem}.npy"
    meta_path = cache_root / "projections" / key / f"{stem}.json"
    resized = _resize_projection(squeezed, projection_size)
    offset_item = projection_correction.get("case_offset_table", {}).get(source_key)
    delta_raw = float(offset_item["delta_raw"]) if offset_item else 0.0
    delta_norm = float(offset_item["delta_norm"]) if offset_item else 0.0
    corrected = resized + delta_raw
    shift_metadata = None
    if setup_enabled and int(augmentation.translation_mm) != 0:
        geometry = read_rtk_scan_geometry(record.case_path / "Geometry")
        shift_metadata = _detector_pixel_shift_for_si(
            int(augmentation.translation_mm),
            sid_mm=geometry.source_to_isocenter_distance,
            sdd_mm=geometry.source_to_detector_distance,
            detector_spacing_y_mm=detector_spacing[0],
            detector_height=detector_shape[0],
            projection_size=int(projection_size),
        )
        fill = image_background_value(corrected)
        corrected = shift_image_vertical(corrected, shift_metadata["resized_pixel_shift_y"], fill_value=fill)
        shift_metadata["projection_fill_value"] = float(fill)
    pre_norm_mean = _normalized_projection_mean(resized)
    post_norm_mean = _normalized_projection_mean(corrected)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        norm = normalize_projection(corrected).astype(np.float16)
        np.save(out_path, norm)
    correction_metadata = {
        "mode": projection_correction["mode"],
        "method": "additive_offset",
        "applied": offset_item is not None,
        "delta_norm": delta_norm,
        "delta_raw": delta_raw,
        "pre_norm_mean": pre_norm_mean,
        "post_norm_mean": post_norm_mean,
        "post_clip_high_ratio": _normalized_projection_clip_high_ratio(corrected),
    }
    if offset_item:
        correction_metadata.update(
            {
                "patient_dark_mean": float(offset_item["patient_dark_mean"]),
                "patient_bright_mean": float(offset_item["patient_bright_mean"]),
            }
        )
    metadata = {
        "raw_projection_path": str(raw_path),
        "payload_path": relative_to_cache(out_path, cache_root),
        "metadata_path": relative_to_cache(meta_path, cache_root),
        "target_angle_deg": float(selected.target_angle_deg),
        "ordinal": int(selected.ordinal),
        "file": str(selected.file),
        "gantry_angle_deg": float(selected.gantry_angle_deg),
        "angular_residual_deg": float(selected.angular_residual_deg),
        "detector_shape": list(detector_shape),
        "detector_spacing_yx": list(detector_spacing),
        "projection_size": int(projection_size),
        "normalization": {"rule": "clip_0_8.6_div_8.6", "min": PROJECTION_MIN, "max": PROJECTION_MAX},
        "projection_correction": correction_metadata,
    }
    if setup_enabled:
        setup_payload: dict[str, Any] = {
            **augmentation.to_dict(),
            "source_patient_id": int(record.patient_id),
            "source_case_id": int(record.case_id),
            "source_case_key": source_key,
            "selected_source_file": str(selected.file),
            "selected_source_ordinal": int(selected.ordinal),
            "selected_target_angle_deg": float(selected.target_angle_deg),
            "selected_gantry_angle_deg": float(selected.gantry_angle_deg),
            "selected_angular_residual_deg": float(selected.angular_residual_deg),
        }
        if shift_metadata is not None:
            setup_payload["translation_projection_shift"] = shift_metadata
        metadata["setup_augmentation"] = setup_payload
    if not dry_run:
        write_json(metadata, meta_path)
    return metadata, detector_shape, detector_spacing


def _case_payload(
    record: FixedViewRecord,
    augmentation: SetupAugmentation,
    cache_root: Path,
    target_size: int,
    dry_run: bool,
    cbct_normalization: dict[str, Any],
    *,
    setup_enabled: bool,
    target_volume_mode: str,
) -> dict[str, Any]:
    key = augmented_case_key(record, augmentation, enabled=setup_enabled)
    source_key = case_key(record)
    case_dir = cache_root / "cases" / key
    target_mode = str(target_volume_mode)
    base_reference_target = setup_enabled and target_mode == "base_reference"
    cbct_dir = cache_root / "cases" / (source_key if base_reference_target else key)
    cbct_path = cbct_dir / "cbct_norm.npy"
    case_manifest_path = case_dir / "case_manifest.json"
    geometry_summary_path = case_dir / "geometry_summary.json"
    cbct_image, cbct_raw = _read_image(record.case_path / "CBCT.nii.gz")
    raw_shape_zyx = tuple(int(item) for item in cbct_raw.shape)
    shape_zyx = (int(target_size), int(target_size), int(target_size))
    raw_spacing_xyz = tuple(float(item) for item in cbct_image.GetSpacing())
    spacing_xyz = _resized_spacing_xyz(cbct_image, raw_shape_zyx, int(target_size))
    origin_xyz = tuple(float(item) for item in cbct_image.GetOrigin())
    direction = [float(item) for item in cbct_image.GetDirection()]
    geometry = read_rtk_scan_geometry(record.case_path / "Geometry")
    setup_payload = None
    if setup_enabled:
        raw_fill_value = volume_background_value(cbct_raw)
        normalized_fill_value = float(
            normalize_cbct_hu(
                np.asarray([raw_fill_value], dtype=np.float32),
                normalization=cbct_normalization,
            )[0]
        )
        setup_payload = {
            **augmentation.to_dict(),
            "source_patient_id": int(record.patient_id),
            "source_case_id": int(record.case_id),
            "source_case_key": source_key,
            "volume_transform_order": "rotate_yaw_then_translate_si",
            "volume_fill_value": float(raw_fill_value),
            "target_sampling_fill_value": normalized_fill_value,
        }
    if not dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)
        cbct_dir.mkdir(parents=True, exist_ok=True)
        if base_reference_target:
            cbct_for_resize = cbct_raw
        elif setup_enabled:
            cbct_for_resize = transform_volume_setup(
                cbct_raw,
                translation_mm=int(augmentation.translation_mm),
                rotation_deg=int(augmentation.rotation_deg),
                spacing_z_mm=raw_spacing_xyz[2],
                fill_value=float(setup_payload["volume_fill_value"]),
            )
        else:
            cbct_for_resize = cbct_raw
        if not (base_reference_target and cbct_path.exists()):
            resized = _resize_volume_to_cube(cbct_for_resize, int(target_size))
            np.save(cbct_path, normalize_cbct_hu(resized, normalization=cbct_normalization).astype(np.float16))
    geometry_summary = {
        "geometry_path": str(record.case_path / "Geometry"),
        "source_to_isocenter_distance": geometry.source_to_isocenter_distance,
        "source_to_detector_distance": geometry.source_to_detector_distance,
    }
    grid_payload = volume_grid_payload(
        raw_shape_zyx=raw_shape_zyx,
        raw_spacing_xyz=raw_spacing_xyz,
        raw_origin_xyz=origin_xyz,
        raw_direction=direction,
        processed_shape_zyx=shape_zyx,
        processed_spacing_xyz=spacing_xyz,
        processed_origin_xyz=origin_xyz,
        processed_direction=direction,
    )
    case_manifest = {
        "case_key": key,
        "patient_id": record.patient_id,
        "case_id": record.case_id,
        "geometry_branch": "real_projection",
        "target_volume_mode": target_mode,
        **grid_payload,
        "raw_case_path": str(record.case_path),
        "case_manifest_path": relative_to_cache(case_manifest_path, cache_root),
        "cbct_payload_path": relative_to_cache(cbct_path, cache_root),
        "geometry_summary_path": relative_to_cache(geometry_summary_path, cache_root),
        # Backward-compatible aliases consumed by current dataset/viewer code.
        "raw_shape_zyx": list(raw_shape_zyx),
        "raw_spacing_xyz": list(raw_spacing_xyz),
        "shape_zyx": list(shape_zyx),
        "spacing_xyz": list(spacing_xyz),
        "origin_xyz": list(origin_xyz),
        "direction": direction,
        "target_size": int(target_size),
        "preprocess": {
            "policy": "resize_to_target_size_preserve_physical_extent",
            "target_size": int(target_size),
            "resize_method": "scipy.ndimage.zoom_order1",
            "processed_spacing_derived_from_raw_extent": True,
        },
        "projection_geometry": {
            "type": "real_projection",
            "rtk_geometry_path": str(record.case_path / "Geometry"),
            "source_to_isocenter_distance": geometry.source_to_isocenter_distance,
            "source_to_detector_distance": geometry.source_to_detector_distance,
        },
        "point_blocks": official_lattice_metadata(int(target_size)),
        "normalization": cbct_normalization,
    }
    if setup_payload is not None:
        case_manifest["setup_augmentation"] = setup_payload
    if not dry_run:
        write_json(geometry_summary, geometry_summary_path)
        write_json(case_manifest, case_manifest_path)
    return case_manifest


def _sample_payload(
    record: FixedViewRecord,
    augmentation: SetupAugmentation,
    split: str,
    view_key: str,
    cache_root: Path,
    case_manifest: dict[str, Any],
    projection_payloads: list[dict[str, Any]],
    dry_run: bool,
    *,
    setup_enabled: bool,
) -> dict[str, Any]:
    key = augmented_case_key(record, augmentation, enabled=setup_enabled)
    sample_id = f"{key}_{view_key}"
    sample_path = cache_root / view_key / split / f"{sample_id}.npz"
    first = projection_payloads[0]
    if not dry_run:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_id": np.array(sample_id),
            "split": np.array(split),
            "view_key": np.array(view_key),
            "case_key": np.array(key),
            "patient_id": np.array(record.patient_id),
            "case_id": np.array(record.case_id),
            "case_manifest_path": np.array(case_manifest["case_manifest_path"]),
            "cbct_payload_path": np.array(case_manifest["cbct_payload_path"]),
            "projection_payload_paths": np.array([item["payload_path"] for item in projection_payloads]),
            "projection_files": np.array([item["file"] for item in projection_payloads]),
            "target_angles_deg": np.array([item["target_angle_deg"] for item in projection_payloads], dtype=np.float32),
            "gantry_angles_deg": np.array([item["gantry_angle_deg"] for item in projection_payloads], dtype=np.float32),
            "angular_residuals_deg": np.array([item["angular_residual_deg"] for item in projection_payloads], dtype=np.float32),
            "detector_shape": np.array(first["detector_shape"], dtype=np.int32),
            "detector_spacing_yx": np.array(first["detector_spacing_yx"], dtype=np.float32),
            "projection_size": np.array(first["projection_size"], dtype=np.int32),
            "target_volume_mode": np.array(case_manifest.get("target_volume_mode", "full")),
        }
        if setup_enabled:
            payload.update(
                {
                    "augmentation_id": np.array(augmentation.id),
                    "translation_mm": np.array(int(augmentation.translation_mm), dtype=np.int32),
                    "rotation_deg": np.array(int(augmentation.rotation_deg), dtype=np.int32),
                    "source_patient_id": np.array(record.patient_id),
                    "source_case_id": np.array(record.case_id),
                    "source_case_key": np.array(case_key(record)),
                    "volume_fill_value": np.array(float(case_manifest["setup_augmentation"]["volume_fill_value"]), dtype=np.float32),
                    "target_sampling_fill_value": np.array(
                        float(case_manifest["setup_augmentation"].get("target_sampling_fill_value", case_manifest["setup_augmentation"]["volume_fill_value"])),
                        dtype=np.float32,
                    ),
                }
            )
        np.savez_compressed(sample_path, **payload)
    sample = {
        "sample_id": sample_id,
        "split": split,
        "view_key": view_key,
        "geometry_branch": "real_projection",
        "patient_id": record.patient_id,
        "case_id": record.case_id,
        "sample_path": relative_to_cache(sample_path, cache_root),
        "case_manifest_path": case_manifest["case_manifest_path"],
        "cbct_payload_path": case_manifest["cbct_payload_path"],
        "target_volume_mode": case_manifest.get("target_volume_mode", "full"),
        "projection_payload_paths": [item["payload_path"] for item in projection_payloads],
        "selected_projections": projection_payloads,
        "projection_geometry": {
            "type": "real_projection",
            "detector_shape": first["detector_shape"],
            "detector_spacing_yx": first["detector_spacing_yx"],
            "gantry_angles_deg": [item["gantry_angle_deg"] for item in projection_payloads],
            "selected_projection_files": [item["file"] for item in projection_payloads],
        },
    }
    if setup_enabled:
        sample["setup_augmentation"] = {
            **augmentation.to_dict(),
            "source_patient_id": int(record.patient_id),
            "source_case_id": int(record.case_id),
            "source_case_key": case_key(record),
            "volume_fill_value": float(case_manifest["setup_augmentation"]["volume_fill_value"]),
            "target_sampling_fill_value": float(
                case_manifest["setup_augmentation"].get("target_sampling_fill_value", case_manifest["setup_augmentation"]["volume_fill_value"])
            ),
        }
    return sample


def build_difnet_v0b_cache(options: CacheBuildOptions) -> dict[str, Any]:
    root = options.project_root.resolve()
    cache_root = options.output_root.resolve()
    if cache_root.exists() and any(cache_root.iterdir()) and not options.overwrite:
        raise FileExistsError(f"cache output already exists: {cache_root}")
    split_policy = normalize_split_policy(options.split_policy)
    projection_correction_mode = str(options.projection_correction_mode)
    if projection_correction_mode not in PROJECTION_CORRECTION_MODES:
        raise ValueError(f"unsupported projection_correction_mode: {projection_correction_mode}")
    availability_views = options.views
    if projection_correction_mode != PROJECTION_CORRECTION_NONE:
        availability_views = tuple(dict.fromkeys((*options.views, options.projection_correction_view_key)))
    calibration = load_calibration_report(root, require_pass=True)
    fixed = load_fixed_view_manifest(root)
    fixed_raw = read_json(fixed.path)
    split_manifest_payload: dict[str, Any] | None
    custom_split_summary: dict[str, Any] | None = None
    if options.records_output_split not in {"train", "test"}:
        raise ValueError("records_output_split must be train or test")
    if options.records_manifest_split not in {"train", "test"}:
        raise ValueError("records_manifest_split must be train or test")
    if options.training_base_manifest_path is not None and options.records_manifest_path is not None:
        raise ValueError("training_base_manifest_path and records_manifest_path cannot both be set")
    if options.training_base_manifest_path is not None:
        training_base_manifest_path = options.training_base_manifest_path.resolve()
        training_base_records = _records_from_training_base_manifest(training_base_manifest_path, fixed)
        split_manifest_payload = {
            "path": str(training_base_manifest_path),
            "sha256": sha256_file(training_base_manifest_path),
            "protocol": "training_base_union_manifest",
        }
        indexed_records_by_split = {"train": training_base_records, "test": []}
        available_records_by_split, local_excluded_cases = _filter_available_records(
            indexed_records_by_split,
            availability_views,
        )
        split_seed = int(options.split_seed)
        split_train_ratio = 1.0
        custom_split_summary = _training_base_split_summary(
            available_records_by_split["train"],
            manifest_path=training_base_manifest_path,
        )
    elif options.records_manifest_path is not None:
        records_manifest_path = options.records_manifest_path.resolve()
        manifest_records = _records_from_derivative_manifest(
            records_manifest_path,
            fixed,
            source_split=options.records_manifest_split,
        )
        indexed_records_by_split = {"train": [], "test": []}
        indexed_records_by_split[options.records_output_split] = manifest_records
        split_manifest_payload = {
            "path": str(records_manifest_path),
            "sha256": sha256_file(records_manifest_path),
            "protocol": "derivative_manifest",
            "source_split": options.records_manifest_split,
            "output_split": options.records_output_split,
        }
        available_records_by_split, local_excluded_cases = _filter_available_records(
            indexed_records_by_split,
            availability_views,
        )
        split_seed = int(options.split_seed)
        split_train_ratio = float(options.split_train_ratio)
        derivative_payload = read_json(records_manifest_path)
        custom_split_summary = _single_split_summary(
            available_records_by_split[options.records_output_split],
            manifest_path=records_manifest_path,
            output_split=options.records_output_split,
            policy=str(derivative_payload.get("split_policy", "derivative_manifest")),
        )
    elif split_policy == PATIENT_LEVEL_POLICY:
        split_manifest = load_split_manifest(root, fail_on_overlap=True)
        split_raw = read_json(split_manifest.path)
        split_manifest_payload = {
            "path": str(split_manifest.path),
            "sha256": sha256_file(split_manifest.path),
            "protocol": split_manifest.protocol,
        }
        split_seed = int(split_raw.get("seed", options.split_seed))
        split_train_ratio = float(1.0 - float(split_raw.get("test_ratio", 1.0 - options.split_train_ratio)))
    else:
        split_manifest_payload = None
        available_all, local_excluded_cases = _filter_available_records({"all": fixed.records}, availability_views)
        mixed = assign_train_test_split(
            available_all["all"],
            policy=split_policy,
            train_ratio=options.split_train_ratio,
            seed=options.split_seed,
        )
        available_records_by_split = {"train": mixed.train, "test": mixed.test}
        indexed_records_by_split = {"train": mixed.train, "test": mixed.test}
        split_seed = int(options.split_seed)
        split_train_ratio = float(options.split_train_ratio)
    if (
        options.training_base_manifest_path is None
        and options.records_manifest_path is None
        and split_policy == PATIENT_LEVEL_POLICY
    ):
        indexed_records_by_split = _records_by_split(root)
        available_records_by_split, local_excluded_cases = _filter_available_records(
            indexed_records_by_split,
            availability_views,
        )
    if local_excluded_cases and not options.allow_missing_required_case_files:
        raise RuntimeError(_missing_required_files_error(local_excluded_cases))
    records_by_split = _limit_records_by_split(
        available_records_by_split,
        options.max_train_cases,
        options.max_test_cases,
    )
    effective_split_policy = custom_split_summary["policy"] if custom_split_summary is not None else split_policy
    print(
        (
            f"[cache] records ready train={len(records_by_split['train'])} "
            f"test={len(records_by_split['test'])} excluded={len(local_excluded_cases)} "
            f"split_policy={effective_split_policy}"
        ),
        flush=True,
    )
    split_summary = custom_split_summary or summarize_train_test_split(
        records_by_split["train"],
        records_by_split["test"],
        policy=split_policy,
        train_ratio=split_train_ratio,
        seed=split_seed,
    )
    if options.hu_window_mode == "legacy":
        cbct_normalization = dict(LEGACY_CBCT_NORMALIZATION)
        cbct_profile_case_stats = []
    elif options.hu_window_mode == "fixed":
        cbct_normalization = cbct_normalization_profile(
            hu_min=options.hu_fixed_min,
            hu_max=options.hu_fixed_max,
            profile_id=f"self_projection1208_fixed_hu_{options.hu_fixed_min:g}_{options.hu_fixed_max:g}",
            mode="fixed",
            source={"dataset": "self_projection1208", "split": "train"},
        )
        cbct_profile_case_stats = []
    else:
        profile_rows = [
            {
                "case_id": case_key(record),
                "patient_id": record.patient_id,
                "split": "train",
                "cbct_path": str(record.case_path / "CBCT.nii.gz"),
            }
            for record in records_by_split["train"]
        ]
        profiles, cbct_profile_case_stats = estimate_grouped_hu_profiles(
            profile_rows,
            image_path=lambda item: Path(item["cbct_path"]),
            group_key=lambda _item: "self_projection1208",
            source_split="train",
            target_size=int(options.target_size),
            percentiles=parse_percentiles(options.hu_percentiles),
            mask_policy="finite",
            max_voxels_per_case=options.hu_sample_voxels_per_case,
        )
        cbct_normalization = profiles["self_projection1208"]
    if projection_correction_mode == PROJECTION_CORRECTION_NONE:
        projection_correction = {"mode": PROJECTION_CORRECTION_NONE, "rule": "no projection brightness correction"}
    else:
        profile_path = (
            Path(options.projection_correction_profile_path).resolve()
            if options.projection_correction_profile_path is not None
            else default_projection_correction_profile_path(root).resolve()
        )
        if not profile_path.exists():
            raise FileNotFoundError(
                "projection correction profile is required before cache build: "
                f"{profile_path}. Run tools/build_self_projection1208_projection_correction_profile.py first."
            )
        projection_correction = _load_projection_correction_profile(
            profile_path,
            expected_mode=projection_correction_mode,
            projection_size=options.projection_size,
            view_key=options.projection_correction_view_key,
        )
        print(
            (
                f"[cache] projection-correction profile loaded "
                f"path={profile_path} cases={projection_correction.get('source_case_count')} "
                f"dark_cases={projection_correction.get('dark_case_count_in_profile')}"
            ),
            flush=True,
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    augmentations = iter_augmentations(options.setup_augmentation)
    setup_enabled = bool(options.setup_augmentation.enabled)
    total_source_cases = sum(len(records) for records in records_by_split.values())
    total_augmented_cases = total_source_cases * len(augmentations)
    total_samples = total_augmented_cases * len(options.views)
    progress_interval = max(0, int(options.progress_interval))
    build_start_time = time.perf_counter()
    print(
        (
            f"[cache] build started source_cases={total_source_cases} "
            f"augmentations_per_case={len(augmentations)} augmented_cases={total_augmented_cases} "
            f"views={','.join(options.views)} samples={total_samples} "
            f"target_volume_mode={options.target_volume_mode}"
        ),
        flush=True,
    )
    case_manifests: dict[str, dict[str, Any]] = {}
    projection_cache: dict[tuple[str, str, float, int, int], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    done_augmented_cases = 0

    for split, records in records_by_split.items():
        for record in records:
            for augmentation in augmentations:
                key = augmented_case_key(record, augmentation, enabled=setup_enabled)
                try:
                    if key not in case_manifests:
                        case_manifests[key] = _case_payload(
                            record,
                            augmentation,
                            cache_root,
                            options.target_size,
                            options.dry_run,
                            cbct_normalization,
                            setup_enabled=setup_enabled,
                            target_volume_mode=options.target_volume_mode,
                        )
                    for view_key in options.views:
                        projection_payloads = []
                        for selected in _selected_for_augmentation(
                            record,
                            view_key,
                            augmentation,
                            enabled=setup_enabled,
                        ):
                            cache_key = (
                                key,
                                str(selected.file),
                                float(selected.target_angle_deg),
                                int(augmentation.translation_mm),
                                int(augmentation.rotation_deg),
                            )
                            if cache_key not in projection_cache:
                                payload, _, _ = _projection_payload(
                                    record,
                                    selected,
                                    augmentation,
                                    cache_root,
                                    options.projection_size,
                                    options.dry_run,
                                    projection_correction,
                                    setup_enabled=setup_enabled,
                                )
                                projection_cache[cache_key] = payload
                            projection_payloads.append(projection_cache[cache_key])
                        samples.append(
                            _sample_payload(
                                record,
                                augmentation,
                                split,
                                view_key,
                                cache_root,
                                case_manifests[key],
                                projection_payloads,
                                options.dry_run,
                                setup_enabled=setup_enabled,
                            )
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "split": split,
                            "patient_id": record.patient_id,
                            "case_id": record.case_id,
                            "augmentation_id": augmentation.id if setup_enabled else None,
                            "translation_mm": int(augmentation.translation_mm),
                            "rotation_deg": int(augmentation.rotation_deg),
                            "error": repr(exc),
                        }
                    )
                finally:
                    done_augmented_cases += 1
                    if progress_interval > 0 and (
                        done_augmented_cases == 1
                        or done_augmented_cases == total_augmented_cases
                        or done_augmented_cases % progress_interval == 0
                    ):
                        elapsed = time.perf_counter() - build_start_time
                        print(
                            (
                                f"[cache] build progress {done_augmented_cases}/{total_augmented_cases} "
                                f"augmented_cases split={split} patient={record.patient_id} case={record.case_id} "
                                f"augmentation={augmentation.id} samples={len(samples)}/{total_samples} "
                                f"failures={len(failures)} elapsed={elapsed:.1f}s"
                            ),
                            flush=True,
                        )

    view_counts = {
        view_key: {
            split: sum(1 for sample in samples if sample["view_key"] == view_key and sample["split"] == split)
            for split in ("train", "test")
        }
        for view_key in options.views
    }
    manifest = {
        "schema": "x2cbct_difnet_v0b_cache.v2",
        "geometry_branch": "real_projection",
        "axis_convention": axis_convention(),
        "cache_version": cache_root.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_command": options.command,
        "dry_run": options.dry_run,
        "raw_dataset_root": str(root / "dataset_list" / "self_projection1208"),
        "fixed_view_manifest": {"path": str(fixed.path), "sha256": sha256_file(fixed.path), "protocol": fixed.protocol},
        "split_manifest": split_manifest_payload,
        "calibration_report": {
            "path": str(calibration.path),
            "sha256": sha256_file(calibration.path),
            "status": calibration.status,
            "candidate_id": calibration.candidate_id,
            "adapter_version": calibration.adapter_version,
        },
        "views": {key: list(VIEW_TARGETS[key]) for key in options.views},
        "projection_resize": {"size": options.projection_size, "method": "cv2.INTER_AREA"},
        "target_resize": {"size": options.target_size, "method": "scipy.ndimage.zoom_order1"},
        "projection_normalization": {"rule": "clip(projection, 0, 8.6) / 8.6", "min": PROJECTION_MIN, "max": PROJECTION_MAX},
        "projection_correction": projection_correction,
        "target_volume_mode": options.target_volume_mode,
        "setup_augmentation": options.setup_augmentation.to_dict(),
        "cbct_normalization": cbct_normalization,
        "cbct_normalization_estimation": {
            "mode": options.hu_window_mode,
            "source_split": "train",
            "percentiles": list(options.hu_percentiles),
            "fixed_hu_window": [float(options.hu_fixed_min), float(options.hu_fixed_max)]
            if options.hu_window_mode == "fixed"
            else None,
            "sample_voxels_per_case": options.hu_sample_voxels_per_case,
            "case_stats": cbct_profile_case_stats,
        },
        "point_sampling": {
            "train": "official interleaved lattice point block sampling",
            "name": "official_interleaved_lattice",
            "fixed_training_points_saved": False,
            "grid_size": options.target_size,
            "stride": 4,
            "block_count": 64,
            "points_per_block": (int(options.target_size) // 4) ** 3,
        },
        "split": {
            **split_summary,
            "train_cases": len(records_by_split["train"]),
            "test_cases": len(records_by_split["test"]),
            "indexed_train_cases": len(indexed_records_by_split["train"]),
            "indexed_test_cases": len(indexed_records_by_split["test"]),
            "has_validation_split": False,
            "test_split_role": "test-as-validation during V0b development",
        },
        "source_case_counts": {
            "included": fixed.included_case_count,
            "excluded": fixed.excluded_case_count,
            "local_available_train": len(available_records_by_split["train"]),
            "local_available_test": len(available_records_by_split["test"]),
            "local_excluded": len(local_excluded_cases),
            "setup_augmentation_combinations_per_source_case": options.setup_augmentation.combination_count,
        },
        "local_dataset_filter": {
            "policy": "exclude_missing_required_case_files_before_cache_build",
            "views_checked": list(availability_views),
            "excluded_case_count": len(local_excluded_cases),
            "excluded_cases": local_excluded_cases,
        },
        "excluded_cases": fixed_raw.get("excluded_cases", []),
        "view_sample_counts": view_counts,
        "samples": samples,
        "failed_cases": failures,
    }
    write_json(manifest, cache_root / "cache_manifest.json")
    write_json(
        {
            "ok": not failures,
            "cache_root": str(cache_root),
            "view_sample_counts": view_counts,
            "failed_case_count": len(failures),
            "local_excluded_case_count": len(local_excluded_cases),
            "local_available_case_counts": {
                "train": len(available_records_by_split["train"]),
                "test": len(available_records_by_split["test"]),
            },
            "setup_augmentation": options.setup_augmentation.to_dict(),
        },
        cache_root / "reports" / "cache_summary.json",
    )
    write_json({"failed_cases": failures}, cache_root / "reports" / "failed_cases.json")
    if failures:
        raise RuntimeError(f"cache build completed with {len(failures)} failed cases; see reports/failed_cases.json")
    return manifest
