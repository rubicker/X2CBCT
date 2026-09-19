from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class VolumeRecord:
    case_id: str
    path: Path
    source_shape_zyx: tuple[int, int, int]
    source_spacing_xyz: tuple[float, float, float]
    source_origin_xyz: tuple[float, float, float]
    source_direction: tuple[float, ...]
    normalization: Mapping[str, Any]
    value_space: str
    augmentation_id: str | None = None
    reconstruction_metrics: Mapping[str, Any] | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str | Path, *, project_root: Path, manifest_path: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_candidate = (project_root / path).resolve()
    if project_candidate.exists() or manifest_path is None:
        return project_candidate
    return (manifest_path.parent / path).resolve()


def _normalization(row: Mapping[str, Any], manifest: Mapping[str, Any], fallback: Mapping[str, Any] | None) -> Mapping[str, Any]:
    value = row.get("normalization") or row.get("cbct_normalization")
    if not value:
        value = manifest.get("normalization") or manifest.get("cbct_normalization")
    return value or fallback or {}


def _geometry(
    row: Mapping[str, Any],
    *,
    fallback: VolumeRecord | None = None,
) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]:
    fallback_shape = fallback.source_shape_zyx if fallback else (128, 128, 128)
    fallback_spacing = fallback.source_spacing_xyz if fallback else (1.0, 1.0, 1.0)
    fallback_origin = fallback.source_origin_xyz if fallback else (0.0, 0.0, 0.0)
    fallback_direction = fallback.source_direction if fallback else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    shape = tuple(int(value) for value in (row.get("processed_shape_zyx") or fallback_shape))
    spacing = tuple(float(value) for value in (row.get("processed_spacing_xyz") or fallback_spacing))
    origin = tuple(float(value) for value in (row.get("processed_origin_xyz") or fallback_origin))
    direction = tuple(float(value) for value in (row.get("processed_direction") or fallback_direction))
    return shape, spacing, origin, direction


def load_reference_records(
    manifest_path: Path,
    *,
    project_root: Path,
    view_key: str,
    value_space: str = "normalized",
) -> dict[str, VolumeRecord]:
    manifest = read_json(manifest_path)
    rows = manifest.get("samples") or manifest.get("cases") or manifest.get("items")
    if not isinstance(rows, list):
        raise ValueError(f"reference manifest has no samples/cases/items list: {manifest_path}")
    records: dict[str, VolumeRecord] = {}
    for row in rows:
        if row.get("view_key") and str(row["view_key"]) != view_key:
            continue
        case_id = str(row.get("source_case_id") or row.get("case_id") or "")
        volume_value = row.get("processed_cbct_path") or row.get("target_cbct_path") or row.get("volume_path")
        if not case_id or not volume_value:
            continue
        shape, spacing, origin, direction = _geometry(row)
        record = VolumeRecord(
            case_id=case_id,
            path=resolve_project_path(volume_value, project_root=project_root, manifest_path=manifest_path),
            source_shape_zyx=shape,
            source_spacing_xyz=spacing,
            source_origin_xyz=origin,
            source_direction=direction,
            normalization=_normalization(row, manifest, None),
            value_space=value_space,
            augmentation_id=row.get("augmentation_id"),
        )
        previous = records.get(case_id)
        if previous is not None and previous.path != record.path:
            raise ValueError(f"duplicate reference case with different paths: {case_id}")
        records[case_id] = record
    if not records:
        raise ValueError(f"no reference records matched view_key={view_key}: {manifest_path}")
    return records


def load_prediction_records(
    manifest_path: Path,
    *,
    project_root: Path,
    reference_records: Mapping[str, VolumeRecord],
    augmentation_id: str | None = None,
    value_space: str = "normalized",
) -> dict[str, VolumeRecord]:
    manifest = read_json(manifest_path)
    rows = manifest.get("predictions") or manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"prediction manifest has no predictions/cases list: {manifest_path}")
    records: dict[str, VolumeRecord] = {}
    for row in rows:
        row_augmentation = row.get("augmentation_id")
        if augmentation_id is None:
            if row_augmentation not in (None, ""):
                continue
            case_id = str(row.get("case_id") or "")
        else:
            if str(row_augmentation or "") != augmentation_id:
                continue
            case_id = str(row.get("source_case_id") or row.get("case_id") or "")
        volume_value = row.get("generated_cbct_path") or row.get("prediction_path")
        if not case_id or not volume_value:
            continue
        fallback = reference_records.get(case_id)
        shape, spacing, origin, direction = _geometry(row, fallback=fallback)
        record = VolumeRecord(
            case_id=case_id,
            path=resolve_project_path(volume_value, project_root=project_root, manifest_path=manifest_path),
            source_shape_zyx=shape,
            source_spacing_xyz=spacing,
            source_origin_xyz=origin,
            source_direction=direction,
            normalization=_normalization(row, manifest, fallback.normalization if fallback else None),
            value_space=value_space,
            augmentation_id=str(row_augmentation) if row_augmentation else None,
            reconstruction_metrics=row.get("metrics") or {
                key: row[key] for key in ("psnr", "ssim", "mae_hu", "mae_norm") if key in row
            },
        )
        if case_id in records:
            raise ValueError(
                f"duplicate prediction case {case_id}; select one augmentation_id explicitly in {manifest_path}"
            )
        records[case_id] = record
    if not records:
        suffix = f", augmentation_id={augmentation_id}" if augmentation_id else " with unaugmented rows"
        raise ValueError(f"no prediction records found{suffix}: {manifest_path}")
    return records
