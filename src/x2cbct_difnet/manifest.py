from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SelectedProjection:
    target_angle_deg: float
    ordinal: int
    file: str
    gantry_angle_deg: float
    angular_residual_deg: float


@dataclass(frozen=True)
class FixedViewRecord:
    patient_id: str
    case_id: str
    case_path: Path
    selected: dict[str, list[SelectedProjection]]

    @property
    def key(self) -> tuple[str, str]:
        return self.patient_id, self.case_id


@dataclass(frozen=True)
class FixedViewManifest:
    path: Path
    protocol: str
    included_case_count: int
    excluded_case_count: int
    records: list[FixedViewRecord]

    def by_case(self) -> dict[tuple[str, str], FixedViewRecord]:
        return {record.key: record for record in self.records}


@dataclass(frozen=True)
class SplitGroup:
    patient_count: int
    case_count: int
    patients: tuple[str, ...]
    cases: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    protocol: str
    train: SplitGroup
    test: SplitGroup

    @property
    def patient_overlap(self) -> set[str]:
        return set(self.train.patients).intersection(self.test.patients)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _selected_projection(raw: dict[str, Any]) -> SelectedProjection:
    return SelectedProjection(
        target_angle_deg=float(raw["target_angle_deg"]),
        ordinal=int(raw["ordinal"]),
        file=str(raw["file"]),
        gantry_angle_deg=float(raw["gantry_angle_deg"]),
        angular_residual_deg=float(raw["angular_residual_deg"]),
    )


def load_fixed_view_manifest(project_root: str | Path) -> FixedViewManifest:
    root = Path(project_root).resolve()
    path = root / "data_manifest" / "fixed_view_sampling_v0b.json"
    raw = _read_json(path)
    records = []
    for item in raw["records"]:
        selected = {
            str(view_key): [_selected_projection(proj) for proj in projections]
            for view_key, projections in item["selected"].items()
        }
        records.append(
            FixedViewRecord(
                patient_id=str(item["patient_id"]),
                case_id=str(item["case_id"]),
                case_path=Path(item["case_path"]),
                selected=selected,
            )
        )
    return FixedViewManifest(
        path=path,
        protocol=str(raw["protocol"]),
        included_case_count=int(raw["included_case_count"]),
        excluded_case_count=int(raw["excluded_case_count"]),
        records=records,
    )


def _split_group(raw: dict[str, Any]) -> SplitGroup:
    return SplitGroup(
        patient_count=int(raw["patient_count"]),
        case_count=int(raw["case_count"]),
        patients=tuple(str(item) for item in raw["patients"]),
        cases=tuple(dict(item) for item in raw["cases"]),
    )


def load_split_manifest(project_root: str | Path, fail_on_overlap: bool = True) -> SplitManifest:
    root = Path(project_root).resolve()
    path = root / "data_manifest" / "splits_v0b_train_test.json"
    raw = _read_json(path)
    manifest = SplitManifest(
        path=path,
        protocol=str(raw["protocol"]),
        train=_split_group(raw["train"]),
        test=_split_group(raw["test"]),
    )
    if fail_on_overlap and manifest.patient_overlap:
        overlap = ", ".join(sorted(manifest.patient_overlap))
        raise ValueError(f"patient-level split overlap detected: {overlap}")
    return manifest


def records_for_split(project_root: str | Path, split: str) -> list[FixedViewRecord]:
    fixed = load_fixed_view_manifest(project_root)
    split_manifest = load_split_manifest(project_root)
    group = split_manifest.train if split == "train" else split_manifest.test
    indexed = fixed.by_case()
    records: list[FixedViewRecord] = []
    for case in group.cases:
        key = (str(case["patient_id"]), str(case["case_id"]))
        if key not in indexed:
            raise KeyError(f"case {key} is missing from fixed-view manifest")
        records.append(indexed[key])
    return records
