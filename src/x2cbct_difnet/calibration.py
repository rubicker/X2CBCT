from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationCandidate:
    candidate_id: str
    axis_order: tuple[int, int, int]
    axis_signs: tuple[int, int, int]
    detector_axis_order: tuple[int, int]
    detector_axis_signs: tuple[int, int]
    detector_center: str
    projection_model: str
    gantry_angle_sign: int
    gantry_angle_offset_deg: float


@dataclass(frozen=True)
class CalibrationReport:
    path: Path
    schema: str
    status: str
    adapter_version: str
    candidate_id: str
    candidate: CalibrationCandidate
    reviewed_projection_count: int

    @property
    def is_pass(self) -> bool:
        return self.status == "pass"


def _candidate(raw: dict[str, Any]) -> CalibrationCandidate:
    return CalibrationCandidate(
        candidate_id=str(raw["candidate_id"]),
        axis_order=tuple(int(item) for item in raw["axis_order"]),  # type: ignore[arg-type]
        axis_signs=tuple(int(item) for item in raw["axis_signs"]),  # type: ignore[arg-type]
        detector_axis_order=tuple(int(item) for item in raw["detector_axis_order"]),  # type: ignore[arg-type]
        detector_axis_signs=tuple(int(item) for item in raw["detector_axis_signs"]),  # type: ignore[arg-type]
        detector_center=str(raw["detector_center"]),
        projection_model=str(raw["projection_model"]),
        gantry_angle_sign=int(raw["gantry_angle_sign"]),
        gantry_angle_offset_deg=float(raw["gantry_angle_offset_deg"]),
    )


def load_calibration_report(project_root: str | Path, require_pass: bool = True) -> CalibrationReport:
    root = Path(project_root).resolve()
    path = root / "runs" / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    report = CalibrationReport(
        path=path,
        schema=str(raw["schema"]),
        status=str(raw["status"]),
        adapter_version=str(raw["adapter_version"]),
        candidate_id=str(raw["candidate_id"]),
        candidate=_candidate(raw["candidate"]),
        reviewed_projection_count=int(raw.get("reviewed_projection_count", 0)),
    )
    if require_pass and not report.is_pass:
        raise ValueError(f"calibration report is not pass: {path}")
    return report
