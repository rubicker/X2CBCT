from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - checked at runtime for CLI users
    sitk = None  # type: ignore[assignment]


CBCT_NAME = "CBCT.nii.gz"


@dataclass(frozen=True)
class VolumeHeader:
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


@dataclass(frozen=True)
class SetupCase:
    patient_id: str
    case_id: str
    case_path: str
    cbct_path: str
    header: VolumeHeader | None


@dataclass(frozen=True)
class SetupPair:
    patient_id: str
    reference_case_id: str
    moving_case_id: str
    reference_cbct_path: str
    moving_cbct_path: str


def numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


def read_volume_header(path: Path) -> VolumeHeader:
    if sitk is None:
        raise RuntimeError("SimpleITK is required to read CBCT headers")
    image = sitk.ReadImage(str(path))
    return VolumeHeader(
        size_xyz=tuple(int(item) for item in image.GetSize()),
        spacing_xyz=tuple(float(item) for item in image.GetSpacing()),
        origin_xyz=tuple(float(item) for item in image.GetOrigin()),
        direction=tuple(float(item) for item in image.GetDirection()),
    )


def iter_patient_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=numeric_sort_key)


def iter_case_dirs(patient_dir: Path) -> list[Path]:
    return sorted([p for p in patient_dir.iterdir() if p.is_dir()], key=numeric_sort_key)


def scan_setup_cases(root: Path, *, read_headers: bool = True) -> tuple[list[SetupCase], list[dict[str, str]]]:
    cases: list[SetupCase] = []
    missing: list[dict[str, str]] = []
    for patient_dir in iter_patient_dirs(root):
        for case_dir in iter_case_dirs(patient_dir):
            cbct_path = case_dir / CBCT_NAME
            if not cbct_path.exists():
                missing.append(
                    {
                        "patient_id": patient_dir.name,
                        "case_id": case_dir.name,
                        "case_path": str(case_dir),
                        "missing": CBCT_NAME,
                    }
                )
                continue
            header = read_volume_header(cbct_path) if read_headers else None
            cases.append(
                SetupCase(
                    patient_id=patient_dir.name,
                    case_id=case_dir.name,
                    case_path=str(case_dir),
                    cbct_path=str(cbct_path),
                    header=header,
                )
            )
    return cases, missing


def group_cases_by_patient(cases: Iterable[SetupCase]) -> dict[str, list[SetupCase]]:
    grouped: dict[str, list[SetupCase]] = {}
    for case in cases:
        grouped.setdefault(case.patient_id, []).append(case)
    for patient_cases in grouped.values():
        patient_cases.sort(key=lambda item: (int(item.case_id) if item.case_id.isdigit() else 10**9, item.case_id))
    return dict(sorted(grouped.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0])))


def build_setup_pairs(cases: Iterable[SetupCase]) -> list[SetupPair]:
    pairs: list[SetupPair] = []
    for patient_id, patient_cases in group_cases_by_patient(cases).items():
        if len(patient_cases) < 2:
            continue
        reference = patient_cases[0]
        for moving in patient_cases[1:]:
            pairs.append(
                SetupPair(
                    patient_id=patient_id,
                    reference_case_id=reference.case_id,
                    moving_case_id=moving.case_id,
                    reference_cbct_path=reference.cbct_path,
                    moving_cbct_path=moving.cbct_path,
                )
            )
    return pairs


def header_consistency(cases: Iterable[SetupCase]) -> dict[str, Any]:
    rows = [case for case in cases if case.header is not None]
    if not rows:
        return {"checked": 0, "all_consistent": None}
    first = rows[0].header
    assert first is not None
    fields = ("size_xyz", "spacing_xyz", "origin_xyz", "direction")
    result: dict[str, Any] = {"checked": len(rows), "all_consistent": True}
    for field in fields:
        values = [getattr(case.header, field) for case in rows if case.header is not None]
        unique = sorted({repr(value) for value in values})
        result[field] = {
            "unique_count": len(unique),
            "reference": getattr(first, field),
            "all_consistent": len(unique) == 1,
        }
        result["all_consistent"] = bool(result["all_consistent"] and len(unique) == 1)
    return result


def build_pair_manifest(root: Path, *, read_headers: bool = True) -> dict[str, Any]:
    cases, missing = scan_setup_cases(root, read_headers=read_headers)
    pairs = build_setup_pairs(cases)
    grouped = group_cases_by_patient(cases)
    return {
        "schema": "x2cbct_setup_error_pair_manifest.v1",
        "dataset_root": str(root),
        "patient_count": len(grouped),
        "case_count": len(cases),
        "pair_count": len(pairs),
        "missing_count": len(missing),
        "header_consistency": header_consistency(cases),
        "patients": [
            {
                "patient_id": patient_id,
                "case_count": len(patient_cases),
                "reference_case_id": patient_cases[0].case_id if patient_cases else None,
                "case_ids": [case.case_id for case in patient_cases],
            }
            for patient_id, patient_cases in grouped.items()
        ],
        "cases": [case_to_dict(case) for case in cases],
        "pairs": [asdict(pair) for pair in pairs],
        "missing": missing,
    }


def case_to_dict(case: SetupCase) -> dict[str, Any]:
    row = asdict(case)
    if case.header is not None:
        row["header"] = asdict(case.header)
    return row


def write_pair_manifest(manifest: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pair_manifest.json"
    csv_path = output_dir / "pair_manifest.csv"
    summary_path = output_dir / "pair_summary.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_pairs_csv(manifest["pairs"], csv_path)
    summary = {
        key: manifest[key]
        for key in ("schema", "dataset_root", "patient_count", "case_count", "pair_count", "missing_count")
    }
    summary["header_consistency"] = manifest["header_consistency"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "summary": str(summary_path)}


def write_pairs_csv(pairs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "patient_id",
        "reference_case_id",
        "moving_case_id",
        "reference_cbct_path",
        "moving_cbct_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({field: pair.get(field, "") for field in fields})


def read_pair_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
