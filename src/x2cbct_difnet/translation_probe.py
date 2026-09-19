from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage
import SimpleITK as sitk


CBCT_AIR_HU = -1024.0


@dataclass(frozen=True)
class ProbeCase:
    patient_id: str
    case_id: str
    case_path: Path


@dataclass(frozen=True)
class ProjectionChoice:
    path: Path
    ordinal: int
    file_angle_hint_deg: float | None
    gantry_angle_deg: float | None
    detector_spacing_yx: tuple[float, float]
    detector_shape_yx: tuple[int, int]


def numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


def discover_probe_cases(root: Path, *, max_cases: int) -> list[ProbeCase]:
    cases: list[ProbeCase] = []
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != "back"], key=numeric_sort_key)
    for patient_dir in patient_dirs:
        case_dirs = sorted([p for p in patient_dir.iterdir() if p.is_dir()], key=numeric_sort_key)
        for case_dir in case_dirs:
            if (case_dir / "CBCT.nii.gz").exists() and (case_dir / "Geometry").exists():
                cases.append(ProbeCase(patient_dir.name, case_dir.name, case_dir))
                break
        if len(cases) >= max_cases:
            break
    return cases


def parse_projection_ordinal(path: Path) -> int | None:
    first = path.stem.split("_", 1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def parse_projection_angle_hint(path: Path) -> float | None:
    parts = path.stem.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def angular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def geometry_gantry_angles(geometry_path: Path) -> list[float]:
    root = ET.parse(geometry_path).getroot()
    out: list[float] = []
    for projection in root.findall("Projection"):
        text = projection.findtext("GantryAngle")
        out.append(float(text) if text is not None else float("nan"))
    return out


def choose_projection(case_path: Path, *, target_angle_deg: float = 290.0) -> ProjectionChoice:
    candidates = sorted(
        [p for p in case_path.glob("*.nii.gz") if p.name != "CBCT.nii.gz"],
        key=lambda p: (parse_projection_ordinal(p) is None, parse_projection_ordinal(p) or 10**9, p.name),
    )
    if not candidates:
        raise ValueError(f"no projection files under {case_path}")
    angles = geometry_gantry_angles(case_path / "Geometry")

    def score(path: Path) -> tuple[float, str]:
        ordinal = parse_projection_ordinal(path)
        gantry = angles[ordinal] if ordinal is not None and ordinal < len(angles) else None
        hint = parse_projection_angle_hint(path)
        angle = gantry if gantry is not None and math.isfinite(gantry) else hint
        if angle is None:
            return 10**9, path.name
        return angular_distance_deg(angle, target_angle_deg), path.name

    path = min(candidates, key=score)
    ordinal = parse_projection_ordinal(path)
    if ordinal is None:
        raise ValueError(f"projection file has no ordinal prefix: {path}")
    image = sitk.ReadImage(str(path))
    array = np.squeeze(sitk.GetArrayFromImage(image))
    spacing = image.GetSpacing()
    if len(spacing) < 2:
        raise ValueError(f"projection spacing has fewer than two axes: {path}")
    gantry = angles[ordinal] if ordinal < len(angles) and math.isfinite(angles[ordinal]) else None
    return ProjectionChoice(
        path=path,
        ordinal=ordinal,
        file_angle_hint_deg=parse_projection_angle_hint(path),
        gantry_angle_deg=gantry,
        detector_spacing_yx=(float(spacing[1]), float(spacing[0])),
        detector_shape_yx=(int(array.shape[-2]), int(array.shape[-1])),
    )


def detector_pixel_shift_for_si(
    shift_mm: float,
    *,
    sid_mm: float,
    sdd_mm: float,
    detector_spacing_y_mm: float,
) -> dict[str, float]:
    detector_shift_mm = -float(shift_mm) * float(sdd_mm) / float(sid_mm)
    raw_pixel_shift = detector_shift_mm / float(detector_spacing_y_mm)
    return {
        "si_shift_mm": float(shift_mm),
        "sid_mm": float(sid_mm),
        "sdd_mm": float(sdd_mm),
        "magnification": float(sdd_mm) / float(sid_mm),
        "detector_shift_mm": detector_shift_mm,
        "raw_pixel_shift_y": raw_pixel_shift,
    }


def edge_background_value(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("projection array must be 2D")
    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.nanmedian(edge))


def volume_background_value(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("volume array must be 3D")
    edge = np.concatenate(
        [
            arr[0, :, :].ravel(),
            arr[-1, :, :].ravel(),
            arr[:, 0, :].ravel(),
            arr[:, -1, :].ravel(),
            arr[:, :, 0].ravel(),
            arr[:, :, -1].ravel(),
        ]
    )
    return float(np.nanmedian(edge))


def shift_projection_vertical(array: np.ndarray, pixel_shift_y: float, *, fill_value: float) -> np.ndarray:
    shifted = scipy.ndimage.shift(
        np.asarray(array, dtype=np.float32),
        shift=(float(pixel_shift_y), 0.0),
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=False,
    )
    return shifted.astype(np.float32)


def shift_volume_head(array: np.ndarray, shift_mm: float, *, spacing_z_mm: float, fill_hu: float = CBCT_AIR_HU) -> np.ndarray:
    shifted = scipy.ndimage.shift(
        np.asarray(array, dtype=np.float32),
        shift=(float(shift_mm) / float(spacing_z_mm), 0.0, 0.0),
        order=1,
        mode="constant",
        cval=float(fill_hu),
        prefilter=False,
    )
    return shifted.astype(np.float32)


def read_rtk_sid_sdd(geometry_path: Path) -> tuple[float, float]:
    root = ET.parse(geometry_path).getroot()
    sid = root.findtext("SourceToIsocenterDistance")
    sdd = root.findtext("SourceToDetectorDistance")
    if sid is None or sdd is None:
        raise ValueError(f"Geometry missing SID/SDD: {geometry_path}")
    return float(sid), float(sdd)


def to_uint8_percentile(array: np.ndarray, *, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if high <= low + 1e-8:
        high = low + 1.0
    scaled = np.clip((np.nan_to_num(arr, nan=low) - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def percentile_window(array: np.ndarray, *, low_pct: float = 1.0, high_pct: float = 99.0) -> dict[str, float]:
    arr = np.asarray(array, dtype=np.float32)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return {"low": 0.0, "high": 1.0, "low_pct": low_pct, "high_pct": high_pct}
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if high <= low + 1e-8:
        high = low + 1.0
    return {"low": low, "high": high, "low_pct": low_pct, "high_pct": high_pct}


def to_uint8_window(array: np.ndarray, window: dict[str, float]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    low = float(window["low"])
    high = float(window["high"])
    scaled = np.clip((np.nan_to_num(arr, nan=low) - low) / max(high - low, 1e-8), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _labeled_image(array: np.ndarray, label: str, *, window: dict[str, float] | None = None) -> np.ndarray:
    img = to_uint8_window(array, window) if window is not None else to_uint8_percentile(array)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, label[:60], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def write_panel(path: Path, panels: list[tuple[str, np.ndarray, dict[str, float] | None]]) -> None:
    rendered = []
    for label, array, window in panels:
        rendered.append(_labeled_image(array, label, window=window))
    height = max(img.shape[0] for img in rendered)
    padded = []
    for img in rendered:
        if img.shape[0] < height:
            pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=0)
        padded.append(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(padded, axis=1))


def write_translation_probe_panel(
    path: Path,
    *,
    cbct_original: np.ndarray,
    cbct_shifted: np.ndarray,
    projection_original: np.ndarray,
    projection_shifted: np.ndarray,
    shift_mm: float,
) -> dict[str, dict[str, float]]:
    cbct_window = percentile_window(cbct_original)
    projection_window = percentile_window(projection_original)
    write_panel(
        path,
        [
            ("CBCT coronal original", cbct_original, cbct_window),
            (f"CBCT coronal {shift_mm:+g}mm", cbct_shifted, cbct_window),
            ("projection original", projection_original, projection_window),
            (f"projection {shift_mm:+g}mm", projection_shifted, projection_window),
            ("projection abs diff", np.abs(projection_shifted - projection_original), None),
        ],
    )
    return {
        "cbct_window": cbct_window,
        "projection_window": projection_window,
        "difference_window": {"mode": "independent_percentile"},
    }


def common_z_extent_for_shift(shape_z: int, shift_voxels_z: float) -> tuple[slice, slice]:
    offset = int(round(float(shift_voxels_z)))
    if abs(offset) >= int(shape_z):
        raise ValueError(f"shift too large for common z extent: shift={shift_voxels_z}, shape_z={shape_z}")
    if offset > 0:
        return slice(0, shape_z - offset), slice(offset, shape_z)
    if offset < 0:
        return slice(-offset, shape_z), slice(0, shape_z + offset)
    return slice(0, shape_z), slice(0, shape_z)


def write_cbct_common_fov_panel(
    path: Path,
    *,
    cbct_original: np.ndarray,
    cbct_shifted: np.ndarray,
    shift_voxels_z: float,
    shift_mm: float,
) -> dict[str, float]:
    original_slice, shifted_slice = common_z_extent_for_shift(cbct_original.shape[0], shift_voxels_z)
    original_common = cbct_original[original_slice, :]
    shifted_common = cbct_shifted[shifted_slice, :]
    cbct_window = percentile_window(original_common)
    write_panel(
        path,
        [
            ("CBCT common original", original_common, cbct_window),
            (f"CBCT common {shift_mm:+g}mm", shifted_common, cbct_window),
            ("CBCT common abs diff", np.abs(shifted_common - original_common), None),
        ],
    )
    return {
        "mean_abs_diff": float(np.mean(np.abs(shifted_common - original_common))),
        "max_abs_diff": float(np.max(np.abs(shifted_common - original_common))),
        "original_uint8_mean": float(np.mean(to_uint8_window(original_common, cbct_window))),
        "shifted_uint8_mean": float(np.mean(to_uint8_window(shifted_common, cbct_window))),
    }


def build_translation_probe(
    dataset_root: Path,
    output_dir: Path,
    *,
    shifts_mm: tuple[float, ...] = (8.0, 12.0),
    max_cases: int = 5,
    target_angle_deg: float = 290.0,
) -> dict[str, Any]:
    cases = discover_probe_cases(dataset_root, max_cases=max_cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        cbct_image = sitk.ReadImage(str(case.case_path / "CBCT.nii.gz"))
        cbct = sitk.GetArrayFromImage(cbct_image).astype(np.float32)
        spacing_xyz = tuple(float(v) for v in cbct_image.GetSpacing())
        projection = choose_projection(case.case_path, target_angle_deg=target_angle_deg)
        projection_array = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(projection.path)))).astype(np.float32)
        sid, sdd = read_rtk_sid_sdd(case.case_path / "Geometry")
        projection_background = edge_background_value(projection_array)
        cbct_background = volume_background_value(cbct)
        z_center = cbct.shape[0] // 2
        x_center = cbct.shape[2] // 2
        case_out = output_dir / f"patient_{case.patient_id}_case_{case.case_id}"
        case_out.mkdir(parents=True, exist_ok=True)
        shift_rows = []
        for shift_mm in shifts_mm:
            shift_meta = detector_pixel_shift_for_si(
                shift_mm,
                sid_mm=sid,
                sdd_mm=sdd,
                detector_spacing_y_mm=projection.detector_spacing_yx[0],
            )
            shifted_cbct = shift_volume_head(cbct, shift_mm, spacing_z_mm=spacing_xyz[2], fill_hu=cbct_background)
            shifted_projection = shift_projection_vertical(
                projection_array,
                shift_meta["raw_pixel_shift_y"],
                fill_value=projection_background,
            )
            shift_voxels_z = float(shift_mm) / float(spacing_xyz[2])
            panel_path = case_out / f"shift_{shift_mm:+g}mm_panel.png"
            common_fov_path = case_out / f"shift_{shift_mm:+g}mm_cbct_common_fov.png"
            cbct_original_coronal = cbct[:, cbct.shape[1] // 2, :]
            cbct_shifted_coronal = shifted_cbct[:, shifted_cbct.shape[1] // 2, :]
            display_windows = write_translation_probe_panel(
                panel_path,
                cbct_original=cbct_original_coronal,
                cbct_shifted=cbct_shifted_coronal,
                projection_original=projection_array,
                projection_shifted=shifted_projection,
                shift_mm=shift_mm,
            )
            cbct_common_fov_metrics = write_cbct_common_fov_panel(
                common_fov_path,
                cbct_original=cbct_original_coronal,
                cbct_shifted=cbct_shifted_coronal,
                shift_voxels_z=shift_voxels_z,
                shift_mm=shift_mm,
            )
            np.save(case_out / f"projection_shift_{shift_mm:+g}mm.npy", shifted_projection.astype(np.float32))
            shift_rows.append(
                {
                    **shift_meta,
                    "panel_png": str(panel_path),
                    "cbct_common_fov_png": str(common_fov_path),
                    "cbct_shift_voxels_z": shift_voxels_z,
                    "cbct_fill_value": cbct_background,
                    "cbct_common_fov_metrics": cbct_common_fov_metrics,
                    "projection_fill_value": projection_background,
                    "display_windows": display_windows,
                }
            )
        records.append(
            {
                "patient_id": case.patient_id,
                "case_id": case.case_id,
                "case_path": str(case.case_path),
                "cbct_path": str(case.case_path / "CBCT.nii.gz"),
                "cbct_shape_zyx": [int(v) for v in cbct.shape],
                "cbct_spacing_xyz": [float(v) for v in spacing_xyz],
                "direction_note": "positive shift is defined as movement toward head; implementation uses positive physical z / array z candidate for this probe",
                "projection": {
                    "path": str(projection.path),
                    "ordinal": projection.ordinal,
                    "file_angle_hint_deg": projection.file_angle_hint_deg,
                    "gantry_angle_deg": projection.gantry_angle_deg,
                    "detector_shape_yx": list(projection.detector_shape_yx),
                    "detector_spacing_yx": list(projection.detector_spacing_yx),
                },
                "sid_mm": sid,
                "sdd_mm": sdd,
                "center_slice_indices": {"z": z_center, "x": x_center},
                "shifts": shift_rows,
            }
        )
    report = {
        "schema": "x2cbct_v1_translation_range_probe.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "target_angle_deg": float(target_angle_deg),
        "shift_direction": "+mm means toward head",
        "case_count": len(records),
        "records": records,
    }
    report_path = output_dir / "translation_range_probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
