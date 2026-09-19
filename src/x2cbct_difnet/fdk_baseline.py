from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import read_json, resolve_cache_path
from .calibration import load_calibration_report
from .geometry import RtkScanGeometry, read_rtk_scan_geometry


FDK_NORMALIZATION_SCHEMA = "x2cbct_fdk_output_normalization.v1"
FDK_NORMALIZATION_METHOD = "per_case_output_percentile_0p5_99p5"


@dataclass(frozen=True)
class FdkReconstruction:
    volume: np.ndarray
    raw_volume: np.ndarray
    normalization: dict[str, Any]
    geometry: dict[str, Any]


def normalize_fdk_output(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        low = 0.0
        high = 1.0
    else:
        low = float(np.percentile(finite, 0.5))
        high = float(np.percentile(finite, 99.5))
        if high <= low + 1e-8:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low + 1e-8:
            high = low + 1.0
    norm = np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return norm, {
        "schema": FDK_NORMALIZATION_SCHEMA,
        "method": FDK_NORMALIZATION_METHOD,
        "uses_target_volume": False,
        "low": low,
        "high": high,
        "raw_min": float(np.min(arr)) if arr.size else 0.0,
        "raw_max": float(np.max(arr)) if arr.size else 0.0,
        "raw_mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def _project_root_for_cache(cache_root: Path) -> Path:
    calibration_rel = Path("runs") / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"
    for candidate in (cache_root, *cache_root.parents):
        if (candidate / calibration_rel).exists():
            return candidate
        if candidate.name == "data_cache":
            return candidate.parent
    return Path.cwd().resolve()


def _read_sample_npz(sample_path: Path) -> dict[str, Any]:
    with np.load(sample_path, allow_pickle=False) as raw:
        return {
            "case_manifest_path": str(np.asarray(raw["case_manifest_path"]).item()),
        }


def _real_scan_geometry(cache_root: Path, trace: dict[str, Any]) -> RtkScanGeometry:
    sample = _read_sample_npz(Path(trace["sample_path"]))
    case_manifest = read_json(resolve_cache_path(cache_root, sample["case_manifest_path"]))
    projection_geometry = case_manifest.get("projection_geometry")
    if isinstance(projection_geometry, dict):
        sid = projection_geometry.get("source_to_isocenter_distance")
        sdd = projection_geometry.get("source_to_detector_distance")
        if sid is not None and sdd is not None:
            return RtkScanGeometry(float(sid), float(sdd))
    summary_path = case_manifest.get("geometry_summary_path")
    if summary_path:
        summary = read_json(resolve_cache_path(cache_root, str(summary_path)))
        return RtkScanGeometry(
            float(summary["source_to_isocenter_distance"]),
            float(summary["source_to_detector_distance"]),
        )
    return read_rtk_scan_geometry(resolve_cache_path(cache_root, str(case_manifest["raw_case_path"])) / "Geometry")


def _simulated_drr_geometry(trace: dict[str, Any]) -> RtkScanGeometry:
    config = trace.get("drr_config") or {}
    return RtkScanGeometry(
        source_to_isocenter_distance=float(config.get("dso_mm", 1000.0)),
        source_to_detector_distance=float(config.get("dsd_mm", 2000.0)),
    )


def _angles_rad(cache_root: Path, trace: dict[str, Any], angles_deg: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    geometry_branch = str(trace.get("geometry_branch", "real_projection"))
    if geometry_branch == "real_projection":
        calibration = load_calibration_report(_project_root_for_cache(cache_root), require_pass=True)
        candidate = calibration.candidate
        calibrated_deg = candidate.gantry_angle_sign * angles_deg.astype(np.float32) + float(
            candidate.gantry_angle_offset_deg
        )
        return np.deg2rad(calibrated_deg).astype(np.float32), {
            "angle_source": "gantry_angles_deg",
            "calibration_candidate_id": calibration.candidate_id,
            "adapter_version": calibration.adapter_version,
            "gantry_angle_sign": candidate.gantry_angle_sign,
            "gantry_angle_offset_deg": candidate.gantry_angle_offset_deg,
        }
    return np.deg2rad(angles_deg.astype(np.float32)).astype(np.float32), {
        "angle_source": "drr_angles_deg",
        "gantry_angle_sign": 1,
        "gantry_angle_offset_deg": 0.0,
    }


def _tigre_geometry(
    *,
    scan_geometry: RtkScanGeometry,
    detector_shape: tuple[int, int],
    detector_spacing_yx: tuple[float, float],
    processed_shape_zyx: tuple[int, int, int],
    processed_spacing_xyz: tuple[float, float, float],
) -> Any:
    from tigre.utilities.geometry import Geometry

    z_size, y_size, x_size = processed_shape_zyx
    spacing_x, spacing_y, spacing_z = processed_spacing_xyz
    detector_h, detector_w = detector_shape
    spacing_det_y, spacing_det_x = detector_spacing_yx

    geo = Geometry()
    geo.mode = "cone"
    geo.DSD = float(scan_geometry.source_to_detector_distance)
    geo.DSO = float(scan_geometry.source_to_isocenter_distance)
    geo.nDetector = np.array([detector_h, detector_w], dtype=np.int32)
    geo.dDetector = np.array([spacing_det_y, spacing_det_x], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([x_size, y_size, z_size], dtype=np.int32)
    geo.dVoxel = np.array([spacing_x, spacing_y, spacing_z], dtype=np.float32)
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.rotDetector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.accuracy = 0.5
    return geo


def reconstruct_fdk_item(item: dict[str, Any]) -> FdkReconstruction:
    import tigre.algorithms as algs

    trace = item["trace"]
    cache_root = Path(trace["cache_root"]).resolve()
    geometry_branch = str(trace.get("geometry_branch", "real_projection"))
    scan_geometry = (
        _simulated_drr_geometry(trace)
        if geometry_branch == "simulated_drr"
        else _real_scan_geometry(cache_root, trace)
    )
    projections = np.asarray(item["projs"], dtype=np.float32)
    if projections.ndim == 4:
        projections = projections[:, 0, :, :]
    if projections.ndim != 3:
        raise ValueError(f"FDK projections must have shape [views, height, width], got {projections.shape}")
    raw_detector_shape = tuple(int(value) for value in trace["detector_shape"])
    raw_detector_spacing_yx = tuple(float(value) for value in trace["detector_spacing_yx"])
    detector_shape = (int(projections.shape[1]), int(projections.shape[2]))
    detector_spacing_yx = (
        raw_detector_spacing_yx[0] * raw_detector_shape[0] / detector_shape[0],
        raw_detector_spacing_yx[1] * raw_detector_shape[1] / detector_shape[1],
    )
    processed_shape_zyx = tuple(int(value) for value in trace["processed_shape_zyx"])
    processed_spacing_xyz = tuple(float(value) for value in trace["processed_spacing_xyz"])
    geo = _tigre_geometry(
        scan_geometry=scan_geometry,
        detector_shape=detector_shape,
        detector_spacing_yx=detector_spacing_yx,
        processed_shape_zyx=processed_shape_zyx,
        processed_spacing_xyz=processed_spacing_xyz,
    )
    angles_deg = np.asarray(item["view_angles_deg"], dtype=np.float32)
    angles_rad, angle_metadata = _angles_rad(cache_root, trace, angles_deg)
    raw = algs.fdk(projections[:, ::-1, :].copy(), geo, angles_rad).astype(np.float32)
    raw_zyx = raw.transpose(2, 1, 0).astype(np.float32)
    normalized, normalization = normalize_fdk_output(raw_zyx)
    geometry = {
        "geometry_branch": geometry_branch,
        "source_to_isocenter_distance": scan_geometry.source_to_isocenter_distance,
        "source_to_detector_distance": scan_geometry.source_to_detector_distance,
        "detector_shape": list(detector_shape),
        "detector_spacing_yx": list(detector_spacing_yx),
        "raw_detector_shape": list(raw_detector_shape),
        "raw_detector_spacing_yx": list(raw_detector_spacing_yx),
        "processed_shape_zyx": list(processed_shape_zyx),
        "processed_spacing_xyz": list(processed_spacing_xyz),
        "tigre_projection_flip": "vertical_before_fdk",
        "tigre_output_transform": "transpose_xyz_to_zyx",
        **angle_metadata,
    }
    return FdkReconstruction(
        volume=normalized,
        raw_volume=raw_zyx,
        normalization=normalization,
        geometry=geometry,
    )
