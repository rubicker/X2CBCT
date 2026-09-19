from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fdk_baseline import _angles_rad, _real_scan_geometry, _simulated_drr_geometry, _tigre_geometry


SART_NORMALIZATION_SCHEMA = "x2cbct_sart_output_normalization.v1"
SART_NORMALIZATION_METHOD = "per_case_output_percentile_0p5_99p5"
DEFAULT_SART_ITERATIONS = 20
DEFAULT_SART_LAMBDA = 1.0
DEFAULT_SART_LAMBDA_RED = 0.999


@dataclass(frozen=True)
class SartReconstruction:
    volume: np.ndarray
    raw_volume: np.ndarray
    normalization: dict[str, Any]
    geometry: dict[str, Any]
    algorithm: dict[str, Any]


def normalize_sart_output(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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
        "schema": SART_NORMALIZATION_SCHEMA,
        "method": SART_NORMALIZATION_METHOD,
        "uses_target_volume": False,
        "low": low,
        "high": high,
        "raw_min": float(np.min(arr)) if arr.size else 0.0,
        "raw_max": float(np.max(arr)) if arr.size else 0.0,
        "raw_mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def reconstruct_sart_item(
    item: dict[str, Any],
    *,
    niter: int = DEFAULT_SART_ITERATIONS,
    lmbda: float = DEFAULT_SART_LAMBDA,
    lmbda_red: float = DEFAULT_SART_LAMBDA_RED,
) -> SartReconstruction:
    import tigre.algorithms as algs

    if niter <= 0:
        raise ValueError(f"SART niter must be positive, got {niter}")
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
        raise ValueError(f"SART projections must have shape [views, height, width], got {projections.shape}")

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
    angles_rad, angle_metadata = _angles_rad(
        cache_root,
        trace,
        np.asarray(item["view_angles_deg"], dtype=np.float32),
    )
    raw = algs.sart(
        projections[:, ::-1, :].copy(),
        geo,
        angles_rad,
        int(niter),
        lmbda=float(lmbda),
        lmbda_red=float(lmbda_red),
        verbose=False,
    ).astype(np.float32)
    raw_zyx = raw.transpose(2, 1, 0).astype(np.float32)
    normalized, normalization = normalize_sart_output(raw_zyx)
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
        "tigre_projection_flip": "vertical_before_sart",
        "tigre_output_transform": "transpose_xyz_to_zyx",
        **angle_metadata,
    }
    algorithm = {
        "method": "SART",
        "implementation": "TIGRE",
        "niter": int(niter),
        "lmbda": float(lmbda),
        "lmbda_red": float(lmbda_red),
    }
    return SartReconstruction(
        volume=normalized,
        raw_volume=raw_zyx,
        normalization=normalization,
        geometry=geometry,
        algorithm=algorithm,
    )
