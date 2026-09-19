from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import CalibrationReport


@dataclass(frozen=True)
class RtkScanGeometry:
    source_to_isocenter_distance: float
    source_to_detector_distance: float


@dataclass(frozen=True)
class ProjectionResult:
    grid: np.ndarray
    detector_pixel: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VolumeGeometry:
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: np.ndarray
    shape_zyx: tuple[int, int, int]

    def physical_points_from_indices(self, indices_zyx: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices_zyx, dtype=np.float64)
        index_xyz = indices[:, [2, 1, 0]]
        scaled = index_xyz * np.asarray(self.spacing_xyz, dtype=np.float64)
        physical = np.asarray(self.origin_xyz, dtype=np.float64) + scaled @ self.direction.T
        return physical.astype(np.float32)

    def model_points_from_indices(self, indices_zyx: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices_zyx, dtype=np.float32)
        z_size, y_size, x_size = self.shape_zyx
        x = indices[:, 2] / max(x_size - 1, 1)
        y = indices[:, 1] / max(y_size - 1, 1)
        z = indices[:, 0] / max(z_size - 1, 1)
        points = np.stack([x, y, z], axis=1)
        return (points * 2.0 - 1.0).astype(np.float32)


def read_rtk_scan_geometry(geometry_path: str | Path) -> RtkScanGeometry:
    root = ET.parse(geometry_path).getroot()
    sid = root.findtext("SourceToIsocenterDistance")
    sdd = root.findtext("SourceToDetectorDistance")
    if sid is None or sdd is None:
        raise ValueError(f"RTK geometry missing SID/SDD: {geometry_path}")
    return RtkScanGeometry(
        source_to_isocenter_distance=float(sid),
        source_to_detector_distance=float(sdd),
    )


class ClinicalGeometryAdapter:
    def __init__(self, calibration: CalibrationReport, scan_geometry: RtkScanGeometry | None = None) -> None:
        if not calibration.is_pass:
            raise ValueError(f"calibration report is not pass: {calibration.path}")
        candidate = calibration.candidate
        if candidate.projection_model != "gantry_angle_xy":
            raise ValueError(f"unsupported projection model: {candidate.projection_model}")
        self.calibration = calibration
        self.scan_geometry = scan_geometry or RtkScanGeometry(1000.0, 1536.0)

    def project_points_to_grid(
        self,
        points_xyz_mm: np.ndarray,
        gantry_angle_deg: float,
        detector_shape: tuple[int, int],
        detector_spacing: tuple[float, float],
        feature_shape: tuple[int, int] | None = None,
    ) -> ProjectionResult:
        points = np.asarray(points_xyz_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz_mm must have shape [N, 3]")

        candidate = self.calibration.candidate
        angle = math.radians(
            candidate.gantry_angle_sign * float(gantry_angle_deg) + candidate.gantry_angle_offset_deg
        )
        sad = self.scan_geometry.source_to_isocenter_distance
        sdd = self.scan_geometry.source_to_detector_distance

        source = np.array([sad * math.cos(angle), sad * math.sin(angle), 0.0], dtype=np.float64)
        central_ray = -source / np.linalg.norm(source)
        detector_center = source + central_ray * sdd
        u_base = np.array([-math.sin(angle), math.cos(angle), 0.0], dtype=np.float64)
        v_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        sign_u, sign_v = candidate.detector_axis_signs
        u_axis = sign_u * u_base
        v_axis = sign_v * v_base

        rays = points - source[None, :]
        denom = rays @ central_ray
        safe_denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
        t = sdd / safe_denom
        hits = source[None, :] + rays * t[:, None]
        rel = hits - detector_center[None, :]
        detector_mm = np.stack([rel @ u_axis, rel @ v_axis], axis=1)

        detector_h, detector_w = detector_shape
        spacing_y, spacing_x = detector_spacing
        center_x = (detector_w - 1) / 2.0
        center_y = (detector_h - 1) / 2.0
        pixel_x = detector_mm[:, 0] / spacing_x + center_x
        pixel_y = detector_mm[:, 1] / spacing_y + center_y
        detector_pixel = np.stack([pixel_x, pixel_y], axis=1)

        feature_h, feature_w = feature_shape or detector_shape
        if feature_shape is not None:
            pixel_x = pixel_x * ((feature_w - 1) / max(detector_w - 1, 1))
            pixel_y = pixel_y * ((feature_h - 1) / max(detector_h - 1, 1))

        grid_x = 2.0 * pixel_x / max(feature_w - 1, 1) - 1.0
        grid_y = 2.0 * pixel_y / max(feature_h - 1, 1) - 1.0
        grid = np.stack([grid_x, grid_y], axis=1).astype(np.float32)
        valid = np.isfinite(grid).all(axis=1)
        valid &= (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0)
        valid &= (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)

        return ProjectionResult(
            grid=grid,
            detector_pixel=detector_pixel.astype(np.float32),
            valid_mask=valid,
            metadata={
                "candidate_id": self.calibration.candidate_id,
                "adapter_version": self.calibration.adapter_version,
                "gantry_angle_deg": float(gantry_angle_deg),
                "detector_shape": list(detector_shape),
                "feature_shape": list(feature_shape or detector_shape),
            },
        )

    def project_many_views(
        self,
        points_xyz_mm: np.ndarray,
        gantry_angles_deg: list[float],
        detector_shape: tuple[int, int],
        detector_spacing: tuple[float, float],
        feature_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        grids = []
        masks = []
        for angle in gantry_angles_deg:
            result = self.project_points_to_grid(
                points_xyz_mm,
                angle,
                detector_shape=detector_shape,
                detector_spacing=detector_spacing,
                feature_shape=feature_shape,
            )
            grids.append(result.grid)
            masks.append(result.valid_mask)
        return np.stack(grids, axis=0), np.stack(masks, axis=0)
