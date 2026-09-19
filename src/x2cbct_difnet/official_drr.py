from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import scipy.ndimage

from .normalization import normalize_projection


OFFICIAL_DIFNET_PROJECTION_NORM = 0.10
C2RV_PROJECTION_LOADER_DENOMINATOR = 0.20


@dataclass(frozen=True)
class OfficialDrrConfig:
    projection_count: int = 50
    total_angle_deg: float = 180.0
    start_angle_deg: float = 0.0
    detector_size: int = 256
    voxel_size: int = 256
    dsd_mm: float = 2000.0
    dso_mm: float = 1000.0
    detector_spacing_mm: float = 2.0
    voxel_spacing_mm: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_count": self.projection_count,
            "total_angle_deg": self.total_angle_deg,
            "start_angle_deg": self.start_angle_deg,
            "detector_size": self.detector_size,
            "voxel_size": self.voxel_size,
            "dsd_mm": self.dsd_mm,
            "dso_mm": self.dso_mm,
            "detector_spacing_mm": self.detector_spacing_mm,
            "voxel_spacing_mm": self.voxel_spacing_mm,
        }


def official_angles_deg(config: OfficialDrrConfig = OfficialDrrConfig()) -> np.ndarray:
    return np.linspace(
        config.start_angle_deg,
        config.start_angle_deg + config.total_angle_deg,
        config.projection_count + 1,
        dtype=np.float32,
    )[:-1]


def official_angles_rad(config: OfficialDrrConfig = OfficialDrrConfig()) -> np.ndarray:
    return official_angles_deg(config) / 180.0 * math.pi


def generate_drr_with_tigre(volume: np.ndarray, config: OfficialDrrConfig = OfficialDrrConfig()) -> np.ndarray:
    try:
        import tigre
        from tigre.utilities.geometry import Geometry
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package
        raise RuntimeError("TIGRE is required for official DRR generation") from exc

    geo = Geometry()
    geo.DSD = config.dsd_mm / 1000.0
    geo.DSO = config.dso_mm / 1000.0
    geo.nDetector = np.array([config.detector_size, config.detector_size])
    geo.dDetector = np.array([config.detector_spacing_mm, config.detector_spacing_mm]) / 1000.0
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([config.voxel_size, config.voxel_size, config.voxel_size])
    geo.dVoxel = np.array([config.voxel_spacing_mm, config.voxel_spacing_mm, config.voxel_spacing_mm]) / 1000.0
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array([0.0, 0.0, 0.0])
    geo.offDetector = np.array([0.0, 0.0, 0.0])
    geo.accuracy = 0.5
    geo.mode = "cone"
    geo.filter = None
    arr = np.asarray(volume, dtype=np.float32)
    projections = tigre.Ax(arr.transpose(2, 1, 0).copy(), geo, official_angles_rad(config))[:, ::-1, :]
    return projections.astype(np.float32)


def generate_drr_with_fallback_ray_sum(volume: np.ndarray, config: OfficialDrrConfig = OfficialDrrConfig()) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    projections = []
    for angle in official_angles_deg(config):
        rotated = scipy.ndimage.rotate(arr, float(angle), axes=(2, 1), reshape=False, order=1, prefilter=False)
        projection = rotated.sum(axis=0)
        projection = cv2.resize(projection.astype(np.float32), (config.detector_size, config.detector_size), interpolation=cv2.INTER_AREA)
        projections.append(projection)
    return np.stack(projections, axis=0).astype(np.float32)


def normalize_official_difnet_projection(
    projections: np.ndarray,
    projection_norm: float = OFFICIAL_DIFNET_PROJECTION_NORM,
) -> np.ndarray:
    scaled = np.asarray(projections, dtype=np.float32) / float(projection_norm)
    scaled = scaled * 255.0
    scaled = np.clip(scaled, 0.0, 255.0)
    return scaled.astype(np.uint8)


def normalize_c2rv_projection(
    projections: np.ndarray,
    *,
    loader_denominator: float = C2RV_PROJECTION_LOADER_DENOMINATOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(projections, dtype=np.float32)
    projection_max = float(np.max(arr)) if arr.size else 0.0
    if projection_max <= 0.0:
        normalized = np.zeros_like(arr, dtype=np.uint8)
        loader_scale = 1.0
    else:
        scaled = arr / projection_max * 255.0
        scaled = np.clip(scaled, 0.0, 255.0)
        normalized = scaled.astype(np.uint8)
        loader_scale = 255.0 * float(loader_denominator) / projection_max
    return normalized, {
        "projection_normalization": "c2rv_per_case_max_uint8",
        "projection_max": projection_max,
        "projection_loader_denominator": float(loader_denominator),
        "projection_loader_scale": float(loader_scale),
    }


def generate_raw_drr(
    volume: np.ndarray,
    *,
    projector: str = "tigre",
    config: OfficialDrrConfig = OfficialDrrConfig(),
) -> np.ndarray:
    if projector == "tigre":
        return generate_drr_with_tigre(volume, config)
    elif projector == "fallback-ray-sum":
        return generate_drr_with_fallback_ray_sum(volume, config)
    else:
        raise ValueError("projector must be tigre or fallback-ray-sum")


def generate_normalized_drr(
    volume: np.ndarray,
    *,
    projector: str = "tigre",
    config: OfficialDrrConfig = OfficialDrrConfig(),
) -> np.ndarray:
    projections = generate_raw_drr(volume, projector=projector, config=config)
    if projector == "tigre":
        return normalize_official_difnet_projection(projections)
    return normalize_projection(projections).astype(np.float16)
