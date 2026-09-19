from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from .calibration import load_calibration_report
from .geometry import ClinicalGeometryAdapter, VolumeGeometry, read_rtk_scan_geometry
from .manifest import FixedViewRecord, records_for_split
from .normalization import SUPPORTED_VIEW_KEYS, normalize_cbct_hu, normalize_projection
from .point_blocks import OFFICIAL_GRID_SIZE, sample_official_lattice_indices


@dataclass(frozen=True)
class NormalizationStats:
    minimum: float
    maximum: float

    def normalize(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        scale = self.maximum - self.minimum
        if abs(scale) < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - self.minimum) / scale).astype(np.float32)


def _stats(array: np.ndarray) -> NormalizationStats:
    arr = np.asarray(array, dtype=np.float32)
    return NormalizationStats(float(np.nanmin(arr)), float(np.nanmax(arr)))


def _read_image(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return image, np.asarray(array)


def _volume_geometry(image: sitk.Image, array: np.ndarray) -> VolumeGeometry:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    return VolumeGeometry(
        spacing_xyz=tuple(float(item) for item in image.GetSpacing()),
        origin_xyz=tuple(float(item) for item in image.GetOrigin()),
        direction=direction,
        shape_zyx=tuple(int(item) for item in array.shape),
    )


def _projection_spacing_yx(image: sitk.Image) -> tuple[float, float]:
    spacing = image.GetSpacing()
    if len(spacing) < 2:
        raise ValueError("projection image must have at least two spacing dimensions")
    return float(spacing[1]), float(spacing[0])


def _resize_projection(array: np.ndarray, size: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(array, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"projection must be 2D after squeeze, got shape {arr.shape}")
    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def _resize_volume_to_cube(array: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if tuple(arr.shape) == (size, size, size):
        return arr
    factors = [size / dim for dim in arr.shape]
    return scipy.ndimage.zoom(arr, factors, order=1, prefilter=False).astype(np.float32)


def _resized_volume_geometry(image: sitk.Image, raw_shape_zyx: tuple[int, int, int], target_size: int) -> VolumeGeometry:
    spacing = tuple(float(item) for item in image.GetSpacing())
    z_size, y_size, x_size = raw_shape_zyx
    spacing_xyz = (
        spacing[0] * max(x_size - 1, 1) / max(target_size - 1, 1),
        spacing[1] * max(y_size - 1, 1) / max(target_size - 1, 1),
        spacing[2] * max(z_size - 1, 1) / max(target_size - 1, 1),
    )
    return VolumeGeometry(
        spacing_xyz=spacing_xyz,
        origin_xyz=tuple(float(item) for item in image.GetOrigin()),
        direction=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
        shape_zyx=(target_size, target_size, target_size),
    )


class DifNetV0bDataset:
    """Minimal local dataset adapter that returns official DIF-Net-style items."""

    def __init__(
        self,
        project_root: str | Path,
        split: str,
        view_key: str,
        num_points: int = 4096,
        projection_size: int = OFFICIAL_GRID_SIZE,
        seed: int = 20260522,
        max_cases: int | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if view_key not in SUPPORTED_VIEW_KEYS:
            raise ValueError(f"view_key must be one of {', '.join(SUPPORTED_VIEW_KEYS)}")
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.view_key = view_key
        self.num_points = int(num_points)
        self.projection_size = int(projection_size)
        self.seed = int(seed)
        self.records = records_for_split(self.project_root, split)
        if max_cases is not None:
            self.records = self.records[: int(max_cases)]
        calibration = load_calibration_report(self.project_root, require_pass=True)
        self.geometry_adapter = ClinicalGeometryAdapter(calibration)

    def __len__(self) -> int:
        return len(self.records)

    def _load_projections(self, record: FixedViewRecord) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
        selected = record.selected[self.view_key]
        projections = []
        detector_shape = None
        detector_spacing = None
        for item in selected:
            image, array = _read_image(record.case_path / item.file)
            squeezed = np.squeeze(array)
            detector_shape = tuple(int(value) for value in squeezed.shape[-2:])
            detector_spacing = _projection_spacing_yx(image)
            resized = _resize_projection(squeezed, self.projection_size)
            projections.append(normalize_projection(resized)[None, ...])
        assert detector_shape is not None
        assert detector_spacing is not None
        return np.stack(projections, axis=0).astype(np.float32), detector_shape, detector_spacing

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = np.random.default_rng(self.seed + index)
        projs, detector_shape, detector_spacing = self._load_projections(record)

        cbct_image, cbct = _read_image(record.case_path / "CBCT.nii.gz")
        raw_shape_zyx = tuple(int(item) for item in cbct.shape)
        cbct = _resize_volume_to_cube(cbct, OFFICIAL_GRID_SIZE)
        volume_geometry = _resized_volume_geometry(cbct_image, raw_shape_zyx, OFFICIAL_GRID_SIZE)
        indices, sampling_trace = sample_official_lattice_indices(self.num_points, rng)
        values = cbct[indices[:, 0], indices[:, 1], indices[:, 2]]
        p_gt = normalize_cbct_hu(values)[None, :]
        points = volume_geometry.model_points_from_indices(indices)
        physical_points = volume_geometry.physical_points_from_indices(indices)

        scan_geometry = read_rtk_scan_geometry(record.case_path / "Geometry")
        adapter = ClinicalGeometryAdapter(self.geometry_adapter.calibration, scan_geometry)
        selected = record.selected[self.view_key]
        angles = [item.gantry_angle_deg for item in selected]
        proj_points, valid_mask = adapter.project_many_views(
            physical_points,
            angles,
            detector_shape=detector_shape,
            detector_spacing=detector_spacing,
            feature_shape=(self.projection_size, self.projection_size),
        )

        return {
            "name": f"patient{record.patient_id}_case{record.case_id}",
            "projs": projs,
            "points": points.astype(np.float32),
            "proj_points": proj_points.astype(np.float32),
            "p_gt": p_gt.astype(np.float32),
            "valid_mask": valid_mask,
            "trace": {
                "patient_id": record.patient_id,
                "case_id": record.case_id,
                "case_path": str(record.case_path),
                "view_key": self.view_key,
                "selected_projections": [
                    {
                        "file": item.file,
                        "target_angle_deg": item.target_angle_deg,
                        "gantry_angle_deg": item.gantry_angle_deg,
                        "angular_residual_deg": item.angular_residual_deg,
                    }
                    for item in selected
                ],
                "detector_shape": list(detector_shape),
                "projection_size": self.projection_size,
                "point_sampling": sampling_trace,
                "adapter_version": adapter.calibration.adapter_version,
                "candidate_id": adapter.calibration.candidate_id,
            },
        }
