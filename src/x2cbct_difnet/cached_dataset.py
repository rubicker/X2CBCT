from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset

from .cache import read_json, resolve_cache_path
from .calibration import load_calibration_report
from .geometry import ClinicalGeometryAdapter, RtkScanGeometry, VolumeGeometry, read_rtk_scan_geometry
from .normalization import LEGACY_CBCT_NORMALIZATION
from .official_drr import OfficialDrrConfig
from .point_blocks import sample_official_lattice_indices
from .setup_augmentation import sample_transformed_volume_setup, transform_volume_setup, volume_background_value
from .volume_grid import axis_convention, processed_grid_from_case_manifest


LOCAL_CACHE_SCHEMA = "x2cbct_difnet_v0b_cache.v2"
SYNTHRAD_CACHE_SCHEMA = "x2cbct_synthrad2025_task2_view_cache.v1"
FULL_TARGET_VOLUME_MODE = "full"
BASE_REFERENCE_TARGET_VOLUME_MODE = "base_reference"
C2RV_GRID_SCALES = (16, 8, 4)
_BASE_REFERENCE_ALIASES = {
    BASE_REFERENCE_TARGET_VOLUME_MODE,
    "base-reference",
    "lightweight",
    "online",
    "online_base",
}


def load_cache_manifest(cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    manifest = read_json(root / "cache_manifest.json")
    schema = manifest.get("schema")
    if schema == LOCAL_CACHE_SCHEMA:
        manifest.setdefault("geometry_branch", "real_projection")
        manifest.setdefault("axis_convention", axis_convention())
        if manifest["split"].get("patient_overlap") and not manifest["split"].get("allow_patient_overlap", False):
            raise ValueError("cache manifest contains train/test patient overlap")
        if manifest["calibration_report"].get("status") != "pass":
            raise ValueError("cache manifest calibration report is not pass")
        point_sampling = manifest["point_sampling"]
        if point_sampling.get("fixed_training_points_saved") is not False:
            raise ValueError("cache manifest must not save fixed training point payloads")
        if point_sampling.get("name") != "official_interleaved_lattice":
            raise ValueError("cache manifest must use official interleaved lattice point blocks")
        return manifest
    if schema == SYNTHRAD_CACHE_SCHEMA:
        manifest.setdefault("geometry_branch", "simulated_drr")
        manifest.setdefault("axis_convention", axis_convention())
        point_sampling = manifest["point_sampling"]
        if point_sampling.get("name") != "official_interleaved_lattice":
            raise ValueError("SynthRAD cache manifest must use official interleaved lattice point blocks")
        if not manifest.get("samples"):
            raise ValueError("SynthRAD cache manifest has no samples")
        out = dict(manifest)
        source = str(out.get("source", "simulated_drr"))
        group = str(out.get("dataset_group", out.get("anatomy", root.name)))
        out.setdefault("cache_version", f"{source}_{group}")
        out.setdefault("cbct_normalization", dict(LEGACY_CBCT_NORMALIZATION))
        return out
    if schema != "x2cbct_difnet_v0b_cache.v2":
        raise ValueError(f"unsupported cache manifest schema: {manifest.get('schema')}")


def samples_for(cache_manifest: dict[str, Any], split: str, view_key: str) -> list[dict[str, Any]]:
    samples = [
        item
        for item in cache_manifest["samples"]
        if item["split"] == split and item["view_key"] == view_key
    ]
    if not samples:
        counts: dict[tuple[str, str], int] = {}
        for item in cache_manifest["samples"]:
            key = (str(item.get("split")), str(item.get("view_key")))
            counts[key] = counts.get(key, 0) + 1
        available = "; ".join(
            f"split={available_split}, view_key={available_view}, count={count}"
            for (available_split, available_view), count in sorted(counts.items())
        )
        matching_splits = sorted(
            available_split
            for available_split, available_view in counts
            if available_view == view_key
        )
        suggestion = (
            f"; use split={matching_splits[0]} for view_key={view_key}"
            if len(matching_splits) == 1
            else ""
        )
        raise ValueError(
            f"no samples for split={split}, view_key={view_key}; "
            f"available combinations: {available or 'none'}{suggestion}"
        )
    return samples


def _as_scalar(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def _load_sample_npz(cache_root: Path, sample: dict[str, Any]) -> dict[str, Any]:
    path = resolve_cache_path(cache_root, sample["sample_path"])
    with np.load(path, allow_pickle=False) as raw:
        out = {
            "path": path,
            "sample_id": _as_scalar(raw["sample_id"]),
            "split": _as_scalar(raw["split"]),
            "view_key": _as_scalar(raw["view_key"]),
            "case_key": _as_scalar(raw["case_key"]),
            "patient_id": _as_scalar(raw["patient_id"]),
            "case_id": _as_scalar(raw["case_id"]),
            "case_manifest_path": _as_scalar(raw["case_manifest_path"]),
            "cbct_payload_path": _as_scalar(raw["cbct_payload_path"]),
            "projection_payload_paths": [str(item) for item in raw["projection_payload_paths"].tolist()],
            "projection_files": [str(item) for item in raw["projection_files"].tolist()],
            "target_angles_deg": raw["target_angles_deg"].astype(np.float32),
            "gantry_angles_deg": raw["gantry_angles_deg"].astype(np.float32),
            "angular_residuals_deg": raw["angular_residuals_deg"].astype(np.float32),
            "detector_shape": tuple(int(item) for item in raw["detector_shape"].tolist()),
            "detector_spacing_yx": tuple(float(item) for item in raw["detector_spacing_yx"].tolist()),
            "projection_size": int(raw["projection_size"].item()),
        }
        if "augmentation_id" in raw.files:
            out.update(
                {
                    "augmentation_id": _as_scalar(raw["augmentation_id"]),
                    "translation_mm": float(raw["translation_mm"].item()),
                    "rotation_deg": float(raw["rotation_deg"].item()),
                    "source_patient_id": _as_scalar(raw["source_patient_id"]),
                    "source_case_id": _as_scalar(raw["source_case_id"]),
                    "source_case_key": _as_scalar(raw["source_case_key"]),
                }
            )
        if "target_volume_mode" in raw.files:
            out["target_volume_mode"] = _as_scalar(raw["target_volume_mode"])
        if "volume_fill_value" in raw.files:
            out["volume_fill_value"] = float(raw["volume_fill_value"].item())
        if "target_sampling_fill_value" in raw.files:
            out["target_sampling_fill_value"] = float(raw["target_sampling_fill_value"].item())
        return out


def _setup_trace_fields(source: dict[str, Any]) -> dict[str, Any]:
    if "setup_augmentation" in source and isinstance(source["setup_augmentation"], dict):
        setup = source["setup_augmentation"]
        return {
            key: value
            for key, value in {
                "augmentation_id": setup.get("augmentation_id"),
                "translation_mm": setup.get("translation_mm"),
                "rotation_deg": setup.get("rotation_deg"),
                "source_patient_id": setup.get("source_patient_id"),
                "source_case_id": setup.get("source_case_id"),
                "source_case_key": setup.get("source_case_key"),
            }.items()
            if value is not None
        }
    fields = {}
    for key in (
        "augmentation_id",
        "translation_mm",
        "rotation_deg",
        "source_patient_id",
        "source_case_id",
        "source_case_key",
    ):
        if key in source:
            fields[key] = source[key]
    return fields


def _target_volume_mode(cache_manifest: dict[str, Any], sample: dict[str, Any]) -> str:
    raw = sample.get("target_volume_mode", cache_manifest.get("target_volume_mode", FULL_TARGET_VOLUME_MODE))
    mode = str(raw or FULL_TARGET_VOLUME_MODE).strip().lower()
    if mode == FULL_TARGET_VOLUME_MODE:
        return FULL_TARGET_VOLUME_MODE
    if mode in _BASE_REFERENCE_ALIASES:
        return BASE_REFERENCE_TARGET_VOLUME_MODE
    raise ValueError(f"unsupported target_volume_mode: {raw}")


def _setup_values(sample: dict[str, Any]) -> tuple[float, float, float | None]:
    setup = sample.get("setup_augmentation") if isinstance(sample.get("setup_augmentation"), dict) else {}
    translation_mm = float(setup.get("translation_mm", sample.get("translation_mm", 0)) or 0)
    rotation_deg = float(setup.get("rotation_deg", sample.get("rotation_deg", 0)) or 0)
    fill = setup.get(
        "target_sampling_fill_value",
        setup.get("volume_fill_value", sample.get("target_sampling_fill_value", sample.get("volume_fill_value"))),
    )
    return translation_mm, rotation_deg, None if fill is None else float(fill)


def _volume_fill_value(sample: dict[str, Any], cbct: np.ndarray) -> float:
    _, _, fill = _setup_values(sample)
    if fill is not None:
        return fill
    return volume_background_value(cbct)


def _target_values_at_indices(
    cbct: np.ndarray,
    indices_zyx: np.ndarray,
    *,
    cache_manifest: dict[str, Any],
    sample: dict[str, Any],
    spacing_z_mm: float,
) -> np.ndarray:
    mode = _target_volume_mode(cache_manifest, sample)
    translation_mm, rotation_deg, _ = _setup_values(sample)
    if mode == FULL_TARGET_VOLUME_MODE or (translation_mm == 0 and rotation_deg == 0):
        return np.asarray(cbct[indices_zyx[:, 0], indices_zyx[:, 1], indices_zyx[:, 2]], dtype=np.float32)
    return sample_transformed_volume_setup(
        cbct,
        indices_zyx,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        spacing_z_mm=float(spacing_z_mm),
        fill_value=_volume_fill_value(sample, cbct),
    )


def _materialized_target_volume(
    cbct: np.ndarray,
    *,
    cache_manifest: dict[str, Any],
    sample: dict[str, Any],
    spacing_z_mm: float,
) -> np.ndarray:
    mode = _target_volume_mode(cache_manifest, sample)
    translation_mm, rotation_deg, _ = _setup_values(sample)
    if mode == FULL_TARGET_VOLUME_MODE or (translation_mm == 0 and rotation_deg == 0):
        return np.asarray(cbct, dtype=np.float32)
    return transform_volume_setup(
        cbct,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        spacing_z_mm=float(spacing_z_mm),
        fill_value=_volume_fill_value(sample, cbct),
    )


def _load_projections(cache_root: Path, sample_npz: dict[str, Any]) -> np.ndarray:
    arrays = []
    for rel_path in sample_npz["projection_payload_paths"]:
        arr = np.load(resolve_cache_path(cache_root, rel_path)).astype(np.float32)
        arrays.append(arr[None, ...])
    return np.stack(arrays, axis=0).astype(np.float32)


def _load_synthrad_projections(cache_root: Path, sample: dict[str, Any]) -> np.ndarray:
    scale = float(sample.get("projection_loader_scale") or 1.0)
    arrays = []
    for path in sample["projection_paths"]:
        arr = np.load(resolve_cache_path(cache_root, str(path))).astype(np.float32)
        if scale != 1.0:
            arr = arr / scale
        arrays.append(arr[None, ...])
    return np.stack(arrays, axis=0).astype(np.float32)


def _volume_geometry(case_manifest: dict[str, Any]) -> VolumeGeometry:
    grid = processed_grid_from_case_manifest(case_manifest)
    return VolumeGeometry(
        spacing_xyz=tuple(float(item) for item in grid["spacing_xyz"]),
        origin_xyz=tuple(float(item) for item in grid["origin_xyz"]),
        direction=np.asarray(grid["direction"], dtype=np.float64).reshape(3, 3),
        shape_zyx=tuple(int(item) for item in grid["shape_zyx"]),
    )


def _scan_geometry_from_case_manifest(cache_root: Path, case_manifest: dict[str, Any]) -> RtkScanGeometry:
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


def _eval_indices(shape_zyx: tuple[int, int, int], eval_grid_size: int) -> np.ndarray:
    axes = [np.linspace(0, size - 1, eval_grid_size, dtype=np.float32) for size in shape_zyx]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    return np.stack([z, y, x], axis=-1).reshape(-1, 3).astype(np.float32)


def _c2rv_grid_indices(shape_zyx: tuple[int, int, int], grid_size: int) -> np.ndarray:
    axes = [np.linspace(0, size - 1, int(grid_size), dtype=np.float32) for size in shape_zyx]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    return np.stack([z, y, x], axis=-1).reshape(-1, 3).astype(np.float32)


def _c2rv_points01(grid_size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([z, y, x], axis=-1).reshape(-1, 3).astype(np.float32)


def _resize_volume(array: np.ndarray, size: int) -> np.ndarray:
    factors = [size / dim for dim in array.shape]
    return scipy.ndimage.zoom(array.astype(np.float32), factors, order=1, prefilter=False).astype(np.float32)


def _points_01_from_indices(indices_zyx: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    denom = np.asarray([max(size - 1, 1) for size in shape_zyx], dtype=np.float32)
    return (np.asarray(indices_zyx, dtype=np.float32) / denom[None, :]).astype(np.float32)


def _official_model_points(points_01_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_01_zyx, dtype=np.float32).copy()
    points[:, :2] -= 0.5
    points[:, 2] = 0.5 - points[:, 2]
    return (points * 2.0).astype(np.float32)


def _official_project_points(points_01_zyx: np.ndarray, angles_deg: list[float], config: OfficialDrrConfig) -> tuple[np.ndarray, np.ndarray]:
    points_mm = np.asarray(points_01_zyx, dtype=np.float64).copy()
    points_mm[:, :2] -= 0.5
    points_mm[:, 2] = 0.5 - points_mm[:, 2]
    points_mm *= float(config.voxel_size) * float(config.voxel_spacing_mm)

    projected = []
    masks = []
    for angle_deg in angles_deg:
        angle = -float(angle_deg) * np.pi / 180.0
        rot = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotated = points_mm @ rot.T
        coeff = float(config.dsd_mm) / (float(config.dso_mm) - rotated[:, 0])
        detector_points = rotated[:, [2, 1]] * coeff[:, None]
        detector_points /= float(config.detector_size) * float(config.detector_spacing_mm)
        detector_points *= 2.0
        projected.append(detector_points.astype(np.float32))
        masks.append(np.logical_and(detector_points >= -1.0, detector_points <= 1.0).all(axis=1))
    return np.stack(projected, axis=0).astype(np.float32), np.stack(masks, axis=0).astype(bool)


def _official_drr_config_from_manifest(
    cache_manifest: dict[str, Any],
    cbct_shape: tuple[int, int, int],
    sample: dict[str, Any] | None = None,
) -> OfficialDrrConfig:
    raw = dict((sample or {}).get("drr_config") or cache_manifest.get("drr_config") or {})
    spacing = (sample or {}).get("processed_spacing_xyz")
    voxel_spacing = float(raw.get("voxel_spacing_mm", 0.8))
    if spacing is not None and max(abs(float(item) - float(spacing[0])) for item in spacing) <= 1e-5:
        voxel_spacing = float(spacing[0])
    return OfficialDrrConfig(
        projection_count=int(raw.get("projection_count", 50)),
        total_angle_deg=float(raw.get("total_angle_deg", 180.0)),
        start_angle_deg=float(raw.get("start_angle_deg", 0.0)),
        detector_size=int(raw.get("detector_size", cache_manifest.get("projection_size", 256))),
        voxel_size=int(raw.get("voxel_size", cache_manifest.get("target_size", cbct_shape[0]))),
        dsd_mm=float(raw.get("dsd_mm", 2000.0)),
        dso_mm=float(raw.get("dso_mm", 1000.0)),
        detector_spacing_mm=float(raw.get("detector_spacing_mm", 2.0)),
        voxel_spacing_mm=voxel_spacing,
    )


def _project_root_for_cache(cache_root: Path) -> Path:
    calibration_rel = Path("runs") / "audits" / "rtk_coordinate_candidates" / "calibration_report.json"
    for candidate in (cache_root, *cache_root.parents):
        if (candidate / calibration_rel).exists():
            return candidate
        if candidate.name == "data_cache":
            return candidate.parent
    return Path.cwd().resolve()


class CachedDifNetV0bDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        view_key: str,
        num_points: int = 4096,
        max_cases: int | None = None,
        seed: int = 20260522,
        deterministic: bool = False,
        eval_grid_size: int | None = None,
        c2rv_grids: bool = False,
        c2rv_grid_scales: tuple[int, ...] = C2RV_GRID_SCALES,
    ) -> None:
        if split not in {"train", "eval", "test"}:
            raise ValueError("split must be train, eval, or test")
        self.cache_root = Path(cache_root).resolve()
        self.cache_manifest = load_cache_manifest(self.cache_root)
        self.cache_schema = str(self.cache_manifest["schema"])
        self.geometry_branch = str(self.cache_manifest.get("geometry_branch", "real_projection"))
        if self.geometry_branch not in {"real_projection", "simulated_drr"}:
            raise ValueError(f"unsupported geometry_branch: {self.geometry_branch}")
        if self.cache_schema == SYNTHRAD_CACHE_SCHEMA and self.geometry_branch != "simulated_drr":
            raise ValueError("SynthRAD view cache must declare geometry_branch=simulated_drr")
        if self.cache_schema == LOCAL_CACHE_SCHEMA and self.geometry_branch != "real_projection":
            raise ValueError("local V0b cache must declare geometry_branch=real_projection")
        if view_key not in self.cache_manifest["views"]:
            raise ValueError(f"view_key {view_key} is not present in cache")
        self.split = split
        self.view_key = view_key
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.eval_grid_size = eval_grid_size
        self.c2rv_grids = bool(c2rv_grids)
        self.c2rv_grid_scales = tuple(int(item) for item in c2rv_grid_scales)
        self.samples = samples_for(self.cache_manifest, split, view_key)
        if max_cases is not None:
            self.samples = self.samples[: int(max_cases)]
        self.calibration = None
        if self.cache_schema == LOCAL_CACHE_SCHEMA:
            project_root = _project_root_for_cache(self.cache_root)
            self.calibration = load_calibration_report(project_root, require_pass=True)
            if self.calibration.candidate_id != self.cache_manifest["calibration_report"]["candidate_id"]:
                raise ValueError("calibration candidate in cache does not match current report")
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _add_c2rv_real_grids(
        self,
        item: dict[str, Any],
        *,
        adapter: ClinicalGeometryAdapter,
        volume_geometry: VolumeGeometry,
        sample: dict[str, Any],
    ) -> None:
        if not self.c2rv_grids:
            return
        for scale_index, grid_size in enumerate(self.c2rv_grid_scales):
            indices = _c2rv_grid_indices(volume_geometry.shape_zyx, grid_size)
            points = volume_geometry.model_points_from_indices(indices)
            physical_points = volume_geometry.physical_points_from_indices(indices)
            proj_points, valid_mask = adapter.project_many_views(
                physical_points,
                [float(value) for value in sample["gantry_angles_deg"]],
                detector_shape=sample["detector_shape"],
                detector_spacing=sample["detector_spacing_yx"],
                feature_shape=(sample["projection_size"], sample["projection_size"]),
            )
            item[f"c2rv_grid_points_s{scale_index}"] = points.astype(np.float32)
            item[f"c2rv_grid_proj_points_s{scale_index}"] = proj_points.astype(np.float32)
            item[f"c2rv_grid_valid_mask_s{scale_index}"] = valid_mask.astype(bool)

    def _add_c2rv_synthrad_grids(
        self,
        item: dict[str, Any],
        *,
        sample: dict[str, Any],
        config: OfficialDrrConfig,
    ) -> None:
        if not self.c2rv_grids:
            return
        for scale_index, grid_size in enumerate(self.c2rv_grid_scales):
            points01 = _c2rv_points01(grid_size)
            proj_points, valid_mask = _official_project_points(
                points01,
                [float(value) for value in sample["angles_deg"]],
                config,
            )
            item[f"c2rv_grid_points_s{scale_index}"] = _official_model_points(points01).astype(np.float32)
            item[f"c2rv_grid_proj_points_s{scale_index}"] = proj_points.astype(np.float32)
            item[f"c2rv_grid_valid_mask_s{scale_index}"] = valid_mask.astype(bool)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + int(index))
        return self._rng

    def _load_common(self, index: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, VolumeGeometry]:
        sample = _load_sample_npz(self.cache_root, self.samples[index])
        case_manifest = read_json(resolve_cache_path(self.cache_root, sample["case_manifest_path"]))
        cbct_path = resolve_cache_path(self.cache_root, sample["cbct_payload_path"])
        if self.eval_grid_size is None:
            cbct = np.load(cbct_path, mmap_mode="r")
        else:
            cbct = np.load(cbct_path).astype(np.float32)
        return sample, case_manifest, cbct, _volume_geometry(case_manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.geometry_branch == "simulated_drr":
            return self._getitem_synthrad(index)
        if self.geometry_branch != "real_projection":
            raise ValueError(f"unsupported geometry_branch: {self.geometry_branch}")
        sample, case_manifest, cbct, volume_geometry = self._load_common(index)
        projs = _load_projections(self.cache_root, sample)
        if self.calibration is None:
            raise RuntimeError("local V0b cache requires a loaded calibration report")
        adapter = ClinicalGeometryAdapter(
            self.calibration,
            _scan_geometry_from_case_manifest(self.cache_root, case_manifest),
        )
        if self.eval_grid_size is None:
            rng = self._rng_for_index(index)
            grid_size = int(case_manifest.get("target_size", case_manifest["shape_zyx"][0]))
            indices, sampling_trace = sample_official_lattice_indices(self.num_points, rng, grid_size=grid_size)
            p_gt = _target_values_at_indices(
                cbct,
                indices,
                cache_manifest=self.cache_manifest,
                sample=sample,
                spacing_z_mm=float(volume_geometry.spacing_xyz[2]),
            )[None, :]
            points = volume_geometry.model_points_from_indices(indices)
            physical_points = volume_geometry.physical_points_from_indices(indices)
            image = None
        else:
            sampling_trace = {"point_sampling": "eval_grid", "eval_grid_size": int(self.eval_grid_size)}
            indices = _eval_indices(volume_geometry.shape_zyx, int(self.eval_grid_size))
            full_target = _materialized_target_volume(
                cbct,
                cache_manifest=self.cache_manifest,
                sample=sample,
                spacing_z_mm=float(volume_geometry.spacing_xyz[2]),
            )
            target = _resize_volume(full_target, int(self.eval_grid_size))
            p_gt = target.reshape(1, -1).astype(np.float32)
            points = volume_geometry.model_points_from_indices(indices)
            physical_points = volume_geometry.physical_points_from_indices(indices)
            image = target
        proj_points, valid_mask = adapter.project_many_views(
            physical_points,
            [float(item) for item in sample["gantry_angles_deg"]],
            detector_shape=sample["detector_shape"],
            detector_spacing=sample["detector_spacing_yx"],
            feature_shape=(sample["projection_size"], sample["projection_size"]),
        )
        item = {
            "name": sample["sample_id"],
            "projs": projs,
            "points": points.astype(np.float32),
            "proj_points": proj_points.astype(np.float32),
            "p_gt": p_gt.astype(np.float32),
            "valid_mask": valid_mask.astype(bool),
            "view_angles_deg": sample["gantry_angles_deg"].astype(np.float32),
            "trace": {
                "sample_path": str(sample["path"]),
                "cache_root": str(self.cache_root),
                "cache_version": self.cache_manifest["cache_version"],
                "patient_id": sample["patient_id"],
                "case_id": sample["case_id"],
                "case_key": sample["case_key"],
                "view_key": sample["view_key"],
                "split": sample["split"],
                "projection_files": sample["projection_files"],
                "target_angles_deg": sample["target_angles_deg"].tolist(),
                "gantry_angles_deg": sample["gantry_angles_deg"].tolist(),
                "angular_residuals_deg": sample["angular_residuals_deg"].tolist(),
                "detector_shape": list(sample["detector_shape"]),
                "detector_spacing_yx": list(sample["detector_spacing_yx"]),
                "projection_size": sample["projection_size"],
                "geometry_branch": self.cache_manifest.get("geometry_branch", "real_projection"),
                "processed_shape_zyx": list(volume_geometry.shape_zyx),
                "processed_spacing_xyz": list(volume_geometry.spacing_xyz),
                "processed_origin_xyz": list(volume_geometry.origin_xyz),
                "processed_direction": [float(item) for item in volume_geometry.direction.reshape(-1)],
                "point_sampling": sampling_trace,
                "candidate_id": self.calibration.candidate_id,
                "adapter_version": self.calibration.adapter_version,
                "target_volume_mode": _target_volume_mode(self.cache_manifest, sample),
            },
        }
        item["trace"].update(_setup_trace_fields(sample))
        self._add_c2rv_real_grids(
            item,
            adapter=adapter,
            volume_geometry=volume_geometry,
            sample=sample,
        )
        if image is not None:
            item["image"] = image.astype(np.float32)
        return item

    def _getitem_synthrad(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cbct_path = resolve_cache_path(self.cache_root, str(sample["processed_cbct_path"]))
        cbct = np.load(cbct_path, mmap_mode=None if self.eval_grid_size is not None else "r")
        cbct_shape = tuple(int(item) for item in cbct.shape)
        projs = _load_synthrad_projections(self.cache_root, sample)
        config = _official_drr_config_from_manifest(self.cache_manifest, cbct_shape, sample)
        if self.eval_grid_size is None:
            rng = self._rng_for_index(index)
            grid_size = int(self.cache_manifest.get("target_size", cbct_shape[0]))
            indices, sampling_trace = sample_official_lattice_indices(self.num_points, rng, grid_size=grid_size)
            spacing = sample.get("processed_spacing_xyz", [config.voxel_spacing_mm] * 3)
            p_gt = _target_values_at_indices(
                cbct,
                indices,
                cache_manifest=self.cache_manifest,
                sample=sample,
                spacing_z_mm=float(spacing[2]),
            )[None, :]
            points_01 = _points_01_from_indices(indices, cbct_shape)
            image = None
        else:
            sampling_trace = {"point_sampling": "eval_grid", "eval_grid_size": int(self.eval_grid_size)}
            indices = _eval_indices(cbct_shape, int(self.eval_grid_size))
            spacing = sample.get("processed_spacing_xyz", [1.0, 1.0, 1.0])
            full_target = _materialized_target_volume(
                cbct,
                cache_manifest=self.cache_manifest,
                sample=sample,
                spacing_z_mm=float(spacing[2]),
            )
            target = _resize_volume(full_target, int(self.eval_grid_size))
            p_gt = target.reshape(1, -1).astype(np.float32)
            points_01 = _points_01_from_indices(indices, cbct_shape)
            image = target
        points = _official_model_points(points_01)
        proj_points, valid_mask = _official_project_points(
            points_01,
            [float(item) for item in sample["angles_deg"]],
            config,
        )
        case_key = f"{sample.get('anatomy', self.cache_manifest.get('anatomy', 'synthrad'))}_{sample['case_id']}"
        item = {
            "name": sample["sample_id"],
            "projs": projs,
            "points": points.astype(np.float32),
            "proj_points": proj_points.astype(np.float32),
            "p_gt": p_gt.astype(np.float32),
            "valid_mask": valid_mask.astype(bool),
            "view_angles_deg": np.asarray([float(item) for item in sample["angles_deg"]], dtype=np.float32),
            "trace": {
                "sample_path": str(self.cache_root / "cache_manifest.json"),
                "cache_root": str(self.cache_root),
                "cache_version": self.cache_manifest["cache_version"],
                "source": sample.get("source", self.cache_manifest.get("source")),
                "anatomy": sample.get("anatomy", self.cache_manifest.get("anatomy")),
                "patient_id": case_key,
                "case_id": sample["case_id"],
                "case_key": case_key,
                "view_key": sample["view_key"],
                "split": sample["split"],
                "projection_files": sample["projection_paths"],
                "target_angles_deg": [float(item) for item in sample["angles_deg"]],
                "gantry_angles_deg": [float(item) for item in sample["angles_deg"]],
                "angular_residuals_deg": [0.0 for _ in sample["angles_deg"]],
                "detector_shape": [int(self.cache_manifest.get("projection_size", 256)), int(self.cache_manifest.get("projection_size", 256))],
                "detector_spacing_yx": [float(config.detector_spacing_mm), float(config.detector_spacing_mm)],
                "projection_size": int(self.cache_manifest.get("projection_size", 256)),
                "geometry_branch": self.cache_manifest.get("geometry_branch", "simulated_drr"),
                "processed_shape_zyx": list(sample.get("processed_shape_zyx", cbct_shape)),
                "processed_spacing_xyz": [float(item) for item in sample.get("processed_spacing_xyz", [config.voxel_spacing_mm] * 3)],
                "processed_origin_xyz": [float(item) for item in sample.get("processed_origin_xyz", [0.0, 0.0, 0.0])],
                "processed_direction": [
                    float(item)
                    for item in sample.get("processed_direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                ],
                "drr_config": config.to_dict(),
                "point_sampling": sampling_trace,
                "projector": sample.get("projector", self.cache_manifest.get("projector")),
                "projection_value_space": sample.get("projection_value_space", self.cache_manifest.get("projection_value_space")),
                "projection_loader_scale": sample.get("projection_loader_scale", self.cache_manifest.get("projection_loader_scale")),
                "target_volume_mode": _target_volume_mode(self.cache_manifest, sample),
            },
        }
        item["trace"].update(_setup_trace_fields(sample))
        self._add_c2rv_synthrad_grids(item, sample=sample, config=config)
        if image is not None:
            item["image"] = image.astype(np.float32)
        return item
