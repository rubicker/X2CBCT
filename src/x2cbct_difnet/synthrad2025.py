from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import SimpleITK as sitk

from .cache import write_json
from .ct_only_preprocess import center_crop_pad_zyx, resample_image_to_spacing
from .normalization import LEGACY_CBCT_NORMALIZATION, normalize_cbct_hu
from .point_blocks import official_lattice_metadata
from .volume_grid import axis_convention, volume_grid_payload


TASK2_TRAIN_URL = "https://zenodo.org/api/records/15373853/files/synthRAD2025_Task2_Train.zip/content"
TASK2_VAL_INPUT_URL = "https://zenodo.org/api/records/15373659/files/synthRAD2025_Task2_Val_Input.zip/content"


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    size: int
    md5: str


SYNTHRAD2025_TASK2_DOWNLOADS: dict[str, DownloadSpec] = {
    "task2-train": DownloadSpec(
        name="synthRAD2025_Task2_Train.zip",
        url=TASK2_TRAIN_URL,
        size=11978062547,
        md5="1a39969f6d6c95113189ab284c3e57e7",
    ),
    "task2-val-input": DownloadSpec(
        name="synthRAD2025_Task2_Val_Input.zip",
        url=TASK2_VAL_INPUT_URL,
        size=1218336116,
        md5="3095210657b0bc42db7e871c130d291c",
    ),
}


ANATOMY_ALIASES = {
    "chest": ("chest", "thorax", "thoracic", "lung", "lungs", "th"),
    "abdomen": ("abdomen", "abdominal", "abd", "ab"),
}
SYNTHRAD_VIEW_COUNTS = {"view_2": 2, "view_5": 5, "view_8": 8, "view_10": 10}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(
    spec: DownloadSpec,
    output_dir: Path,
    *,
    proxy: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / spec.name
    temp_path = output_dir / f"{spec.name}.part"
    lock_path = output_dir / f"{spec.name}.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"download lock exists, another download may be running: {lock_path}") from exc

    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)

        if out_path.exists():
            final_size = out_path.stat().st_size
            if final_size == spec.size:
                digest = file_md5(out_path)
                if digest != spec.md5:
                    raise ValueError(f"md5 mismatch for {out_path}: got {digest}, expected {spec.md5}")
                return {
                    "ok": True,
                    "path": str(out_path),
                    "downloaded": final_size,
                    "md5": digest,
                    "complete": True,
                    "reused": True,
                }
            if final_size < spec.size:
                if not temp_path.exists() or temp_path.stat().st_size < final_size:
                    out_path.replace(temp_path)
            else:
                out_path.unlink()
    
        existing = temp_path.stat().st_size if temp_path.exists() else 0
        if existing > spec.size:
            temp_path.unlink()
            existing = 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        proxies = {"http": proxy, "https": proxy} if proxy else None
        with requests.get(spec.url, headers=headers, stream=True, proxies=proxies, timeout=60) as response:
            response.raise_for_status()
            mode = "ab" if existing and response.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0
            with temp_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
        if temp_path.stat().st_size != spec.size:
            return {
                "ok": False,
                "path": str(temp_path),
                "downloaded": temp_path.stat().st_size,
                "expected": spec.size,
                "complete": False,
            }
        digest = file_md5(temp_path)
        if digest != spec.md5:
            raise ValueError(f"md5 mismatch for {temp_path}: got {digest}, expected {spec.md5}")
        temp_path.replace(out_path)
        return {"ok": True, "path": str(out_path), "downloaded": spec.size, "md5": digest, "complete": True}
    finally:
        if lock_path.exists():
            lock_path.unlink()


def extract_zip(zip_path: Path, extract_root: Path) -> dict[str, Any]:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        archive.extractall(extract_root)
    return {"zip_path": str(zip_path), "extract_root": str(extract_root), "file_count": len(names)}


def _path_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for part in path.parts:
        for token in part.replace("-", "_").split("_"):
            cleaned = token.strip().lower()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def infer_anatomy(path: Path) -> str | None:
    tokens = _path_tokens(path)
    for anatomy, aliases in ANATOMY_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            return anatomy
    lower = str(path).lower()
    for anatomy, aliases in ANATOMY_ALIASES.items():
        if any(alias in lower for alias in aliases if len(alias) > 2):
            return anatomy
    return None


def _find_sibling_mask(cbct_path: Path) -> Path | None:
    for path in cbct_path.parent.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if "mask" in name and (name.endswith(".mha") or name.endswith(".mhd") or name.endswith(".nii") or name.endswith(".nii.gz")):
            return path
    return None


def find_cbct_files(raw_root: Path, anatomies: Iterable[str] = ("chest", "abdomen")) -> list[dict[str, Any]]:
    allowed = set(anatomies)
    candidates = []
    suffixes = {".mha", ".mhd", ".nii", ".gz"}
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "cbct" not in name:
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        anatomy = infer_anatomy(path)
        if anatomy not in allowed:
            continue
        case_id = path.parent.name
        mask_path = _find_sibling_mask(path)
        record = {
            "case_id": case_id,
            "anatomy": anatomy,
            "source": "synthrad2025_task2",
            "cbct_path": str(path.resolve()),
            "relative_cbct_path": str(path.resolve().relative_to(raw_root.resolve())),
        }
        if mask_path is not None:
            record["mask_path"] = str(mask_path.resolve())
            record["relative_mask_path"] = str(mask_path.resolve().relative_to(raw_root.resolve()))
        candidates.append(
            record
        )
    return sorted(candidates, key=lambda item: (item["anatomy"], item["relative_cbct_path"]))


def assign_split(records: list[dict[str, Any]], seed: int = 20260525, train_ratio: float = 0.8) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["anatomy"]), []).append(record)
    assigned: list[dict[str, Any]] = []
    for anatomy, rows in sorted(grouped.items()):
        rows = list(rows)
        random.Random(seed + sum(ord(ch) for ch in anatomy)).shuffle(rows)
        train_count = int(round(len(rows) * train_ratio))
        for index, row in enumerate(rows):
            out = dict(row)
            out["split"] = "train" if index < train_count else "test"
            out["split_seed"] = seed
            out["split_train_ratio"] = train_ratio
            assigned.append(out)
    return sorted(assigned, key=lambda item: (item["anatomy"], item["split"], item["case_id"]))


def write_registry(records: list[dict[str, Any]], output_path: Path, raw_root: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        anatomy = str(record["anatomy"])
        split = str(record["split"])
        counts.setdefault(anatomy, {}).setdefault(split, 0)
        counts[anatomy][split] += 1
    registry = {
        "schema": "x2cbct_synthrad2025_task2_registry.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root.resolve()),
        "allowed_anatomies": ["chest", "abdomen"],
        "cbct_only": True,
        "ct_used": False,
        "split_policy": "8:2 train/test; test also acts as validation",
        "counts": counts,
        "cases": records,
    }
    write_json(registry, output_path)
    return registry


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def preprocess_cbct_case(
    record: dict[str, Any],
    output_root: Path,
    target_size: int = 128,
    cbct_normalization: dict[str, Any] | None = None,
    resample_spacing_xyz: tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> dict[str, Any]:
    normalization = cbct_normalization or LEGACY_CBCT_NORMALIZATION
    image = sitk.ReadImage(str(record["cbct_path"]))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    resampled_image = resample_image_to_spacing(image, resample_spacing_xyz, default_value=-1024.0)
    resampled = sitk.GetArrayFromImage(resampled_image).astype(np.float32)
    target_shape_zyx = (int(target_size), int(target_size), int(target_size))
    resized, crop_pad = center_crop_pad_zyx(resampled, target_shape_zyx, pad_value=-1024.0)
    normalized = normalize_cbct_hu(resized, normalization=normalization).astype(np.float16)
    case_dir = output_root / str(record["anatomy"]) / str(record["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / f"cbct_norm_{target_size}.npy"
    np.save(out_path, normalized)
    finite_mask = np.isfinite(resized)
    coverage_mask = finite_mask
    mask_path = record.get("mask_path")
    if mask_path:
        raw_mask_image = sitk.ReadImage(str(mask_path))
        raw_mask = sitk.GetArrayFromImage(raw_mask_image).astype(np.float32)
        if raw_mask.shape != array.shape:
            raise ValueError(f"mask shape {raw_mask.shape} does not match cbct shape {array.shape}")
        mask_resampled_image = sitk.Resample(
            raw_mask_image,
            resampled_image.GetSize(),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            resampled_image.GetOrigin(),
            resampled_image.GetSpacing(),
            resampled_image.GetDirection(),
            0,
            raw_mask_image.GetPixelID(),
        )
        mask_resampled = sitk.GetArrayFromImage(mask_resampled_image).astype(np.float32)
        resized_mask, _ = center_crop_pad_zyx(mask_resampled, target_shape_zyx, pad_value=0.0)
        resized_mask = resized_mask > 0
        coverage_mask = resized_mask & finite_mask
        mask_out_path = case_dir / f"mask_{target_size}.npy"
        np.save(mask_out_path, resized_mask.astype(np.uint8))
    else:
        mask_out_path = None
    spacing_xyz = tuple(float(item) for item in image.GetSpacing())
    target_spacing_xyz = tuple(float(item) for item in resample_spacing_xyz)
    origin_xyz = tuple(float(item) for item in image.GetOrigin())
    direction = [float(item) for item in image.GetDirection()]
    grid_payload = volume_grid_payload(
        raw_shape_zyx=array.shape,
        raw_spacing_xyz=spacing_xyz,
        raw_origin_xyz=origin_xyz,
        raw_direction=direction,
        processed_shape_zyx=target_shape_zyx,
        processed_spacing_xyz=target_spacing_xyz,
        processed_origin_xyz=[0.0, 0.0, 0.0],
        processed_direction=direction,
    )
    coverage = float(np.count_nonzero(coverage_mask) / coverage_mask.size)
    payload = {
        **record,
        "status": "preprocessed",
        "geometry_branch": "simulated_drr",
        "axis_convention": axis_convention(),
        **grid_payload,
        "raw_shape_zyx": list(array.shape),
        "target_shape_zyx": [target_size, target_size, target_size],
        "raw_spacing_xyz": list(spacing_xyz),
        "target_spacing_xyz": [float(item) for item in target_spacing_xyz],
        "shape_zyx": [target_size, target_size, target_size],
        "spacing_xyz": [float(item) for item in target_spacing_xyz],
        "origin_xyz": [0.0, 0.0, 0.0],
        "direction": direction,
        "preprocess": {
            "policy": "spacing_aware_resample_then_center_crop_pad",
            "resample_spacing_xyz": [float(item) for item in target_spacing_xyz],
            "target_size": int(target_size),
            "target_shape_zyx": [int(item) for item in target_shape_zyx],
            "crop_pad_policy": "center",
            "crop_pad": crop_pad,
            "resampler": "SimpleITK.ResampleImageFilter.linear",
            "canonical_center_is_isocenter": True,
            "raw_origin_used_for_training_geometry": False,
        },
        "processed_cbct_path": str(out_path.resolve()),
        "cbct_normalization": normalization,
        "coverage_ratio": coverage,
        "hu_min": float(np.nanmin(resized)),
        "hu_max": float(np.nanmax(resized)),
        "hu_mean": float(np.nanmean(resized)),
    }
    if mask_out_path is not None:
        payload["processed_mask_path"] = str(mask_out_path.resolve())
    return {
        **payload,
    }


def synthrad_view_indices(total_projections: int, view_key: str) -> list[int]:
    if view_key not in SYNTHRAD_VIEW_COUNTS:
        raise ValueError(f"unsupported SynthRAD view key: {view_key}")
    if int(total_projections) == 10:
        fixed_view_10_indices = {
            "view_2": [0, 6],
            "view_5": [0, 2, 4, 6, 8],
            "view_8": [0, 1, 2, 4, 5, 6, 8, 9],
            "view_10": list(range(10)),
        }
        return fixed_view_10_indices[view_key]
    view_count = SYNTHRAD_VIEW_COUNTS[view_key]
    if total_projections < view_count:
        raise ValueError(f"need at least {view_count} projections, got {total_projections}")
    return [int(item) for item in np.linspace(0, total_projections, view_count, endpoint=False).astype(int)]


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "unknown"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _row_group(row: dict[str, Any], group_key: str | None, default_group: str) -> str:
    if group_key and row.get(group_key) is not None:
        return str(row[group_key])
    return str(row.get("anatomy", row.get("dataset_name", default_group)))


def _cache_level_drr_config(rows: list[dict[str, Any]], drr_manifest: dict[str, Any]) -> dict[str, Any]:
    configs = [row.get("drr_config") for row in rows if row.get("drr_config")]
    if not configs:
        return drr_manifest.get("drr_config", {})
    first = configs[0]
    if all(config == first for config in configs):
        return first
    out = dict(drr_manifest.get("drr_config", {}))
    out["per_case_config_required"] = True
    return out


def _safe_component(value: Any) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return cleaned.strip("_") or "case"


def _copy_asset(src: str | Path, dst: Path, copied: dict[str, str]) -> str:
    source = Path(src).resolve()
    key = str(source)
    if key in copied:
        return copied[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if source != dst.resolve():
        shutil.copy2(source, dst)
    copied[key] = str(dst.resolve())
    return copied[key]


def build_synthrad_view_caches(
    drr_manifest: dict[str, Any],
    output_root: Path,
    *,
    views: Iterable[str] = ("view_2", "view_5", "view_8", "view_10"),
    target_size: int = 128,
    source_name: str = "synthrad2025_task2",
    group_key: str | None = "anatomy",
    default_group: str = "synthrad2025_task2",
    split_policy: str = "train/test from upstream registry; test also acts as validation",
    asset_mode: str = "reference",
) -> list[Path]:
    rows = list(drr_manifest.get("cases", []))
    if not rows:
        raise ValueError("drr manifest has no generated cases")
    if asset_mode not in {"reference", "copy"}:
        raise ValueError("asset_mode must be 'reference' or 'copy'")
    selected_views = [str(item) for item in views]
    for view in selected_views:
        if view not in SYNTHRAD_VIEW_COUNTS:
            raise ValueError(f"unsupported view key: {view}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_row_group(row, group_key, default_group), []).append(row)

    written: list[Path] = []
    for group_name, group_rows in sorted(grouped.items()):
        cache_root = output_root / group_name
        copied_assets: dict[str, str] = {}
        cbct_normalization = _cbct_normalization_for_anatomy(drr_manifest, group_name, group_rows)
        cache_drr_config = _cache_level_drr_config(group_rows, drr_manifest)
        samples = []
        for row in sorted(group_rows, key=lambda item: (str(item.get("split")), str(item.get("case_id")))):
            projection_paths = [str(item) for item in row["projection_paths"]]
            angles = [float(item) for item in row["angles_deg"]]
            case_dir_name = _safe_component(row["case_id"])
            if asset_mode == "copy":
                cbct_path = _copy_asset(row["processed_cbct_path"], cache_root / "cases" / case_dir_name / "ct_norm.npy", copied_assets)
                mask_path = (
                    _copy_asset(row["processed_mask_path"], cache_root / "cases" / case_dir_name / "mask.npy", copied_assets)
                    if row.get("processed_mask_path")
                    else None
                )
            else:
                cbct_path = row["processed_cbct_path"]
                mask_path = row.get("processed_mask_path")
            for view_key in selected_views:
                indices = synthrad_view_indices(len(projection_paths), view_key)
                sample_id = f"{source_name}_{group_name}_{row['case_id']}_{view_key}"
                if asset_mode == "copy":
                    selected_projection_paths = [
                        _copy_asset(
                            projection_paths[index],
                            cache_root / "projections" / case_dir_name / Path(projection_paths[index]).name,
                            copied_assets,
                        )
                        for index in indices
                    ]
                else:
                    selected_projection_paths = [projection_paths[index] for index in indices]
                sample = {
                    "sample_id": sample_id,
                    "source": source_name,
                    "anatomy": group_name,
                    "dataset_group": group_name,
                    "case_id": row["case_id"],
                    "split": row["split"],
                    "view_key": view_key,
                    "view_count": SYNTHRAD_VIEW_COUNTS[view_key],
                    "view_indices": indices,
                    "angles_deg": [angles[index] for index in indices],
                    "projection_paths": selected_projection_paths,
                    "processed_cbct_path": cbct_path,
                    "processed_mask_path": mask_path,
                    "processed_shape_zyx": row.get("processed_shape_zyx", row.get("target_shape_zyx")),
                    "processed_size_xyz": row.get("processed_size_xyz"),
                    "processed_spacing_xyz": row.get("processed_spacing_xyz", row.get("target_spacing_xyz")),
                    "processed_origin_xyz": row.get("processed_origin_xyz", row.get("origin_xyz", [0.0, 0.0, 0.0])),
                    "processed_direction": row.get("processed_direction", row.get("direction")),
                    "processed_physical_fov_mm_xyz": row.get("processed_physical_fov_mm_xyz"),
                    "cbct_normalization": row.get("cbct_normalization", cbct_normalization),
                    "projection_value_space": row.get("projection_value_space"),
                    "projection_loader_scale": row.get("projection_loader_scale"),
                    "projector": row.get("projector"),
                    "drr_config": row.get("drr_config", drr_manifest.get("drr_config", {})),
                    "drr_config_notes": row.get("drr_config_notes", {}),
                    "target_volume_mode": row.get("target_volume_mode", drr_manifest.get("target_volume_mode", "full")),
                }
                if isinstance(row.get("setup_augmentation"), dict):
                    setup = row["setup_augmentation"]
                    sample["setup_augmentation"] = setup
                    sample["augmentation_id"] = setup.get("augmentation_id")
                    sample["translation_mm"] = setup.get("translation_mm")
                    sample["rotation_deg"] = setup.get("rotation_deg")
                    sample["source_case_id"] = setup.get("source_case_id", row.get("source_case_id"))
                samples.append(sample)
        cache_manifest = {
            "schema": "x2cbct_synthrad2025_task2_view_cache.v1",
            "geometry_branch": "simulated_drr",
            "axis_convention": axis_convention(),
            "source": source_name,
            "anatomy": group_name,
            "dataset_group": group_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "drr_manifest": str(drr_manifest.get("manifest_path", "")),
            "projection_base_manifest": str(drr_manifest.get("preprocess_report", "")),
            "drr_config": cache_drr_config,
            "test_is_validation": True,
            "split_policy": split_policy,
            "asset_mode": asset_mode,
            "self_contained_assets": asset_mode == "copy",
            "target_size": int(target_size),
            "projection_size": int(cache_drr_config.get("detector_size", target_size)),
            "cbct_normalization": cbct_normalization,
            "views": selected_views,
            "view_counts": {view: SYNTHRAD_VIEW_COUNTS[view] for view in selected_views},
            "point_sampling": official_lattice_metadata(grid_size=target_size),
            "projection_value_space": drr_manifest.get("projection_value_space"),
            "projection_loader_scale": drr_manifest.get("projection_loader_scale"),
            "projector": drr_manifest.get("projector"),
            "target_volume_mode": drr_manifest.get("target_volume_mode", "full"),
            "setup_augmentation": drr_manifest.get("setup_augmentation", {"enabled": False}),
            "counts": {
                "cases": len(group_rows),
                "split": _split_counts(group_rows),
                "samples": len(samples),
            },
            "samples": samples,
        }
        manifest_path = cache_root / "cache_manifest.json"
        write_json(cache_manifest, manifest_path)
        written.append(manifest_path)
    return written


def _cbct_normalization_for_anatomy(
    drr_manifest: dict[str, Any],
    anatomy: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    profiles = (drr_manifest.get("cbct_normalization") or {}).get("profiles", {})
    if anatomy in profiles:
        return profiles[anatomy]
    for row in rows:
        if row.get("cbct_normalization"):
            return row["cbct_normalization"]
    return dict(LEGACY_CBCT_NORMALIZATION)
