from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from .normalization import LEGACY_CBCT_NORMALIZATION, cbct_normalization_profile


def parse_percentiles(text: str | tuple[float, float] | list[float]) -> tuple[float, float]:
    if isinstance(text, str):
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        values = [float(item) for item in text]
    if len(values) != 2:
        raise ValueError("percentiles must contain exactly two values")
    low, high = values
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(f"invalid percentiles: {values}")
    return low, high


def _resize_volume(array: np.ndarray, target_size: int | None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if target_size is None or tuple(arr.shape) == (target_size, target_size, target_size):
        return arr
    factors = [int(target_size) / dim for dim in arr.shape]
    return scipy.ndimage.zoom(arr, factors, order=1, prefilter=False).astype(np.float32)


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if tuple(arr.shape) == target_shape:
        return arr > 0
    factors = [target_shape[index] / arr.shape[index] for index in range(3)]
    return scipy.ndimage.zoom(arr, factors, order=0, prefilter=False) > 0


def deterministic_sample(values: np.ndarray, max_voxels: int | None, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if max_voxels is None or arr.size <= int(max_voxels):
        return arr
    rng = np.random.default_rng(seed)
    indices = rng.choice(arr.size, size=int(max_voxels), replace=False)
    return arr[indices]


def profile_from_values(
    values: np.ndarray,
    *,
    percentiles: tuple[float, float],
    profile_id: str,
    mode: str = "percentile",
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("cannot estimate HU window from zero finite voxels")
    p_low, p_high = parse_percentiles(percentiles)
    hu_min = float(np.percentile(arr, p_low))
    hu_max = float(np.percentile(arr, p_high))
    adjusted_constant_window = False
    if hu_max <= hu_min:
        center = hu_min
        hu_min = center - 0.5
        hu_max = center + 0.5
        adjusted_constant_window = True
    profile = cbct_normalization_profile(
        hu_min=hu_min,
        hu_max=hu_max,
        profile_id=profile_id,
        mode=mode,
        source=source,
    )
    profile.update(
        {
            "percentiles": {"low": p_low, "high": p_high},
            "sampled_voxel_count": int(arr.size),
            "source_min": float(np.min(arr)),
            "source_max": float(np.max(arr)),
            "source_mean": float(np.mean(arr)),
            "source_median": float(np.median(arr)),
            "clipped_low_ratio": float(np.mean(arr < hu_min)),
            "clipped_high_ratio": float(np.mean(arr > hu_max)),
            "adjusted_constant_window": adjusted_constant_window,
        }
    )
    return profile


def _case_values(
    row: dict[str, Any],
    *,
    image_path: Callable[[dict[str, Any]], Path],
    mask_path: Callable[[dict[str, Any]], Path | None] | None,
    target_size: int | None,
    mask_policy: str,
    max_voxels_per_case: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = sitk.ReadImage(str(image_path(row)))
    raw = sitk.GetArrayFromImage(image).astype(np.float32)
    resized = _resize_volume(raw, target_size)
    finite = np.isfinite(resized)
    scope = finite
    used_mask = False
    if mask_policy == "body" and mask_path is not None:
        path = mask_path(row)
        if path is not None and path.exists():
            mask_image = sitk.ReadImage(str(path))
            raw_mask = sitk.GetArrayFromImage(mask_image)
            if raw_mask.shape != raw.shape:
                raise ValueError(f"mask shape {raw_mask.shape} does not match image shape {raw.shape}")
            scope = _resize_mask(raw_mask, tuple(resized.shape)) & finite
            used_mask = True
    values = resized[scope]
    sampled = deterministic_sample(values, max_voxels_per_case, seed)
    return sampled, {
        "case_id": row.get("case_id", row.get("patient_id")),
        "split": row.get("split"),
        "shape_zyx": [int(item) for item in raw.shape],
        "spacing_xyz": [float(item) for item in image.GetSpacing()],
        "target_size": target_size,
        "voxel_scope": "body_mask" if used_mask else "finite",
        "candidate_voxel_count": int(values.size),
        "sampled_voxel_count": int(sampled.size),
        "min": float(np.min(sampled)) if sampled.size else None,
        "max": float(np.max(sampled)) if sampled.size else None,
    }


def estimate_grouped_hu_profiles(
    rows: Iterable[dict[str, Any]],
    *,
    image_path: Callable[[dict[str, Any]], Path],
    mask_path: Callable[[dict[str, Any]], Path | None] | None = None,
    group_key: str | Callable[[dict[str, Any]], str] = "dataset",
    source_split: str = "train",
    target_size: int | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    mask_policy: str = "finite",
    max_voxels_per_case: int | None = 200_000,
    seed: int = 20260527,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    all_rows = list(rows)
    selected = [row for row in all_rows if source_split == "all" or row.get("split") == source_split]
    if not selected:
        raise ValueError(f"no rows available for HU window source_split={source_split}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        key = group_key(row) if callable(group_key) else str(row.get(group_key, "global"))
        grouped.setdefault(key, []).append(row)
    profiles: dict[str, dict[str, Any]] = {}
    case_stats: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        samples = []
        for index, row in enumerate(group_rows):
            values, stats = _case_values(
                row,
                image_path=image_path,
                mask_path=mask_path,
                target_size=target_size,
                mask_policy=mask_policy,
                max_voxels_per_case=max_voxels_per_case,
                seed=seed + index,
            )
            stats["profile_group"] = key
            case_stats.append(stats)
            if values.size:
                samples.append(values)
        if not samples:
            profiles[key] = dict(LEGACY_CBCT_NORMALIZATION)
            profiles[key]["fallback_reason"] = "no finite sampled voxels"
            continue
        merged = np.concatenate(samples)
        profiles[key] = profile_from_values(
            merged,
            percentiles=percentiles,
            profile_id=f"{key}_p{percentiles[0]:g}_p{percentiles[1]:g}",
            source={
                "scope": key,
                "source_split": source_split,
                "source_case_count": len(group_rows),
                "target_size": target_size,
                "mask_policy": mask_policy,
                "max_voxels_per_case": max_voxels_per_case,
            },
        )
    return profiles, case_stats
