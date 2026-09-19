from __future__ import annotations

import numpy as np

CBCT_HU_MIN = -290.0
CBCT_HU_MAX = 2060.0
CBCT_HU_SPAN = CBCT_HU_MAX - CBCT_HU_MIN
CBCT_LEGACY_PROFILE_ID = "legacy_hu_minus290_2060"

PROJECTION_MIN = 0.0
PROJECTION_MAX = 8.6

VIEW_TARGETS: dict[str, tuple[int, ...]] = {
    "view_2": (200, 290),
    "view_5": (200, 230, 260, 290, 320),
    "view_8": (200, 215, 230, 260, 275, 290, 320, 335),
    "view_10": (200, 215, 230, 245, 260, 275, 290, 305, 320, 335),
}
SUPPORTED_VIEW_KEYS = tuple(VIEW_TARGETS)
SUPPORTED_VIEW_COUNTS = tuple(int(key.split("_", 1)[1]) for key in VIEW_TARGETS)


def view_key_from_count(view_count: int) -> str:
    key = f"view_{int(view_count)}"
    if key not in VIEW_TARGETS:
        choices = ", ".join(str(item) for item in SUPPORTED_VIEW_COUNTS)
        raise ValueError(f"view_count must be one of {choices}")
    return key


def cbct_normalization_profile(
    *,
    hu_min: float = CBCT_HU_MIN,
    hu_max: float = CBCT_HU_MAX,
    profile_id: str = CBCT_LEGACY_PROFILE_ID,
    mode: str = "legacy",
    source: dict | None = None,
) -> dict:
    hu_min = float(hu_min)
    hu_max = float(hu_max)
    if hu_max <= hu_min:
        raise ValueError(f"hu_max must be greater than hu_min, got {hu_min}, {hu_max}")
    return {
        "profile_id": profile_id,
        "mode": mode,
        "rule": f"clip(HU, {hu_min:g}, {hu_max:g}) then scale to [0, 1]",
        "hu_min": hu_min,
        "hu_max": hu_max,
        "hu_span": float(hu_max - hu_min),
        "value_range": [0.0, 1.0],
        **({"source": source} if source else {}),
    }


LEGACY_CBCT_NORMALIZATION = cbct_normalization_profile()


def cbct_hu_span(normalization: dict | None = None) -> float:
    if normalization is None:
        return CBCT_HU_SPAN
    if "hu_span" in normalization:
        return float(normalization["hu_span"])
    if "hu_min" in normalization and "hu_max" in normalization:
        return float(normalization["hu_max"]) - float(normalization["hu_min"])
    return CBCT_HU_SPAN


def _hu_bounds(
    hu_min: float | None = None,
    hu_max: float | None = None,
    normalization: dict | None = None,
) -> tuple[float, float]:
    if normalization is not None:
        hu_min = float(normalization.get("hu_min", CBCT_HU_MIN))
        hu_max = float(normalization.get("hu_max", CBCT_HU_MAX))
    low = CBCT_HU_MIN if hu_min is None else float(hu_min)
    high = CBCT_HU_MAX if hu_max is None else float(hu_max)
    if high <= low:
        raise ValueError(f"invalid HU window: hu_min={low}, hu_max={high}")
    return low, high


def normalize_cbct_hu(
    array: np.ndarray,
    *,
    hu_min: float | None = None,
    hu_max: float | None = None,
    normalization: dict | None = None,
) -> np.ndarray:
    low, high = _hu_bounds(hu_min, hu_max, normalization)
    arr = np.asarray(array, dtype=np.float32)
    arr = np.clip(arr, low, high)
    return ((arr - low) / (high - low)).astype(np.float32)


def denormalize_cbct_hu(
    array: np.ndarray,
    *,
    hu_min: float | None = None,
    hu_max: float | None = None,
    normalization: dict | None = None,
) -> np.ndarray:
    low, high = _hu_bounds(hu_min, hu_max, normalization)
    arr = np.asarray(array, dtype=np.float32)
    return arr * (high - low) + low


def normalize_projection(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.clip(arr, PROJECTION_MIN, PROJECTION_MAX)
    return ((arr - PROJECTION_MIN) / (PROJECTION_MAX - PROJECTION_MIN)).astype(np.float32)
