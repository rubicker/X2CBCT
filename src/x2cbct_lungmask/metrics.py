from __future__ import annotations

import numpy as np
import scipy.ndimage


def dice_coefficient(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {ref.shape}")
    denominator = int(pred.sum()) + int(ref.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred, ref).sum())
    return float(2.0 * intersection / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    eroded = scipy.ndimage.binary_erosion(binary, structure=np.ones((3, 3, 3), dtype=bool))
    return np.logical_and(binary, np.logical_not(eroded))


def average_symmetric_surface_distance_mm(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> float:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {ref.shape}")
    if not pred.any() or not ref.any():
        return float("nan")

    pred_surface = _surface(pred)
    ref_surface = _surface(ref)
    spacing_zyx = tuple(float(value) for value in reversed(spacing_xyz))
    distance_to_ref = scipy.ndimage.distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    distance_to_pred = scipy.ndimage.distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    pred_to_ref = distance_to_ref[pred_surface]
    ref_to_pred = distance_to_pred[ref_surface]
    return float((pred_to_ref.sum() + ref_to_pred.sum()) / (pred_to_ref.size + ref_to_pred.size))
