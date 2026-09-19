from __future__ import annotations

import math
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

from .normalization import CBCT_HU_SPAN, cbct_hu_span


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32))))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def ssim_3d(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred_arr = np.asarray(pred, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"pred and target shape mismatch: {pred_arr.shape} vs {target_arr.shape}")
    if pred_arr.ndim != 3 or min(pred_arr.shape) < 7:
        return float("nan")
    return float(structural_similarity(target_arr, pred_arr, data_range=float(data_range)))


def metric_dict(pred: np.ndarray, target: np.ndarray, *, hu_span: float | None = None, normalization: dict | None = None) -> dict[str, Any]:
    norm_mae = mae(pred, target)
    span = cbct_hu_span(normalization) if normalization is not None else (CBCT_HU_SPAN if hu_span is None else float(hu_span))
    return {
        "mae_norm": norm_mae,
        "mae_hu": norm_mae * span,
        "psnr": psnr(pred, target, data_range=1.0),
        "ssim": ssim_3d(pred, target, data_range=1.0),
    }


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {"mae_norm": float("nan"), "mae_hu": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
    keys = ("mae_norm", "mae_hu", "psnr", "ssim")
    return {key: float(np.mean([float(item[key]) for item in items])) for key in keys}
