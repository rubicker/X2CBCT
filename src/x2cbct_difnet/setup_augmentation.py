from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.ndimage


DEFAULT_SETUP_TRANSLATION_MM = (-10, 0, 10)
DEFAULT_SETUP_ROTATION_DEG = (-10, 0, 10)


@dataclass(frozen=True)
class SetupAugmentationConfig:
    enabled: bool = False
    translation_mm: tuple[int, ...] = DEFAULT_SETUP_TRANSLATION_MM
    rotation_deg: tuple[int, ...] = DEFAULT_SETUP_ROTATION_DEG

    @property
    def combination_count(self) -> int:
        return len(self.translation_mm) * len(self.rotation_deg) if self.enabled else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "translation_mm": list(self.translation_mm),
            "rotation_deg": list(self.rotation_deg),
            "combination_count_per_source_case": self.combination_count,
            "requires_zero_bin": True,
            "integer_bins_only": True,
            "order": "rotation first, then translation",
        }


@dataclass(frozen=True)
class SetupAugmentation:
    translation_mm: int
    rotation_deg: int

    @property
    def id(self) -> str:
        return f"si_{bin_label(self.translation_mm)}_rot_{bin_label(self.rotation_deg)}"

    def to_dict(self) -> dict[str, object]:
        return {
            "augmentation_id": self.id,
            "translation_mm": int(self.translation_mm),
            "rotation_deg": int(self.rotation_deg),
        }


def parse_int_bins(values: Iterable[int | str] | None, *, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    bins: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(f"{name} only accepts integer bins, got {value!r}") from exc
        if str(parsed) != text and not (text.startswith("+") and str(parsed) == text[1:]):
            raise ValueError(f"{name} only accepts integer bins, got {value!r}")
        bins.append(parsed)
    if not bins:
        raise ValueError(f"{name} must contain at least one integer bin")
    deduped = tuple(dict.fromkeys(bins))
    if 0 not in deduped:
        raise ValueError(f"{name} must include 0")
    return deduped


def setup_augmentation_config(
    *,
    enabled: bool,
    translation_mm: Iterable[int | str] | None = None,
    rotation_deg: Iterable[int | str] | None = None,
) -> SetupAugmentationConfig:
    return SetupAugmentationConfig(
        enabled=bool(enabled),
        translation_mm=parse_int_bins(translation_mm, name="translation_mm", default=DEFAULT_SETUP_TRANSLATION_MM),
        rotation_deg=parse_int_bins(rotation_deg, name="rotation_deg", default=DEFAULT_SETUP_ROTATION_DEG),
    )


def iter_augmentations(config: SetupAugmentationConfig) -> tuple[SetupAugmentation, ...]:
    if not config.enabled:
        return (SetupAugmentation(translation_mm=0, rotation_deg=0),)
    return tuple(
        SetupAugmentation(translation_mm=shift, rotation_deg=angle)
        for shift in config.translation_mm
        for angle in config.rotation_deg
    )


def bin_label(value: int) -> str:
    value = int(value)
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "0"


def volume_background_value(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {arr.shape}")
    edge = np.concatenate(
        [
            arr[0, :, :].ravel(),
            arr[-1, :, :].ravel(),
            arr[:, 0, :].ravel(),
            arr[:, -1, :].ravel(),
            arr[:, :, 0].ravel(),
            arr[:, :, -1].ravel(),
        ]
    )
    return float(np.nanmedian(edge))


def image_background_value(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")
    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.nanmedian(edge))


def transform_volume_setup(
    array: np.ndarray,
    *,
    translation_mm: float,
    rotation_deg: float,
    spacing_z_mm: float,
    fill_value: float,
) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if float(rotation_deg) != 0.0:
        out = scipy.ndimage.rotate(
            out,
            float(rotation_deg),
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=float(fill_value),
            prefilter=False,
        ).astype(np.float32)
    if float(translation_mm) != 0.0:
        out = scipy.ndimage.shift(
            out,
            shift=(float(translation_mm) / float(spacing_z_mm), 0.0, 0.0),
            order=1,
            mode="constant",
            cval=float(fill_value),
            prefilter=False,
        ).astype(np.float32)
    return out.astype(np.float32)


def inverse_setup_coordinates_zyx(
    indices_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
    *,
    translation_mm: float,
    rotation_deg: float,
    spacing_z_mm: float,
) -> np.ndarray:
    coords = np.asarray(indices_zyx, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"indices_zyx must have shape (N, 3), got {coords.shape}")
    if len(shape_zyx) != 3:
        raise ValueError(f"shape_zyx must contain three dimensions, got {shape_zyx}")
    out = coords.astype(np.float32, copy=True)
    if float(translation_mm) != 0.0:
        out[:, 0] -= float(translation_mm) / float(spacing_z_mm)
    if float(rotation_deg) != 0.0:
        angle = np.deg2rad(float(rotation_deg))
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        cy = (float(shape_zyx[1]) - 1.0) / 2.0
        cx = (float(shape_zyx[2]) - 1.0) / 2.0
        y = out[:, 1] - cy
        x = out[:, 2] - cx
        out[:, 1] = cos_a * y + sin_a * x + cy
        out[:, 2] = -sin_a * y + cos_a * x + cx
    return out.astype(np.float32)


def sample_transformed_volume_setup(
    array: np.ndarray,
    indices_zyx: np.ndarray,
    *,
    translation_mm: float,
    rotation_deg: float,
    spacing_z_mm: float,
    fill_value: float,
) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {arr.shape}")
    source_coords = inverse_setup_coordinates_zyx(
        indices_zyx,
        tuple(int(item) for item in arr.shape),
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        spacing_z_mm=spacing_z_mm,
    )
    sampled = scipy.ndimage.map_coordinates(
        arr,
        [source_coords[:, 0], source_coords[:, 1], source_coords[:, 2]],
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=False,
    )
    return sampled.astype(np.float32)


def shift_image_vertical(array: np.ndarray, pixel_shift_y: float, *, fill_value: float) -> np.ndarray:
    return scipy.ndimage.shift(
        np.asarray(array, dtype=np.float32),
        shift=(float(pixel_shift_y), 0.0),
        order=1,
        mode="constant",
        cval=float(fill_value),
        prefilter=False,
    ).astype(np.float32)
