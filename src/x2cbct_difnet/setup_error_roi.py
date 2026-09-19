from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


@dataclass(frozen=True)
class RoiBox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int


@dataclass(frozen=True)
class RoiCandidate:
    patient_id: str
    case_id: str
    cbct_path: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    threshold_low: float
    threshold_high: float
    status: str
    note: str
    mask_voxel_count: int
    boundary_touch_ratio: float
    box_volume_voxels: int


ROI_FIELDS = [
    "patient_id",
    "case_id",
    "cbct_path",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "threshold_low",
    "threshold_high",
    "status",
    "note",
    "mask_voxel_count",
    "boundary_touch_ratio",
    "box_volume_voxels",
]


def load_cbct(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected 3D CBCT, got shape {array.shape} from {path}")
    return image, array


def generate_bone_mask(
    array: np.ndarray,
    *,
    threshold_low: float = 700.0,
    threshold_high: float = 1700.0,
) -> np.ndarray:
    bone = (array >= threshold_low) & (array <= threshold_high)
    z_size, y_size, x_size = array.shape
    interior = np.zeros_like(bone, dtype=bool)
    interior[
        max(2, int(z_size * 0.02)) : min(z_size, int(z_size * 0.98)),
        max(5, int(y_size * 0.04)) : min(y_size, int(y_size * 0.98)),
        max(5, int(x_size * 0.04)) : min(x_size, int(x_size * 0.96)),
    ] = True
    cervical_search = np.zeros_like(bone, dtype=bool)
    cervical_search[
        int(z_size * 0.16) : int(z_size * 0.84),
        int(y_size * 0.46) : int(y_size * 0.88),
        int(x_size * 0.32) : int(x_size * 0.68),
    ] = True
    return bone & interior & cervical_search


def largest_clean_components(mask: np.ndarray, *, min_voxels: int = 400) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask & False
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.zeros(count + 1, dtype=bool)
    for label, size in enumerate(sizes):
        if label == 0 or size < min_voxels:
            continue
        component = labels == label
        if component_touches_boundary(component):
            continue
        keep[label] = True
    cleaned = keep[labels]
    if cleaned.any():
        return cleaned
    largest = int(np.argmax(sizes))
    return labels == largest


def component_touches_boundary(component: np.ndarray) -> bool:
    return bool(
        component[0].any()
        or component[-1].any()
        or component[:, 0, :].any()
        or component[:, -1, :].any()
        or component[:, :, 0].any()
        or component[:, :, -1].any()
    )


def bbox_from_mask(mask: np.ndarray, pad: int = 8) -> RoiBox | None:
    if not mask.any():
        return None
    zz, yy, xx = np.where(mask)
    z_size, y_size, x_size = mask.shape
    return RoiBox(
        x_min=max(0, int(xx.min()) - pad),
        x_max=min(x_size - 1, int(xx.max()) + pad),
        y_min=max(0, int(yy.min()) - pad),
        y_max=min(y_size - 1, int(yy.max()) + pad),
        z_min=max(0, int(zz.min()) - pad),
        z_max=min(z_size - 1, int(zz.max()) + pad),
    )


def mask_boundary_touch_ratio(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    return float((mask & boundary).sum() / max(mask.sum(), 1))


def generate_roi_candidate(
    patient_id: str,
    case_id: str,
    cbct_path: Path,
    *,
    threshold_low: float = 700.0,
    threshold_high: float = 1700.0,
) -> RoiCandidate:
    _, array = load_cbct(cbct_path)
    raw_mask = generate_bone_mask(array, threshold_low=threshold_low, threshold_high=threshold_high)
    mask = largest_clean_components(raw_mask)
    box = bbox_from_mask(mask)
    if box is None:
        z_size, y_size, x_size = array.shape
        box = RoiBox(
            x_min=int(x_size * 0.35),
            x_max=int(x_size * 0.65),
            y_min=int(y_size * 0.45),
            y_max=int(y_size * 0.90),
            z_min=int(z_size * 0.10),
            z_max=int(z_size * 0.90),
        )
        status = "needs_fix"
        note = "automatic cervical bone mask is empty"
    else:
        z_extent = box.z_max - box.z_min + 1
        y_extent = box.y_max - box.y_min + 1
        x_extent = box.x_max - box.x_min + 1
        too_large = z_extent > int(array.shape[0] * 0.86) or y_extent > int(array.shape[1] * 0.62) or x_extent > int(array.shape[2] * 0.62)
        status = "accepted" if mask.sum() >= 1000 and not too_large else "needs_fix"
        if mask.sum() < 1000:
            note = "small automatic cervical bone mask"
        elif too_large:
            note = "automatic ROI is broad and should be checked"
        else:
            note = ""
    volume = (box.x_max - box.x_min + 1) * (box.y_max - box.y_min + 1) * (box.z_max - box.z_min + 1)
    return RoiCandidate(
        patient_id=str(patient_id),
        case_id=str(case_id),
        cbct_path=str(cbct_path),
        x_min=box.x_min,
        x_max=box.x_max,
        y_min=box.y_min,
        y_max=box.y_max,
        z_min=box.z_min,
        z_max=box.z_max,
        threshold_low=float(threshold_low),
        threshold_high=float(threshold_high),
        status=status,
        note=note,
        mask_voxel_count=int(mask.sum()),
        boundary_touch_ratio=mask_boundary_touch_ratio(mask),
        box_volume_voxels=int(volume),
    )


def roi_mask_from_candidate(array: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    box = RoiBox(
        x_min=int(row["x_min"]),
        x_max=int(row["x_max"]),
        y_min=int(row["y_min"]),
        y_max=int(row["y_max"]),
        z_min=int(row["z_min"]),
        z_max=int(row["z_max"]),
    )
    low = float(row["threshold_low"])
    high = float(row["threshold_high"])
    mask = np.zeros(array.shape, dtype=bool)
    mask[box.z_min : box.z_max + 1, box.y_min : box.y_max + 1, box.x_min : box.x_max + 1] = True
    return mask & (array >= low) & (array <= high)


def write_roi_candidates(rows: list[RoiCandidate], csv_path: Path, json_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROI_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    if json_path is not None:
        json_path.write_text(
            json.dumps({"schema": "x2cbct_cervical_roi_candidates.v1", "cases": [asdict(r) for r in rows]}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )


def read_roi_csv(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            rows[(str(row["patient_id"]), str(row["case_id"]))] = row
        return rows
