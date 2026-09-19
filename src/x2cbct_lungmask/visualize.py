from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .experiment import LEFT_LUNG_LABEL, RIGHT_LUNG_LABEL, _load_experiment, _prepare, _safe_case_name
from .manifests import resolve_project_path


def _slice(array: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return array[index, :, :]
    if plane == "coronal":
        return array[:, index, :]
    if plane == "sagittal":
        return array[:, :, index]
    raise ValueError(plane)


def _overlay(mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask == LEFT_LUNG_LABEL] = (0.90, 0.05, 0.05, 0.42)
    rgba[mask == RIGHT_LUNG_LABEL] = (0.05, 0.80, 0.12, 0.42)
    return rgba


def _representative_indices(reference_mask: np.ndarray) -> dict[str, int]:
    foreground = np.argwhere(reference_mask > 0)
    if foreground.size == 0:
        return {
            "axial": reference_mask.shape[0] // 2,
            "coronal": reference_mask.shape[1] // 2,
            "sagittal": reference_mask.shape[2] // 2,
        }
    center = np.rint(foreground.mean(axis=0)).astype(int)
    return {"axial": int(center[0]), "coronal": int(center[1]), "sagittal": int(center[2])}


def _median_primary_case(experiment_dir: Path, primary_method: str) -> str:
    metrics_path = experiment_dir / "metrics_per_case.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == primary_method
            and row.get("status") == "ok"
            and row.get("bilateral_dice_percent") not in (None, "")
        ]
    if not rows:
        raise ValueError(f"no valid {primary_method} rows in {metrics_path}")
    values = np.asarray([float(row["bilateral_dice_percent"]) for row in rows], dtype=np.float64)
    median = float(np.median(values))
    return str(min(rows, key=lambda row: abs(float(row["bilateral_dice_percent"]) - median))["case_id"])


def plot_case(
    *,
    config_path: Path,
    project_root: Path,
    experiment_dir: Path,
    case_id: str,
    output_path: Path,
    window_hu: tuple[float, float],
) -> Path:
    config, references, methods = _load_experiment(config_path, project_root)
    if case_id == "auto":
        primary_method = str(config.get("primary_method") or "")
        if not primary_method:
            raise ValueError("config.primary_method is required when --case-id auto")
        case_id = _median_primary_case(experiment_dir, primary_method)
    if case_id not in references:
        raise KeyError(f"case_id not found in reference manifest: {case_id}")
    target_shape = tuple(int(value) for value in config.get("target_shape_zyx", [128, 128, 128]))
    safe_case = _safe_case_name(case_id)

    reference_prepared = _prepare(references[case_id], target_shape)
    reference_mask_path = experiment_dir / "masks" / "reference" / f"{safe_case}.nii.gz"
    if not reference_mask_path.exists():
        raise FileNotFoundError(reference_mask_path)
    reference_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(reference_mask_path))).astype(np.uint8)

    columns: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("Ground truth", reference_prepared.hu_zyx, reference_mask)
    ]
    display_names = {
        str(method_name): str(display_name)
        for method_name, display_name in (config.get("display_names") or {}).items()
    }
    for method_name, records in methods.items():
        mask_path = experiment_dir / "masks" / method_name / f"{safe_case}.nii.gz"
        if not mask_path.exists():
            continue
        prepared = _prepare(records[case_id], target_shape)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)
        columns.append((display_names.get(method_name, method_name), prepared.hu_zyx, mask))
    if len(columns) < 2:
        raise RuntimeError("no method masks are available for this case")

    indices = _representative_indices(reference_mask)
    planes = ("axial", "coronal", "sagittal")
    figure, axes = plt.subplots(len(planes), len(columns), figsize=(2.35 * len(columns), 6.8), squeeze=False)
    for row_index, plane in enumerate(planes):
        index = indices[plane]
        for column_index, (name, volume, mask) in enumerate(columns):
            axis = axes[row_index, column_index]
            image_slice = _slice(volume, plane, index)
            mask_slice = _slice(mask, plane, index)
            axis.imshow(image_slice, cmap="gray", vmin=window_hu[0], vmax=window_hu[1], origin="lower")
            axis.imshow(_overlay(mask_slice), origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(name, fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"{plane}\nindex {index}", fontsize=9)
    figure_title = str(config.get("figure_title") or f"LungMask R231 comparison: {case_id}")
    figure.suptitle(f"{figure_title}\nred=left lung, green=right lung", fontsize=11)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main(project_root: Path | None = None) -> int:
    root = project_root or Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Plot a three-plane LungMask comparison for one X2CBCT case.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="auto", help="'auto' selects the valid primary-method case nearest median Dice")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-min-hu", type=float, default=-1000.0)
    parser.add_argument("--window-max-hu", type=float, default=400.0)
    args = parser.parse_args()
    output = plot_case(
        config_path=resolve_project_path(args.config, project_root=root),
        project_root=root,
        experiment_dir=resolve_project_path(args.experiment_dir, project_root=root),
        case_id=args.case_id,
        output_path=resolve_project_path(args.output, project_root=root),
        window_hu=(float(args.window_min_hu), float(args.window_max_hu)),
    )
    print(output)
    return 0
