from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy import ndimage
from skimage.morphology import remove_small_objects


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_ROOT = (
    PROJECT_ROOT
    / "data_cache"
    / "difnet_v1_si10_rot10_3x3_256"
    / "self_projection1208"
    / "base_3x3_256"
    / "cases"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "model_experiments"
    / "_shared_baseline"
    / "registration_si_yaw_fixed"
    / "registration_results.csv"
)
DEFAULT_ROI = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "model_experiments"
    / "_shared_baseline"
    / "cervical_roi_review"
    / "roi_reviewed.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "model_experiments"
    / "_shared_baseline"
    / "paper_figures"
    / "fig7_real_positioning_examples"
)

REFERENCE_COLOR = "#00A6D6"
MOVING_COLOR = "#D94B91"
ARROW_COLOR = "#F0A202"
TEXT_COLOR = "#222222"


@dataclass(frozen=True)
class Example:
    label: str
    patient_id: str
    reference_case_id: str
    moving_case_id: str
    plane: str
    fixed_slice: int
    metric: str
    title: str


EXAMPLES = (
    Example(
        label="Example 1",
        patient_id="0",
        reference_case_id="0",
        moving_case_id="10",
        plane="sagittal",
        fixed_slice=134,
        metric="si",
        title="S–I-dominant example",
    ),
    Example(
        label="Example 2",
        patient_id="24",
        reference_case_id="2",
        moving_case_id="8",
        plane="axial",
        fixed_slice=173,
        metric="yaw",
        title="Yaw-dominant example",
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def read_csv_index(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[tuple(str(row[key]) for key in keys)] = row
    return rows


def load_volume(case_root: Path, patient_id: str, case_id: str) -> np.ndarray:
    path = case_root / f"patient{patient_id}_case{case_id}" / "cbct_norm.npy"
    if not path.exists():
        raise FileNotFoundError(f"CBCT cache not found: {path}")
    volume = np.load(path).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CBCT volume, got {volume.shape} from {path}")
    return volume


def union_roi(
    reference_row: dict[str, str],
    moving_row: dict[str, str],
    shape: tuple[int, int, int],
    margin: int = 8,
) -> tuple[slice, slice, slice]:
    z_size, y_size, x_size = shape
    x0 = max(0, min(int(reference_row["x_min"]), int(moving_row["x_min"])) - margin)
    x1 = min(x_size, max(int(reference_row["x_max"]), int(moving_row["x_max"])) + margin + 1)
    y0 = max(0, min(int(reference_row["y_min"]), int(moving_row["y_min"])) - margin)
    y1 = min(y_size, max(int(reference_row["y_max"]), int(moving_row["y_max"])) + margin + 1)
    z0 = max(0, min(int(reference_row["z_min"]), int(moving_row["z_min"])) - margin)
    z1 = min(z_size, max(int(reference_row["z_max"]), int(moving_row["z_max"])) + margin + 1)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def hu_to_normalized(hu: float) -> float:
    return float(np.clip((hu + 290.0) / 2350.0, 0.0, 1.0))


def window_volume(volume: np.ndarray, low_hu: float = -200.0, high_hu: float = 1700.0) -> np.ndarray:
    low = hu_to_normalized(low_hu)
    high = hu_to_normalized(high_hu)
    clipped = np.clip(volume, low, high)
    return (clipped - low) / max(high - low, 1e-6)


def bone_mask(volume: np.ndarray, crop: tuple[slice, slice, slice]) -> np.ndarray:
    low = hu_to_normalized(700.0)
    high = hu_to_normalized(1700.0)
    smoothed = ndimage.gaussian_filter(volume, sigma=0.7)
    mask = (smoothed >= low) & (smoothed <= high)
    restricted = np.zeros_like(mask, dtype=bool)
    restricted[crop] = mask[crop]
    return restricted


def choose_slice(
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    crop: tuple[slice, slice, slice],
    plane: str,
) -> int:
    z_slice, y_slice, x_slice = crop
    combined = reference_mask | moving_mask
    if plane == "sagittal":
        x0, x1 = x_slice.start, x_slice.stop
        center = 0.5 * (x0 + x1 - 1)
        scores = combined[z_slice, y_slice, x_slice].sum(axis=(0, 1)).astype(float)
        positions = np.arange(x0, x1)
    elif plane == "axial":
        z0, z1 = z_slice.start, z_slice.stop
        center = 0.5 * (z0 + z1 - 1)
        ref = reference_mask[z_slice, y_slice, x_slice]
        mov = moving_mask[z_slice, y_slice, x_slice]
        disagreement = np.logical_xor(ref, mov).sum(axis=(1, 2)).astype(float)
        support = np.minimum(ref.sum(axis=(1, 2)), mov.sum(axis=(1, 2))).astype(float)
        scores = disagreement * (support > np.percentile(support, 35))
        positions = np.arange(z0, z1)
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    if not np.any(scores > 0):
        return int(round(center))
    centrality = 1.0 - 0.25 * np.abs(positions - center) / max((positions[-1] - positions[0]) / 2.0, 1.0)
    return int(positions[int(np.argmax(scores * centrality))])


def extract_plane(
    volume: np.ndarray,
    mask: np.ndarray,
    crop: tuple[slice, slice, slice],
    plane: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_slice, y_slice, x_slice = crop
    if plane == "sagittal":
        image = volume[z_slice, y_slice, index]
        plane_mask = mask[z_slice, y_slice, index]
    elif plane == "axial":
        image = volume[index, y_slice, x_slice]
        plane_mask = mask[index, y_slice, x_slice]
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    return image, clean_2d_mask(plane_mask)


def clean_2d_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((3, 3), dtype=bool))
    cleaned = remove_small_objects(cleaned, min_size=18)
    cleaned = cleaned.copy()
    cleaned[:2, :] = False
    cleaned[-2:, :] = False
    cleaned[:, :2] = False
    cleaned[:, -2:] = False
    return cleaned


def draw_image(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_overlay(
    ax: plt.Axes,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    example: Example,
    result: dict[str, str],
) -> None:
    draw_image(ax, reference_image)
    if reference_mask.any():
        ax.contour(
            reference_mask.astype(float),
            levels=[0.5],
            colors=[REFERENCE_COLOR],
            linewidths=1.05,
            linestyles="solid",
        )
    if moving_mask.any():
        ax.contour(
            moving_mask.astype(float),
            levels=[0.5],
            colors=[MOVING_COLOR],
            linewidths=1.05,
            linestyles="dashed",
        )

    height, width = reference_image.shape
    if example.metric == "si":
        value = float(result["setup_error_SI_mm"])
        arrow = FancyArrowPatch(
            (0.90 * width, 0.32 * height),
            (0.90 * width, 0.68 * height),
            arrowstyle="<->",
            mutation_scale=8,
            linewidth=1.4,
            color=ARROW_COLOR,
        )
        ax.add_patch(arrow)
        label = f"ΔS–I = {value:+.2f} mm"
    else:
        value = float(result["setup_error_yaw_deg"])
        arrow = FancyArrowPatch(
            (0.27 * width, 0.20 * height),
            (0.72 * width, 0.22 * height),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.48",
            mutation_scale=8,
            linewidth=1.4,
            color=ARROW_COLOR,
        )
        ax.add_patch(arrow)
        label = f"Δyaw = {value:+.2f}°"

    ax.text(
        0.035,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111111", "edgecolor": "none", "alpha": 0.78},
    )


def build_figure(
    case_root: Path,
    results_path: Path,
    roi_path: Path,
    output_dir: Path,
) -> None:
    results = read_csv_index(results_path, ("patient_id", "reference_case_id", "moving_case_id"))
    rois = read_csv_index(roi_path, ("patient_id", "case_id"))

    configure_style()
    width_inches = 183.0 / 25.4
    height_inches = 113.0 / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(width_inches, height_inches), constrained_layout=False)

    source_rows: list[dict[str, str | int]] = []
    for row_index, example in enumerate(EXAMPLES):
        key = (example.patient_id, example.reference_case_id, example.moving_case_id)
        result = results[key]
        reference_roi = rois[(example.patient_id, example.reference_case_id)]
        moving_roi = rois[(example.patient_id, example.moving_case_id)]
        reference = load_volume(case_root, example.patient_id, example.reference_case_id)
        moving = load_volume(case_root, example.patient_id, example.moving_case_id)
        if reference.shape != moving.shape:
            raise ValueError(f"Shape mismatch for {key}: {reference.shape} vs {moving.shape}")

        crop = union_roi(reference_roi, moving_roi, reference.shape)
        reference_mask_3d = bone_mask(reference, crop)
        moving_mask_3d = bone_mask(moving, crop)
        slice_index = example.fixed_slice
        reference_window = window_volume(reference)
        moving_window = window_volume(moving)
        reference_image, reference_mask = extract_plane(
            reference_window,
            reference_mask_3d,
            crop,
            example.plane,
            slice_index,
        )
        moving_image, moving_mask = extract_plane(
            moving_window,
            moving_mask_3d,
            crop,
            example.plane,
            slice_index,
        )

        draw_image(axes[row_index, 0], reference_image)
        draw_image(axes[row_index, 1], moving_image)
        draw_overlay(
            axes[row_index, 2],
            reference_image,
            reference_mask,
            moving_mask,
            example,
            result,
        )

        axes[row_index, 0].text(
            -0.14,
            1.055,
            chr(ord("a") + row_index),
            transform=axes[row_index, 0].transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            color=TEXT_COLOR,
        )
        axes[row_index, 0].text(
            0.0,
            1.055,
            example.title,
            transform=axes[row_index, 0].transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="top",
            color=TEXT_COLOR,
        )

        source_rows.append(
            {
                "figure_label": example.label,
                "patient_id": example.patient_id,
                "reference_case_id": example.reference_case_id,
                "moving_case_id": example.moving_case_id,
                "plane": example.plane,
                "slice_index_zyx": slice_index,
                "setup_error_SI_mm": result["setup_error_SI_mm"],
                "setup_error_yaw_deg": result["setup_error_yaw_deg"],
                "similarity_score": result["similarity_score"],
                "window_low_HU": -200,
                "window_high_HU": 1700,
            }
        )

    for x, title in zip(
        (0.20, 0.52, 0.84),
        ("Reference CBCT", "Treatment-day CBCT", "Unregistered contour overlay"),
    ):
        fig.text(x, 0.965, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=TEXT_COLOR)

    handles = [
        mpl.lines.Line2D([0], [0], color=REFERENCE_COLOR, lw=1.5, linestyle="solid", label="Reference bone contour"),
        mpl.lines.Line2D([0], [0], color=MOVING_COLOR, lw=1.5, linestyle="dashed", label="Treatment-day bone contour"),
        mpl.lines.Line2D([0], [0], color=ARROW_COLOR, lw=1.5, linestyle="solid", label="Deviation direction"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        fontsize=6.8,
        handlelength=2.6,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.09, wspace=0.045, hspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig7_real_positioning_deviation_examples"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_path = output_dir / "fig7_source_data.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Fig. 7 examples of real inter-fraction positioning deviation.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--registration-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--roi-reviewed", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    build_figure(
        case_root=args.case_root,
        results_path=args.registration_results,
        roi_path=args.roi_reviewed,
        output_dir=args.output_dir,
    )
    print(f"Wrote Fig. 7 image bundle to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
