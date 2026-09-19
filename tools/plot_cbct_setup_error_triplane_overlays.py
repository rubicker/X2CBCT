from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_error_manifest import read_pair_manifest
from x2cbct_difnet.setup_error_registration import load_roi_review
from x2cbct_difnet.setup_error_roi import load_cbct, roi_mask_from_candidate


DEFAULT_PAIRS = ["24:2:8", "24:2:9"]


def parse_pair(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pair must use patient:reference:moving, e.g. 10:0:4")
    return parts[0], parts[1], parts[2]


def read_results(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["patient_id"]), str(row["reference_case_id"]), str(row["moving_case_id"]))
            rows[key] = row
    return rows


def find_pair(manifest: dict[str, Any], key: tuple[str, str, str]) -> dict[str, Any]:
    for pair in manifest["pairs"]:
        current = (str(pair["patient_id"]), str(pair["reference_case_id"]), str(pair["moving_case_id"]))
        if current == key:
            return pair
    raise KeyError(f"pair not found in manifest: {key}")


def union_crop(row_a: dict[str, Any], row_b: dict[str, Any], shape: tuple[int, int, int], margin: int) -> tuple[slice, slice, slice]:
    z_size, y_size, x_size = shape
    x0 = max(0, min(int(row_a["x_min"]), int(row_b["x_min"])) - margin)
    x1 = min(x_size - 1, max(int(row_a["x_max"]), int(row_b["x_max"])) + margin)
    y0 = max(0, min(int(row_a["y_min"]), int(row_b["y_min"])) - margin)
    y1 = min(y_size - 1, max(int(row_a["y_max"]), int(row_b["y_max"])) + margin)
    z0 = max(0, min(int(row_a["z_min"]), int(row_b["z_min"])) - margin)
    z1 = min(z_size - 1, max(int(row_a["z_max"]), int(row_b["z_max"])) + margin)
    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def crop_center(row_a: dict[str, Any], row_b: dict[str, Any], crop: tuple[slice, slice, slice]) -> tuple[int, int, int]:
    z_slice, y_slice, x_slice = crop
    x_center = round((int(row_a["x_min"]) + int(row_a["x_max"]) + int(row_b["x_min"]) + int(row_b["x_max"])) / 4)
    y_center = round((int(row_a["y_min"]) + int(row_a["y_max"]) + int(row_b["y_min"]) + int(row_b["y_max"])) / 4)
    z_center = round((int(row_a["z_min"]) + int(row_a["z_max"]) + int(row_b["z_min"]) + int(row_b["z_max"])) / 4)
    return z_center - z_slice.start, y_center - y_slice.start, x_center - x_slice.start


def window_image(array: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(array.astype(np.float32), low, high)
    return (clipped - low) / max(high - low, 1e-6)


def plane_data(volume: np.ndarray, mask: np.ndarray, plane: str, center: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    zc, yc, xc = center
    if plane == "axial":
        return volume[zc, :, :], mask[zc, :, :]
    if plane == "coronal":
        return volume[:, yc, :], mask[:, yc, :]
    if plane == "sagittal":
        return volume[:, :, xc], mask[:, :, xc]
    raise ValueError(f"unknown plane: {plane}")


def draw_panel(
    ax: plt.Axes,
    ref_window: np.ndarray,
    ref_mask: np.ndarray,
    mov_mask: np.ndarray,
    plane: str,
    center: tuple[int, int, int],
) -> None:
    image, ref_slice = plane_data(ref_window, ref_mask, plane, center)
    _, mov_slice = plane_data(ref_window, mov_mask, plane, center)
    ax.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    draw_contour(ax, ref_slice, "#00D1FF")
    draw_contour(ax, mov_slice, "#FF2E88")
    ax.set_title(plane.capitalize(), fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_contour(ax: plt.Axes, mask: np.ndarray, color: str) -> None:
    if mask.any() and np.logical_not(mask).any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.1)


def mask_center(mask: np.ndarray, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not mask.any():
        return fallback
    zz, yy, xx = np.where(mask)
    return int(round(float(np.median(zz)))), int(round(float(np.median(yy)))), int(round(float(np.median(xx))))


def metric_text(row: dict[str, str]) -> str:
    si = float(row.get("setup_error_SI_mm", row["SI_mm"]))
    yaw = float(row.get("setup_error_yaw_deg", row["yaw_deg"]))
    return (
        f"SI={si:+.2f} mm (abs {abs(si):.2f}); "
        f"yaw={yaw:+.2f} deg (abs {abs(yaw):.2f})"
    )


def build_pair_payload(
    pair: dict[str, Any],
    key: tuple[str, str, str],
    roi_by_case: dict[tuple[str, str], dict[str, Any]],
    result_row: dict[str, str],
    margin: int,
    window: tuple[float, float],
) -> dict[str, Any]:
    patient_id, ref_case, mov_case = key
    ref_roi = roi_by_case[(patient_id, ref_case)]
    mov_roi = roi_by_case[(patient_id, mov_case)]
    ref_path = Path(pair["reference_cbct_path"])
    mov_path = Path(pair["moving_cbct_path"])
    if not ref_path.exists() or not mov_path.exists():
        raise FileNotFoundError(f"CBCT file missing for {key}: {ref_path} / {mov_path}")
    _, ref_array = load_cbct(ref_path)
    _, mov_array = load_cbct(mov_path)
    if ref_array.shape != mov_array.shape:
        raise ValueError(f"shape mismatch for {key}: {ref_array.shape} vs {mov_array.shape}")
    crop = union_crop(ref_roi, mov_roi, ref_array.shape, margin)
    center = crop_center(ref_roi, mov_roi, crop)
    ref_crop = ref_array[crop]
    ref_mask = roi_mask_from_candidate(ref_array, ref_roi)[crop]
    mov_mask = roi_mask_from_candidate(mov_array, mov_roi)[crop]
    low, high = window
    return {
        "key": key,
        "ref_window": window_image(ref_crop, low, high),
        "ref_mask": ref_mask,
        "mov_mask": mov_mask,
        "center": mask_center(ref_mask | mov_mask, center),
        "metrics": result_row,
    }


def plot_single(payload: dict[str, Any], output_path: Path) -> None:
    patient_id, ref_case, mov_case = payload["key"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), constrained_layout=False)
    for ax, plane in zip(axes, ["axial", "coronal", "sagittal"]):
        draw_panel(ax, payload["ref_window"], payload["ref_mask"], payload["mov_mask"], plane, payload["center"])
    fig.suptitle(f"Patient {patient_id}: raw CBCT {ref_case} vs {mov_case}", fontsize=11, y=0.97)
    fig.text(0.5, 0.885, metric_text(payload["metrics"]), ha="center", va="center", fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.16, wspace=0.08)
    add_legend(fig, y=0.055)
    save_figure(fig, output_path)


def add_legend(fig: plt.Figure, y: float) -> None:
    handles = [
        plt.Line2D([0], [0], color="#00D1FF", lw=2, label="Reference bone edge"),
        plt.Line2D([0], [0], color="#FF2E88", lw=2, label="Moving bone edge"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=8)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined(payloads: list[dict[str, Any]], output_path: Path) -> None:
    width_ratios = [1.45, 1.0, 1.0, 1.0]
    fig, axes = plt.subplots(
        len(payloads),
        4,
        figsize=(11.4, 3.0 * len(payloads)),
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=False,
    )
    if len(payloads) == 1:
        axes = np.asarray([axes])
    for row_idx, payload in enumerate(payloads):
        patient_id, ref_case, mov_case = payload["key"]
        text_ax = axes[row_idx, 0]
        text_ax.axis("off")
        text_ax.text(
            0.98,
            0.5,
            f"Patient {patient_id}\nCBCT {ref_case} vs {mov_case}\n\n{metric_text(payload['metrics'])}",
            ha="right",
            va="center",
            fontsize=8.4,
            wrap=True,
        )
        for col_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
            draw_panel(
                axes[row_idx, col_idx + 1],
                payload["ref_window"],
                payload["ref_mask"],
                payload["mov_mask"],
                plane,
                payload["center"],
            )
    fig.suptitle("Raw unaligned CBCT triplane overlays", fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.93, bottom=0.08, hspace=0.42, wspace=0.08)
    add_legend(fig, y=0.012)
    save_figure(fig, output_path)


def write_summary(payloads: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "patient_id",
        "reference_case_id",
        "moving_case_id",
        "setup_error_SI_mm",
        "setup_error_abs_SI_mm",
        "setup_error_yaw_deg",
        "setup_error_abs_yaw_deg",
        "LR_mm",
        "AP_mm",
        "SI_mm",
        "translation_norm_mm",
        "pitch_deg",
        "roll_deg",
        "yaw_deg",
        "rotation_norm_deg",
        "similarity_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            row = payload["metrics"]
            exported = {field: row.get(field, "") for field in fields}
            exported["setup_error_SI_mm"] = row.get("setup_error_SI_mm", row["SI_mm"])
            exported["setup_error_abs_SI_mm"] = row.get("setup_error_abs_SI_mm", str(abs(float(row["SI_mm"]))))
            exported["setup_error_yaw_deg"] = row.get("setup_error_yaw_deg", row["yaw_deg"])
            exported["setup_error_abs_yaw_deg"] = row.get("setup_error_abs_yaw_deg", str(abs(float(row["yaw_deg"]))))
            writer.writerow(exported)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot raw unaligned CBCT triplane overlays for setup-error pairs.")
    parser.add_argument("--pair-manifest", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "pair_manifest" / "pair_manifest.json")
    parser.add_argument("--roi-reviewed", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "cervical_roi_review" / "roi_reviewed.csv")
    parser.add_argument("--registration-results", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration_si_yaw_fixed" / "registration_results.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "paper_figures" / "triplane_raw_overlay")
    parser.add_argument("--pair", type=parse_pair, action="append", dest="pairs", help="patient:reference:moving. Can be repeated.")
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--window-low", type=float, default=-200.0)
    parser.add_argument("--window-high", type=float, default=1700.0)
    args = parser.parse_args()

    target_pairs = args.pairs if args.pairs else [parse_pair(value) for value in DEFAULT_PAIRS]
    manifest = read_pair_manifest(args.pair_manifest)
    roi_by_case = load_roi_review(args.roi_reviewed)
    results = read_results(args.registration_results)
    payloads = []
    for key in target_pairs:
        pair = find_pair(manifest, key)
        result_row = results[key]
        payload = build_pair_payload(pair, key, roi_by_case, result_row, args.margin, (args.window_low, args.window_high))
        payloads.append(payload)
        patient_id, ref_case, mov_case = key
        stem = f"patient_{patient_id}_case_{ref_case}_vs_{mov_case}_raw_triplane_overlay"
        plot_single(payload, args.output_dir / stem)
    plot_combined(payloads, args.output_dir / "setup_error_top_pairs_raw_triplane_overlay")
    write_summary(payloads, args.output_dir / "figure_pair_metrics.csv")
    print(f"wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
