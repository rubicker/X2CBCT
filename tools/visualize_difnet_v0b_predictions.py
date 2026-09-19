from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.volume_grid import plane_spacing, resize_plane_to_spacing


DISPLAY_WINDOW_PERCENTILE = 99.5
ERROR_WINDOW_PERCENTILE = 99.0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _slice(volume: np.ndarray, mode: str, index: int) -> np.ndarray:
    if mode == "axial":
        return volume[index, :, :]
    if mode == "coronal":
        return volume[:, index, :]
    if mode == "sagittal":
        return volume[:, :, index]
    raise ValueError(mode)


def _indices(shape: tuple[int, int, int], mode: str, num_slices: int) -> list[int]:
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[mode]
    size = shape[axis]
    if num_slices <= 1:
        return [size // 2]
    return [int(round(item)) for item in np.linspace(0, size - 1, num_slices + 2)[1:-1]]


def _display_window(array: np.ndarray, error: bool = False) -> dict:
    arr = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return {"low": 0.0, "high": 1.0, "percentile": ERROR_WINDOW_PERCENTILE if error else DISPLAY_WINDOW_PERCENTILE}
    low = 0.0
    percentile = ERROR_WINDOW_PERCENTILE if error else DISPLAY_WINDOW_PERCENTILE
    high = float(np.percentile(values, percentile))
    if high <= low + 1e-8:
        high = float(np.max(values))
    if high <= low + 1e-8:
        high = 1.0
    return {"low": low, "high": high, "percentile": percentile}


def _shared_intensity_window(target: np.ndarray, generated: np.ndarray) -> dict:
    target_window = _display_window(target, error=False)
    generated_window = _display_window(generated, error=False)
    return {
        "low": 0.0,
        "high": max(float(target_window["high"]), float(generated_window["high"])),
        "percentile": DISPLAY_WINDOW_PERCENTILE,
    }


def _to_panel(array: np.ndarray, display_window: dict | None = None, error: bool = False) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    window = display_window or _display_window(arr, error=error)
    low = float(window["low"])
    high = float(window["high"])
    denom = max(high - low, 1e-8)
    gray = np.clip((arr - low) / denom, 0.0, 1.0)
    if error:
        panel = cv2.applyColorMap((gray * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    else:
        panel = cv2.cvtColor((gray * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return panel


def _with_label_bar(panel: np.ndarray, label: str) -> np.ndarray:
    label_bar = np.zeros((24, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_bar, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.vconcat([label_bar, panel])


def _plot_case(
    target: np.ndarray,
    generated: np.ndarray,
    output: Path,
    title: str,
    intensity_window: dict,
    *,
    plane: str,
    spacing_xyz: list[float],
) -> None:
    error = np.abs(generated - target)
    error_window = _display_window(error, error=True)
    panel_items = [
        ("target", resize_plane_to_spacing(_to_panel(target, display_window=intensity_window), plane, spacing_xyz)),
        ("generated", resize_plane_to_spacing(_to_panel(generated, display_window=intensity_window), plane, spacing_xyz)),
        ("abs error", resize_plane_to_spacing(_to_panel(error, display_window=error_window, error=True), plane, spacing_xyz)),
    ]
    height = max(panel.shape[0] for _, panel in panel_items)
    normalized = []
    for label, panel in panel_items:
        if panel.shape[0] != height:
            scale = height / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        normalized.append(_with_label_bar(panel, label))
    image = cv2.hconcat(normalized)
    title_bar = np.zeros((42, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title[:150], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    image = cv2.vconcat([title_bar, image])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DIF-Net V0b generated CBCT predictions.")
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--slice-mode", choices=("axial", "coronal", "sagittal", "all"), default="all")
    parser.add_argument("--num-slices", type=int, default=1)
    parser.add_argument("--window", choices=("norm", "hu"), default="norm")
    args = parser.parse_args()

    manifest = _read_json(args.prediction_manifest)
    out_dir = args.output_dir or Path(args.prediction_manifest).parent / "comparisons"
    modes = ("axial", "coronal", "sagittal") if args.slice_mode == "all" else (args.slice_mode,)
    records = []
    predictions = manifest["predictions"][: args.max_cases] if args.max_cases is not None else manifest["predictions"]
    for prediction in predictions:
        generated = np.load(prediction["generated_cbct_path"]).astype(np.float32)
        target = np.load(prediction["target_cbct_path"]).astype(np.float32)
        intensity_window = _shared_intensity_window(target, generated)
        case_key = prediction["case_key"]
        spacing_xyz = [float(item) for item in prediction.get("processed_spacing_xyz") or [1.0, 1.0, 1.0]]
        case_records = []
        for mode in modes:
            row_spacing, column_spacing = plane_spacing(mode, spacing_xyz)
            for index in _indices(target.shape, mode, args.num_slices):
                filename = f"{case_key}_{mode}_{index:03d}.png"
                title = (
                    f"{case_key} view={prediction['view_count']} "
                    f"ckpt={Path(prediction['checkpoint_path']).name} "
                    f"MAE={prediction['metrics']['mae_norm']:.4f} "
                    f"PSNR={prediction['metrics']['psnr']:.2f} "
                    f"SSIM={prediction['metrics']['ssim']:.4f}"
                )
                output = out_dir / case_key / filename
                _plot_case(
                    _slice(target, mode, index),
                    _slice(generated, mode, index),
                    output,
                    title,
                    intensity_window,
                    plane=mode,
                    spacing_xyz=spacing_xyz,
                )
                case_records.append(
                    {
                        "mode": mode,
                        "slice_index": index,
                        "path": str(output),
                        "display_window": intensity_window,
                        "row_spacing_mm": row_spacing,
                        "column_spacing_mm": column_spacing,
                    }
                )
        records.append(
            {
                "case_key": case_key,
                "geometry_branch": prediction.get("geometry_branch"),
                "processed_spacing_xyz": spacing_xyz,
                "comparisons": case_records,
            }
        )
    summary = {
        "ok": True,
        "prediction_manifest": str(args.prediction_manifest),
        "output_dir": str(out_dir),
        "display_policy": "spacing_corrected_by_processed_spacing_xyz",
        "case_count": len(records),
        "records": records,
    }
    summary_path = out_dir / "visualization_summary.json"
    _write_json(summary, summary_path)
    print(json.dumps({"ok": True, "summary": str(summary_path), "case_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
