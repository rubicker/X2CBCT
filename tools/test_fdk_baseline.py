from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest
from x2cbct_difnet.fdk_baseline import reconstruct_fdk_item
from x2cbct_difnet.metrics import mean_metrics, metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count

from tools.test_difnet_v0b import (
    DEFAULT_TEST_EVAL_GRID_SIZE,
    DEFAULT_TEST_VIEWER_MAX_CASES,
    DEFAULT_TEST_VIEWER_SLICE_STRIDE,
    DEFAULT_TEST_VISUALIZATION_MAX_CASES,
    _export_visual_artifacts,
)


FDK_CHECKPOINT_LABEL = "FDK"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _safe_rmtree(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    target = path.resolve()
    if not str(target).lower().startswith(str(root).lower()):
        raise RuntimeError(f"refusing to remove outside workspace: {target}")
    if target in {root, root / "runs", root / "data_cache"}:
        raise RuntimeError(f"refusing to remove broad workspace path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output dir already exists and is not empty: {path}")
        _safe_rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _metrics_for_summary(metrics_rows: list[dict[str, Any]]) -> dict[str, float]:
    return mean_metrics(
        [
            {key: float(row[key]) for key in ("mae_norm", "mae_hu", "psnr", "ssim")}
            for row in metrics_rows
        ]
    )


def _match_target_shape(generated: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(int(item) for item in generated.shape) == target_shape:
        return np.asarray(generated, dtype=np.float32)
    factors = [target_shape[index] / generated.shape[index] for index in range(3)]
    return scipy.ndimage.zoom(generated.astype(np.float32), factors, order=1, prefilter=False).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TIGRE FDK baseline on a DIF-Net cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-grid-size", type=int, default=DEFAULT_TEST_EVAL_GRID_SIZE)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--comparisons-dir", type=Path)
    parser.add_argument("--viewer-dir", type=Path)
    parser.add_argument("--visualization-max-cases", type=int, default=DEFAULT_TEST_VISUALIZATION_MAX_CASES)
    parser.add_argument("--visualization-slice-mode", choices=("axial", "coronal", "sagittal", "all"), default="all")
    parser.add_argument("--visualization-num-slices", type=int, default=1)
    parser.add_argument("--viewer-max-cases", type=int, default=DEFAULT_TEST_VIEWER_MAX_CASES)
    parser.add_argument("--viewer-slice-stride", type=int, default=DEFAULT_TEST_VIEWER_SLICE_STRIDE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_manifest = load_cache_manifest(args.cache_root)
    cbct_normalization = cache_manifest["cbct_normalization"]
    view_key = view_key_from_count(args.view_count)
    out_dir = args.output_dir or PROJECT_ROOT / "runs" / "fdk" / f"test-{view_key}-{datetime.now():%Y%m%d-%H%M%S}"
    _prepare_output_dir(out_dir, overwrite=args.overwrite)

    dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split=args.split,
        view_key=view_key,
        max_cases=args.max_test_cases,
        deterministic=True,
        eval_grid_size=args.eval_grid_size,
    )

    prediction_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        reconstruction = reconstruct_fdk_item(item)
        target = np.asarray(item["image"], dtype=np.float32)
        generated = _match_target_shape(reconstruction.volume, tuple(int(value) for value in target.shape))
        generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
        metrics = metric_dict(generated, target, normalization=cbct_normalization)
        trace = item["trace"]
        case_key = trace["case_key"]
        generated_path = out_dir / "generated_cbct" / f"{case_key}_{view_key}.npy"
        target_path = out_dir / "target_cbct" / f"{case_key}_{view_key}.npy"
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(generated_path, generated)
        np.save(target_path, target)
        row = {
            "patient_id": trace["patient_id"],
            "case_id": trace["case_id"],
            "case_key": case_key,
            "augmentation_id": trace.get("augmentation_id"),
            "translation_mm": trace.get("translation_mm"),
            "rotation_deg": trace.get("rotation_deg"),
            "source_patient_id": trace.get("source_patient_id"),
            "source_case_id": trace.get("source_case_id"),
            "source_case_key": trace.get("source_case_key"),
            "view_count": args.view_count,
            "source_cache_item_path": trace["sample_path"],
            "generated_cbct_path": str(generated_path),
            "target_cbct_path": str(target_path),
            "geometry_branch": trace.get("geometry_branch", cache_manifest.get("geometry_branch")),
            "processed_shape_zyx": trace.get("processed_shape_zyx"),
            "processed_spacing_xyz": trace.get("processed_spacing_xyz"),
            "processed_origin_xyz": trace.get("processed_origin_xyz"),
            "processed_direction": trace.get("processed_direction"),
            "drr_config": trace.get("drr_config"),
            "normalization": cbct_normalization,
            "checkpoint_path": FDK_CHECKPOINT_LABEL,
            "model_version": "fdk",
                "model_config": {
                    "method": "FDK",
                    "implementation": "TIGRE",
                    "output_normalization": reconstruction.normalization,
                    "output_resampling": {
                        "source_shape": list(reconstruction.volume.shape),
                        "saved_shape": list(generated.shape),
                        "method": "scipy.ndimage.zoom_order1",
                    },
                    "geometry": reconstruction.geometry,
                },
            "eval_grid_size": args.eval_grid_size,
            "metrics": metrics,
        }
        prediction_rows.append(row)
        metrics_rows.append(
            {
                "case_key": case_key,
                "augmentation_id": trace.get("augmentation_id"),
                "translation_mm": trace.get("translation_mm"),
                "rotation_deg": trace.get("rotation_deg"),
                **metrics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "fdk_case_done",
                    "index": index + 1,
                    "total": len(dataset),
                    "case_key": case_key,
                    "view_count": args.view_count,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = _metrics_for_summary(metrics_rows)
    summary.update(
        {
            "case_count": len(metrics_rows),
            "split": args.split,
            "view_count": args.view_count,
            "cache_version": cache_manifest.get("cache_version"),
            "geometry_branch": cache_manifest.get("geometry_branch"),
            "method": "FDK",
            "implementation": "TIGRE",
        }
    )
    summary["cbct_normalization"] = cbct_normalization
    manifest = {
        "schema": "x2cbct_fdk_predictions.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "cache_version": cache_manifest.get("cache_version"),
        "geometry_branch": cache_manifest.get("geometry_branch"),
        "axis_convention": cache_manifest.get("axis_convention"),
        "checkpoint_path": FDK_CHECKPOINT_LABEL,
        "view_count": args.view_count,
        "view_key": view_key,
        "normalization": cbct_normalization,
        "method": "FDK",
        "implementation": "TIGRE",
        "predictions": prediction_rows,
    }
    _write_json(manifest, out_dir / "prediction_manifest.json")
    _write_json(summary, out_dir / "metrics_summary.json")
    with (out_dir / "metrics_per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_key",
                "augmentation_id",
                "translation_mm",
                "rotation_deg",
                "mae_norm",
                "mae_hu",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    visual_artifacts = {} if args.skip_visualization else _export_visual_artifacts(args, out_dir)
    print(json.dumps({"ok": True, "output_dir": str(out_dir), "metrics_summary": summary, **visual_artifacts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
