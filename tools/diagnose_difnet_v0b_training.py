from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest
from x2cbct_difnet.metrics import metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


DEFAULT_DIAGNOSIS_EVAL_GRID_SIZE = 128


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _finite_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    values = arr[finite]
    if values.size == 0:
        return {"shape": list(arr.shape), "finite_ratio": 0.0}
    return {
        "shape": list(arr.shape),
        "finite_ratio": float(finite.mean()),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "zero_fraction": float((values == 0).mean()),
        "one_fraction": float((values == 1).mean()),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _trend(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0}
    head = finite[: min(5, len(finite))]
    tail = finite[-min(5, len(finite)) :]
    head_mean = float(np.mean(head))
    tail_mean = float(np.mean(tail))
    return {
        "count": len(finite),
        "first": finite[0],
        "last": finite[-1],
        "head_mean": head_mean,
        "tail_mean": tail_mean,
        "relative_change": float((tail_mean - head_mean) / max(abs(head_mean), 1e-8)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    train_rows = [row for row in _read_jsonl(run_dir / "train_log.jsonl") if row.get("event") == "epoch_end"]
    event_rows = _read_jsonl(run_dir / "events.jsonl")
    validations = [row for row in event_rows if row.get("event") == "validation_end"]
    latest_step = next((row for row in reversed(event_rows) if row.get("event") == "train_step"), None)
    train_losses = [float(row["train_loss"]) for row in train_rows if "train_loss" in row]
    return {
        "run_dir": str(run_dir),
        "config": _read_json(run_dir / "config_resolved.json") if (run_dir / "config_resolved.json").exists() else None,
        "train_epoch_count": len(train_rows),
        "latest_epoch": train_rows[-1]["epoch"] if train_rows else None,
        "latest_step": latest_step,
        "train_loss_trend": _trend(train_losses),
        "train_log_csv_exists": (run_dir / "train_log.csv").exists(),
        "validation_count": len(validations),
        "validations": [
            {
                "epoch": row.get("epoch"),
                "kind": row.get("kind"),
                "metrics_summary": row.get("metrics_summary"),
                "output_dir": row.get("output_dir"),
            }
            for row in validations
        ],
    }


def summarize_dataset(
    cache_root: Path,
    split: str,
    view_key: str,
    max_cases: int,
    num_points: int,
    eval_grid_size: int | None,
) -> dict[str, Any]:
    dataset = CachedDifNetV0bDataset(
        cache_root,
        split=split,
        view_key=view_key,
        num_points=num_points,
        max_cases=max_cases,
        deterministic=True,
        eval_grid_size=eval_grid_size,
    )
    rows = []
    target_values = []
    valid_values = []
    for index in range(len(dataset)):
        item = dataset[index]
        p_gt = np.asarray(item["p_gt"], dtype=np.float32)
        valid_mask = np.asarray(item["valid_mask"], dtype=bool)
        target_values.append(p_gt.reshape(-1))
        valid_values.append(valid_mask.reshape(-1))
        rows.append(
            {
                "case_key": item["trace"]["case_key"],
                "p_gt": _finite_stats(p_gt),
                "projs": _finite_stats(np.asarray(item["projs"], dtype=np.float32)),
                "points": _finite_stats(np.asarray(item["points"], dtype=np.float32)),
                "proj_points": _finite_stats(np.asarray(item["proj_points"], dtype=np.float32)),
                "valid_ratio": float(valid_mask.mean()),
                "all_views_valid_ratio": float(valid_mask.all(axis=0).mean()),
            }
        )
    all_targets = np.concatenate(target_values) if target_values else np.asarray([], dtype=np.float32)
    all_valid = np.concatenate(valid_values) if valid_values else np.asarray([], dtype=bool)
    return {
        "split": split,
        "view_key": view_key,
        "sample_count": len(rows),
        "eval_grid_size": eval_grid_size,
        "target": _finite_stats(all_targets),
        "valid_ratio": float(all_valid.mean()) if all_valid.size else float("nan"),
        "cases": rows,
    }


def _load_prediction_rows(validation_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test_dir = validation_dir / "test"
    manifest = _read_json(test_dir / "prediction_manifest.json")
    return manifest, manifest.get("predictions", [])


def summarize_predictions(validation_dir: Path, train_target_mean: float | None) -> dict[str, Any]:
    manifest, rows = _load_prediction_rows(validation_dir)
    normalization = manifest.get("normalization")
    case_rows = []
    generated_values = []
    target_values = []
    for row in rows:
        generated = np.load(row["generated_cbct_path"]).astype(np.float32)
        target = np.load(row["target_cbct_path"]).astype(np.float32)
        generated_values.append(generated.reshape(-1))
        target_values.append(target.reshape(-1))
        target_mean = float(target.mean())
        baselines = {
            "constant_0": metric_dict(np.zeros_like(target), target, normalization=normalization),
            "constant_0_5": metric_dict(np.full_like(target, 0.5), target, normalization=normalization),
            "constant_case_mean": metric_dict(np.full_like(target, target_mean), target, normalization=normalization),
        }
        if train_target_mean is not None and math.isfinite(train_target_mean):
            baselines["constant_train_mean"] = metric_dict(np.full_like(target, train_target_mean), target, normalization=normalization)
        case_rows.append(
            {
                "case_key": row["case_key"],
                "recorded_metrics": row["metrics"],
                "generated": _finite_stats(generated),
                "target": _finite_stats(target),
                "generated_target_corr": _safe_corr(generated, target),
                "baselines": baselines,
            }
        )
    generated_all = np.concatenate(generated_values) if generated_values else np.asarray([], dtype=np.float32)
    target_all = np.concatenate(target_values) if target_values else np.asarray([], dtype=np.float32)
    return {
        "validation_dir": str(validation_dir),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "view_count": manifest.get("view_count"),
        "case_count": len(case_rows),
        "generated": _finite_stats(generated_all),
        "target": _finite_stats(target_all),
        "generated_target_corr": _safe_corr(generated_all, target_all),
        "cases": case_rows,
    }


def find_latest_validation(run_dir: Path) -> Path | None:
    validation_root = run_dir / "validation"
    if not validation_root.exists():
        return None
    candidates = [
        path
        for path in validation_root.iterdir()
        if path.is_dir() and (path / "test" / "prediction_manifest.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose DIF-Net V0b training usefulness and data integrity.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    run = summarize_run(args.run_dir)
    config = run.get("config") or {}
    cache_root = (args.cache_root or Path(config["cache_root"])).resolve()
    view_count = args.view_count or int(config["view_count"])
    view_key = view_key_from_count(view_count)
    cache_manifest = load_cache_manifest(cache_root)
    validation_dir = args.validation_dir or find_latest_validation(args.run_dir)

    train_dataset = summarize_dataset(cache_root, "train", view_key, args.max_cases, args.num_points, None)
    eval_grid_size = None
    if validation_dir is not None:
        eval_grid_size = int(config.get("formal_validation_eval_grid_size") or DEFAULT_DIAGNOSIS_EVAL_GRID_SIZE)
    test_dataset = summarize_dataset(cache_root, "test", view_key, args.max_cases, args.num_points, eval_grid_size)
    train_mean = train_dataset["target"].get("mean")
    predictions = summarize_predictions(validation_dir, train_mean) if validation_dir is not None else None
    report = {
        "schema": "x2cbct_difnet_v0b_training_diagnosis.v1",
        "run": run,
        "cache": {
            "cache_root": str(cache_root),
            "cache_version": cache_manifest["cache_version"],
            "view_key": view_key,
            "split": cache_manifest["split"],
            "cbct_normalization": cache_manifest["cbct_normalization"],
            "projection_normalization": cache_manifest["projection_normalization"],
            "point_sampling": cache_manifest["point_sampling"],
        },
        "dataset": {
            "train_sample": train_dataset,
            "test_eval_sample": test_dataset,
        },
        "predictions": predictions,
        "interpretation_hints": [
            "train_loss is point-wise MSE over random training blocks; compare it with validation volume metrics before calling the run useful",
            "constant_case_mean or constant_train_mean close to generated metrics means the checkpoint behaves like a mean-intensity predictor",
            "low valid_ratio means many supervised points query zero-padded 2D features and should be masked or resampled",
        ],
    }
    output_json = args.output_json or args.run_dir / "diagnosis" / "training_diagnosis.json"
    _write_json(report, output_json)
    print(json.dumps({"ok": True, "output_json": str(output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
