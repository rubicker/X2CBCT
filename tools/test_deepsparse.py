from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_deepsparse import DeepSparseCachedDataset, build_deepsparse_model
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS
from x2cbct_difnet.metrics import mean_metrics, metric_dict


TENSOR_KEYS = (
    "projs",
    "points_ct",
    "points_gt",
    "points_proj",
    "points_lr_proj",
    "view_mask",
    "points_valid_mask",
    "points_lr_valid_mask",
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is only supported on CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        value = batch[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.bool() if key.endswith("mask") else tensor.float()
        tensors[key] = tensor.to(device, non_blocking=device.type == "cuda")
    return tensors


def _safe_volume_name(item: dict[str, Any], index: int) -> str:
    trace = item["trace"]
    parts = [
        str(trace.get("case_key") or item.get("name") or index),
        str(trace.get("augmentation_id") or ""),
    ]
    value = "_".join(part for part in parts if part)
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _evaluate(
    *,
    model: torch.nn.Module,
    dataset: DeepSparseCachedDataset,
    device: torch.device,
    chunk_size: int,
    output_dir: Path,
    save_volumes: bool,
    mixed_precision: bool,
    indices: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_indices = list(range(len(dataset))) if indices is None else [int(index) for index in indices]
    invalid = [index for index in evaluation_indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise IndexError(f"evaluation indices out of range for dataset of size {len(dataset)}: {invalid}")

    model.eval()
    normalization = dataset.cache_manifest["cbct_normalization"]
    rows: list[dict[str, Any]] = []
    started = time.time()
    with torch.no_grad():
        for position, index in enumerate(evaluation_indices, start=1):
            item = dataset[index]
            batch = {key: np.expand_dims(item[key], axis=0) for key in TENSOR_KEYS}
            tensors = _to_device(batch, device)
            with _autocast(device, mixed_precision):
                result = model(tensors, is_eval=True, eval_npoint=int(chunk_size))
            shape = tuple(int(value) for value in item["image"].shape)
            generated = result["points_pred"][0, 0].float().cpu().numpy().reshape(shape)
            generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
            target = np.asarray(item["image"], dtype=np.float32)
            metrics = metric_dict(generated, target, normalization=normalization)
            trace = item["trace"]
            generated_path = None
            if save_volumes:
                generated_path = output_dir / "generated_cbct" / f"{_safe_volume_name(item, index)}.npy"
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(generated_path, generated)
            rows.append(
                {
                    "index": index,
                    "name": str(item["name"]),
                    "patient_id": trace.get("patient_id"),
                    "case_id": trace.get("case_id"),
                    "case_key": trace.get("case_key"),
                    "augmentation_id": trace.get("augmentation_id"),
                    "translation_mm": trace.get("translation_mm"),
                    "rotation_deg": trace.get("rotation_deg"),
                    "generated_cbct_path": str(generated_path) if generated_path is not None else None,
                    **metrics,
                }
            )
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"eval {position}/{len(evaluation_indices)} dataset_index={index} "
                f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim'] * 100.0:.4f}% "
                f"rate={position / elapsed:.3f} case/s",
                flush=True,
            )

    mean = mean_metrics(rows)
    summary = {
        **mean,
        "case_count": len(rows),
        "split": dataset.split,
        "view_count": dataset.view_count,
        "cache_version": dataset.cache_manifest.get("cache_version"),
        "geometry_branch": dataset.cache_manifest.get("geometry_branch"),
        "cbct_normalization": normalization,
        "elapsed_seconds": time.time() - started,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepSparse checkpoint on an X2CBCT cache.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--eval-grid-size", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--eval-indices", nargs="+", type=int)
    parser.add_argument("--lr-res", type=int)
    parser.add_argument("--projection-multiplier", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=None)
    volume_group = parser.add_mutually_exclusive_group()
    volume_group.add_argument("--save-volumes", dest="save_volumes", action="store_true")
    volume_group.add_argument("--no-save-volumes", dest="save_volumes", action="store_false")
    parser.set_defaults(save_volumes=False)
    args = parser.parse_args()

    checkpoint_path = _path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    file_config = _read_yaml(_path(args.config)) if args.config is not None else {}
    config = _deep_merge(checkpoint_config, file_config)
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})
    runtime_config = config.get("runtime", {})

    checkpoint_view_count = dataset_config.get("view_count")
    if checkpoint_view_count is not None and int(checkpoint_view_count) != args.view_count:
        raise ValueError(
            "checkpoint view_count does not match requested view_count: "
            f"{checkpoint_view_count} != {args.view_count}"
        )

    output_dir = _path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else runtime_config.get("seed", 20260721))
    _seed_everything(seed)
    device = _device(str(args.device or runtime_config.get("device", "auto")))
    mixed_precision = (
        bool(args.mixed_precision)
        if args.mixed_precision is not None
        else bool(runtime_config.get("mixed_precision", True))
    )
    if device.type != "cuda":
        mixed_precision = False
    eval_grid_size = int(
        args.eval_grid_size if args.eval_grid_size is not None else evaluation_config.get("grid_size", 128)
    )
    chunk_size = int(args.chunk_size if args.chunk_size is not None else evaluation_config.get("chunk_size", 20_000))
    lr_res = int(args.lr_res if args.lr_res is not None else model_config.get("lr_res", 32))
    projection_multiplier = float(
        args.projection_multiplier
        if args.projection_multiplier is not None
        else dataset_config.get("projection_multiplier", 1.0)
    )
    official_config = _path(
        args.official_config
        if args.official_config is not None
        else model_config.get("official_config", "external/deepsparse_official/configs/finetune_s1.yaml")
    )

    dataset = DeepSparseCachedDataset(
        _path(args.cache_root),
        split=args.split,
        view_count=args.view_count,
        num_points=int(training_config.get("num_points", 10_000)),
        lr_res=lr_res,
        max_cases=args.max_test_cases,
        seed=seed,
        deterministic=True,
        eval_grid_size=eval_grid_size,
        projection_multiplier=projection_multiplier,
    )
    model = build_deepsparse_model(PROJECT_ROOT, config_path=official_config).to(device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)

    resolved = {
        "config_path": str(_path(args.config)) if args.config is not None else None,
        "cache_root": str(_path(args.cache_root)),
        "split": args.split,
        "view_count": args.view_count,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "output_dir": str(output_dir),
        "official_config": str(official_config),
        "eval_grid_size": eval_grid_size,
        "chunk_size": chunk_size,
        "max_test_cases": args.max_test_cases,
        "eval_indices": args.eval_indices,
        "lr_res": lr_res,
        "projection_multiplier": projection_multiplier,
        "save_volumes": args.save_volumes,
        "runtime": {
            "device": str(device),
            "mixed_precision": mixed_precision,
            "seed": seed,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    _write_json(output_dir / "config_resolved.json", resolved)
    summary, rows = _evaluate(
        model=model,
        dataset=dataset,
        device=device,
        chunk_size=chunk_size,
        output_dir=output_dir,
        save_volumes=args.save_volumes,
        mixed_precision=mixed_precision,
        indices=args.eval_indices,
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["model"] = "deepsparse"
    _write_json(output_dir / "metrics_summary.json", summary)
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "schema": "x2cbct_deepsparse_evaluation.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": resolved,
            "cases": rows,
        },
    )
    with (output_dir / "metrics_per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "index",
            "name",
            "patient_id",
            "case_id",
            "case_key",
            "augmentation_id",
            "translation_mm",
            "rotation_deg",
            "mae_norm",
            "mae_hu",
            "psnr",
            "ssim",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "metrics_summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
