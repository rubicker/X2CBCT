from __future__ import annotations

import argparse
import json
import math
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_deepsparse import DeepSparseCachedDataset, build_deepsparse_model
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


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return raw


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        value = batch[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if key.endswith("mask"):
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        out[key] = tensor.to(device, non_blocking=device.type == "cuda")
    return out


def _task_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Supervise every sampled query point, including all-view-invisible points."""

    return torch.nn.functional.mse_loss(prediction, target)


def _visible_point_ratio(valid_mask: torch.Tensor) -> float:
    """Report coverage without using it to restrict the task loss."""

    return float(valid_mask.any(dim=1).float().mean().item())


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is only enabled for CUDA runs")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": config,
        },
        path,
    )


def _evaluate(
    *,
    model: torch.nn.Module,
    dataset: DeepSparseCachedDataset,
    device: torch.device,
    chunk_size: int,
    output_dir: Path,
    save_volumes: bool,
    amp: bool,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    normalization = dataset.cache_manifest.get("cbct_normalization")
    rows: list[dict[str, Any]] = []
    started = time.time()
    evaluation_indices = list(range(len(dataset))) if indices is None else [int(index) for index in indices]
    invalid_indices = [index for index in evaluation_indices if index < 0 or index >= len(dataset)]
    if invalid_indices:
        raise IndexError(
            f"evaluation indices out of range for dataset of size {len(dataset)}: "
            f"{invalid_indices}"
        )
    with torch.no_grad():
        for position, index in enumerate(evaluation_indices, start=1):
            item = dataset[index]
            batch = {key: np.expand_dims(item[key], axis=0) for key in TENSOR_KEYS}
            tensors = _to_device(batch, device)
            with _autocast(device, amp):
                result = model(tensors, is_eval=True, eval_npoint=int(chunk_size))
            shape = tuple(int(value) for value in item["image"].shape)
            generated = result["points_pred"][0, 0].float().cpu().numpy().reshape(shape)
            generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
            target = np.asarray(item["image"], dtype=np.float32)
            metrics = metric_dict(generated, target, normalization=normalization)
            trace = item["trace"]
            row = {
                "index": index,
                "name": str(item["name"]),
                "patient_id": trace.get("patient_id"),
                "case_id": trace.get("case_id"),
                "case_key": trace.get("case_key"),
                **metrics,
            }
            rows.append(row)
            _append_jsonl(output_dir / "evaluation_cases.jsonl", row)

            if save_volumes:
                volume_dir = output_dir / "generated_cbct"
                volume_dir.mkdir(parents=True, exist_ok=True)
                np.save(volume_dir / f"{trace.get('case_key', index)}.npy", generated)

            elapsed = max(time.time() - started, 1e-6)
            print(
                f"eval {position}/{len(evaluation_indices)} dataset_index={index} "
                f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim'] * 100.0:.4f}% "
                f"rate={position / elapsed:.3f} case/s",
                flush=True,
            )

    return {
        "case_count": len(rows),
        "mean": mean_metrics(rows),
        "cases": rows,
        "elapsed_seconds": time.time() - started,
    }


def _resolved_config(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(raw))
    data = cfg.setdefault("dataset", {})
    training = cfg.setdefault("training", {})
    evaluation = cfg.setdefault("evaluation", {})
    runtime = cfg.setdefault("runtime", {})

    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.num_points is not None:
        training["num_points"] = int(args.num_points)
    if args.max_train_cases is not None:
        data["max_train_cases"] = int(args.max_train_cases)
    if args.max_test_cases is not None:
        data["max_test_cases"] = int(args.max_test_cases)
    if args.device is not None:
        runtime["device"] = args.device
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.resume_checkpoint is not None:
        training["resume_checkpoint"] = args.resume_checkpoint
    if args.eval_grid_size is not None:
        evaluation["grid_size"] = int(args.eval_grid_size)
    if args.eval_chunk_size is not None:
        evaluation["chunk_size"] = int(args.eval_chunk_size)
    if args.eval_indices is not None:
        evaluation["indices"] = [int(index) for index in args.eval_indices]
    if args.save_volumes is not None:
        evaluation["save_volumes"] = bool(args.save_volumes)

    if args.smoke:
        training["epochs"] = 1
        training["batch_size"] = 1
        training["num_points"] = 128
        training["checkpoint_interval"] = 1
        data["max_train_cases"] = 1
        data["max_test_cases"] = 1
        evaluation["grid_size"] = 16
        evaluation["chunk_size"] = 4096
        evaluation["save_volumes"] = False
        cfg["output_dir"] = str(Path(".tmp") / "deepsparse_smoke" / Path(args.config).stem)

    required = {
        "dataset.train_cache": data.get("train_cache"),
        "dataset.test_cache": data.get("test_cache"),
        "dataset.view_count": data.get("view_count"),
        "training.epochs": training.get("epochs"),
        "output_dir": cfg.get("output_dir"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required configuration values: {', '.join(missing)}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeepSparse on X2CBCT fixed-view caches")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-points", type=int)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--eval-grid-size", type=int)
    parser.add_argument("--eval-chunk-size", type=int)
    parser.add_argument("--eval-indices", nargs="+", type=int)
    parser.add_argument("--save-volumes", action="store_true", default=None)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = _path(args.config)
    cfg = _resolved_config(_read_yaml(config_path), args)
    data_cfg = cfg["dataset"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    runtime_cfg = cfg.get("runtime", {})
    output_dir = _path(cfg["output_dir"])
    resume_requested = bool(cfg.get("training", {}).get("resume_checkpoint"))
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_requested:
        raise FileExistsError(f"output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(runtime_cfg.get("seed", 20260721))
    _seed_everything(seed)
    device = _device(str(runtime_cfg.get("device", "auto")))
    amp = bool(runtime_cfg.get("mixed_precision", True))
    if device.type != "cuda":
        amp = False

    view_count = int(data_cfg["view_count"])
    train_dataset = DeepSparseCachedDataset(
        _path(data_cfg["train_cache"]),
        split=str(data_cfg.get("train_split", "train")),
        view_count=view_count,
        num_points=int(train_cfg.get("num_points", 10_000)),
        lr_res=int(model_cfg.get("lr_res", 32)),
        max_cases=data_cfg.get("max_train_cases"),
        seed=seed,
        deterministic=False,
        projection_multiplier=float(data_cfg.get("projection_multiplier", 1.0)),
    )
    test_dataset = DeepSparseCachedDataset(
        _path(data_cfg["test_cache"]),
        split=str(data_cfg.get("test_split", "test")),
        view_count=view_count,
        num_points=int(train_cfg.get("num_points", 10_000)),
        lr_res=int(model_cfg.get("lr_res", 32)),
        max_cases=data_cfg.get("max_test_cases"),
        seed=seed,
        deterministic=True,
        eval_grid_size=int(eval_cfg.get("grid_size", 128)),
        projection_multiplier=float(data_cfg.get("projection_multiplier", 1.0)),
    )

    checkpoint_path = model_cfg.get("pretrained_checkpoint")
    model = build_deepsparse_model(
        PROJECT_ROOT,
        config_path=_path(model_cfg["official_config"]) if model_cfg.get("official_config") else None,
        checkpoint_path=_path(checkpoint_path) if checkpoint_path else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    start_epoch = 1
    resume_path = train_cfg.get("resume_checkpoint")
    if resume_path:
        resume = torch.load(str(_path(resume_path)), map_location="cpu")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"]) + 1

    resolved = {
        **cfg,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "dataset": {
            **data_cfg,
            "train_cache": str(_path(data_cfg["train_cache"])),
            "test_cache": str(_path(data_cfg["test_cache"])),
            "train_case_count": len(train_dataset),
            "test_case_count": len(test_dataset),
            "train_cache_version": train_dataset.cache_manifest.get("cache_version"),
            "test_cache_version": test_dataset.cache_manifest.get("cache_version"),
            "cbct_normalization": train_dataset.cache_manifest.get("cbct_normalization"),
        },
        "model": {
            **model_cfg,
            "pretrained_checkpoint": str(_path(checkpoint_path)) if checkpoint_path else None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "runtime": {
            **runtime_cfg,
            "device_resolved": str(device),
            "mixed_precision_resolved": amp,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": seed,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
        "protocol": {
            "fixed_view_count_only": True,
            "official_downstream_weights_used": False,
            "official_stage2_used": False,
            "task_loss_scope": "all_sampled_points",
            "epochs": int(train_cfg["epochs"]),
            "evaluation_grid_size": int(eval_cfg.get("grid_size", 128)),
        },
    }
    _write_json(output_dir / "config_resolved.json", resolved)

    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 1)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(runtime_cfg.get("num_workers", 1)) > 0,
    )
    epochs = int(train_cfg["epochs"])
    vq_weight = float(train_cfg.get("vq_weight", 0.1))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 50))
    events_path = output_dir / "events.jsonl"
    run_started = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_started = time.time()
        total_losses: list[float] = []
        task_losses: list[float] = []
        vq_losses: list[float] = []
        visible_ratios: list[float] = []
        for batch in loader:
            tensors = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                result = model(tensors)
                task_loss = _task_loss(result["points_pred"], tensors["points_gt"])
                visible_ratio = _visible_point_ratio(tensors["points_valid_mask"])
                vq_loss = result["loss_vq"]
                loss = task_loss + vq_weight * vq_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: {float(loss.detach())}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            total_losses.append(float(loss.detach().item()))
            task_losses.append(float(task_loss.detach().item()))
            vq_losses.append(float(vq_loss.detach().item()))
            visible_ratios.append(visible_ratio)

        event = {
            "event": "epoch_complete",
            "epoch": epoch,
            "loss": float(np.mean(total_losses)),
            "task_loss": float(np.mean(task_losses)),
            "vq_loss": float(np.mean(vq_losses)),
            "visible_point_ratio": float(np.mean(visible_ratios)),
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - run_started,
        }
        _append_jsonl(events_path, event)
        print(
            f"epoch {epoch}/{epochs} loss={event['loss']:.6f} "
            f"task={event['task_loss']:.6f} vq={event['vq_loss']:.6f} "
            f"visible={event['visible_point_ratio']:.4f} time={event['epoch_seconds']:.1f}s",
            flush=True,
        )
        if checkpoint_interval > 0 and (epoch % checkpoint_interval == 0 or epoch == epochs):
            _save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=resolved,
            )

    metrics = _evaluate(
        model=model,
        dataset=test_dataset,
        device=device,
        chunk_size=int(eval_cfg.get("chunk_size", 20_000)),
        output_dir=output_dir,
        save_volumes=bool(eval_cfg.get("save_volumes", False)),
        amp=amp,
        indices=eval_cfg.get("indices"),
    )
    metrics.update(
        {
            "dataset": cfg.get("experiment", config_path.stem),
            "view_count": view_count,
            "checkpoint": str(output_dir / f"checkpoint_epoch_{epochs:03d}.pt"),
            "total_elapsed_seconds": time.time() - run_started,
        }
    )
    _write_json(output_dir / "metrics_final.json", metrics)
    mean = metrics["mean"]
    print(
        f"complete: PSNR={mean['psnr']:.4f}, SSIM={mean['ssim'] * 100.0:.4f}%, "
        f"cases={metrics['case_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
