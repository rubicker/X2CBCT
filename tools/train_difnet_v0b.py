from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest
from x2cbct_difnet.capsule_routing import (
    CAPSULE_SQUASH_MODE_CHOICES,
    get_capsule_squash_mode,
    set_capsule_squash_mode,
)
from x2cbct_difnet.metrics import mean_metrics, metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.v2_hash_model import (
    COMBINE_CHOICES,
    HASHGRID_BACKEND_CHOICES,
    HASH_MODEL_VERSIONS,
    MODEL_VERSION_CHOICES,
    build_difnet_model,
    model_config,
    validate_capsule_kwargs_for_combine,
)


DEFAULT_VALIDATION_EVAL_GRID_SIZE = 128
DEFAULT_CONSTANT_LR_STEPS = 8000
DEFAULT_MIN_LR = 1e-5
VALIDATION_TRAIN_DEVICE_POLICY_CHOICES = ("keep-on-device", "offload-to-cpu")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _json_read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


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


def _existing_best_by_psnr(out_dir: Path) -> dict[str, Any] | None:
    metrics_path = out_dir / "metrics_final.json"
    if metrics_path.exists():
        best = _json_read(metrics_path).get("best_by_psnr")
        if isinstance(best, dict):
            return best
    events_path = out_dir / "events.jsonl"
    if not events_path.exists():
        return None
    best = None
    with events_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "best_checkpoint_updated" and row.get("metric") == "psnr":
                best = row
    return best


def _prepare_output_dir(out_dir: Path, *, allow_existing_nonempty: bool = False) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_existing_nonempty:
        raise FileExistsError(f"output dir already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _run_command(cmd: list[str], log_dir: Path, label: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    (log_dir / f"{label}_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / f"{label}_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; logs: {log_dir}")


def _model_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    item = {
        "projs": batch["projs"].float().to(device, non_blocking=non_blocking),
        "points": batch["points"].float().to(device, non_blocking=non_blocking),
        "proj_points": batch["proj_points"].float().to(device, non_blocking=non_blocking),
        "p_gt": batch["p_gt"].float().to(device, non_blocking=non_blocking),
    }
    if "view_angles_deg" in batch:
        item["view_angles_deg"] = batch["view_angles_deg"].float().to(device, non_blocking=non_blocking)
    if "valid_mask" in batch:
        item["valid_mask"] = batch["valid_mask"].bool().to(device, non_blocking=non_blocking)
    for key, value in batch.items():
        if key.startswith("c2rv_grid_"):
            if str(key).endswith("_valid_mask"):
                item[key] = value.bool().to(device, non_blocking=non_blocking)
            else:
                item[key] = value.float().to(device, non_blocking=non_blocking)
    return item


def _loss_mask(valid_mask: torch.Tensor | None, policy: str) -> torch.Tensor | None:
    if policy == "none" or valid_mask is None:
        return None
    if valid_mask.ndim != 3:
        raise ValueError(f"valid_mask must have shape [B, view, N], got {tuple(valid_mask.shape)}")
    if policy == "all":
        point_mask = valid_mask.all(dim=1)
    elif policy == "any":
        point_mask = valid_mask.any(dim=1)
    else:
        raise ValueError(f"unsupported loss valid policy: {policy}")
    return point_mask.unsqueeze(1)


def _masked_mse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None,
    policy: str,
) -> tuple[torch.Tensor, float]:
    mask = _loss_mask(valid_mask, policy)
    if mask is None:
        return torch.nn.functional.mse_loss(pred, gt), float("nan")
    expanded = mask.expand_as(pred)
    valid_ratio = float(expanded.float().mean().item())
    if not bool(expanded.any().item()):
        raise RuntimeError("valid_mask selected zero supervised points")
    return torch.mean((pred[expanded] - gt[expanded]) ** 2), valid_ratio


def _tensor_finite_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    ok = bool(finite.all().item())
    total = int(detached.numel())
    summary: dict[str, Any] = {
        "name": name,
        "finite": ok,
        "total_count": total,
        "non_finite_count": int(total - finite.sum().item()),
    }
    if total and bool(finite.any().item()):
        values = detached[finite].float()
        summary.update(
            {
                "finite_min": float(values.min().item()),
                "finite_max": float(values.max().item()),
                "finite_mean": float(values.mean().item()),
            }
        )
    return summary


def _gradient_finite_summary(model: torch.nn.Module, *, max_bad_names: int = 20) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        checked += 1
        grad_summary = _tensor_finite_summary(name, parameter.grad)
        if not grad_summary["finite"]:
            bad.append(grad_summary)
    return {
        "finite": len(bad) == 0,
        "checked_parameter_count": checked,
        "bad_parameter_count": len(bad),
        "first_bad_parameter": bad[0]["name"] if bad else None,
        "bad_parameters": bad[:max_bad_names],
    }


def _record_non_finite_training_failure(
    *,
    out_dir: Path,
    events_path: Path,
    epoch: int,
    step: int,
    optimizer_step: int,
    reason: str,
    loss_value: float | None,
    valid_ratio: float,
    details: dict[str, Any],
) -> None:
    failure = {
        "event": "non_finite_training_failure",
        "ok": False,
        "epoch": int(epoch),
        "step": int(step),
        "optimizer_step": int(optimizer_step),
        "reason": reason,
        "loss": loss_value,
        "valid_ratio": valid_ratio,
        "details": details,
        "policy": "fail-fast",
    }
    failure_path = out_dir / "failures" / f"non_finite_epoch_{epoch:04d}_step_{step:05d}.json"
    failure["failure_report"] = str(failure_path)
    _json_write(failure, failure_path)
    _append_jsonl(events_path, failure)
    print(json.dumps(failure, ensure_ascii=False), flush=True)


def _raise_non_finite_training_failure(
    *,
    out_dir: Path,
    events_path: Path,
    epoch: int,
    step: int,
    optimizer_step: int,
    reason: str,
    loss_value: float | None,
    valid_ratio: float,
    details: dict[str, Any],
) -> None:
    _record_non_finite_training_failure(
        out_dir=out_dir,
        events_path=events_path,
        epoch=epoch,
        step=step,
        optimizer_step=optimizer_step,
        reason=reason,
        loss_value=loss_value,
        valid_ratio=valid_ratio,
        details=details,
    )
    raise RuntimeError(f"non-finite training failure at epoch={epoch}, step={step}, reason={reason}")


def _assert_forward_finite(
    *,
    pred: torch.Tensor,
    loss: torch.Tensor,
    out_dir: Path,
    events_path: Path,
    epoch: int,
    step: int,
    optimizer_step: int,
    valid_ratio: float,
) -> float:
    loss_value = float(loss.detach().item())
    pred_summary = _tensor_finite_summary("prediction", pred)
    loss_summary = _tensor_finite_summary("loss", loss)
    if not pred_summary["finite"] or not loss_summary["finite"] or not math.isfinite(loss_value):
        _raise_non_finite_training_failure(
            out_dir=out_dir,
            events_path=events_path,
            epoch=epoch,
            step=step,
            optimizer_step=optimizer_step,
            reason="non_finite_forward_or_loss",
            loss_value=loss_value if math.isfinite(loss_value) else None,
            valid_ratio=valid_ratio,
            details={"prediction": pred_summary, "loss": loss_summary},
        )
    return loss_value


def _guarded_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    events_path: Path,
    epoch: int,
    step: int,
    optimizer_step: int,
    loss_value: float,
    valid_ratio: float,
    max_grad_norm: float = 1.0,
) -> float:
    grad_summary = _gradient_finite_summary(model)
    if not grad_summary["finite"]:
        _raise_non_finite_training_failure(
            out_dir=out_dir,
            events_path=events_path,
            epoch=epoch,
            step=step,
            optimizer_step=optimizer_step,
            reason="non_finite_gradient",
            loss_value=loss_value,
            valid_ratio=valid_ratio,
            details={"gradient": grad_summary},
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
    if not math.isfinite(grad_norm_value):
        _raise_non_finite_training_failure(
            out_dir=out_dir,
            events_path=events_path,
            epoch=epoch,
            step=step,
            optimizer_step=optimizer_step,
            reason="non_finite_grad_norm",
            loss_value=loss_value,
            valid_ratio=valid_ratio,
            details={"grad_norm": grad_norm_value, "gradient": grad_summary},
        )
    optimizer.step()
    return grad_norm_value


def _samples_for_split(cache_manifest: dict[str, Any], view_key: str, split: str) -> list[dict[str, Any]]:
    return [
        item
        for item in cache_manifest["samples"]
        if item["split"] == split and item["view_key"] == view_key
    ]


def _validation_split(cache_manifest: dict[str, Any], view_key: str) -> str:
    return "eval" if _samples_for_split(cache_manifest, view_key, "eval") else "test"


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None,
    loss_valid_policy: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    cbct_normalization = getattr(loader.dataset, "cache_manifest", {}).get("cbct_normalization")
    losses = []
    metrics = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            item = _model_batch(batch, device)
            pred, gt = model(item)
            loss, _ = _masked_mse(pred, gt, item.get("valid_mask"), loss_valid_policy)
            losses.append(float(loss.item()))
            metrics.append(metric_dict(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), normalization=cbct_normalization))
            if max_steps is not None and step >= max_steps:
                break
    return float(np.mean(losses)) if losses else float("nan"), mean_metrics(metrics)


def _validation_kind(epoch: int, total_epochs: int, full_test_interval: int, skip_final: bool) -> str | None:
    if epoch == total_epochs and not skip_final:
        return "full_final"
    if full_test_interval > 0 and epoch % full_test_interval == 0:
        return "full"
    return None


def _learning_rate_for_step(
    step_index: int,
    total_steps: int,
    constant_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if constant_steps < 0:
        raise ValueError(f"constant_steps must be >= 0, got {constant_steps}")
    if not 0.0 <= min_lr <= base_lr:
        raise ValueError(f"expected 0 <= min_lr <= base_lr, got min_lr={min_lr}, base_lr={base_lr}")
    if step_index < constant_steps or total_steps <= constant_steps:
        return base_lr
    progress = min(max((step_index - constant_steps + 1) / (total_steps - constant_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resolved: dict[str, Any],
    epoch: int,
    optimizer_step: int,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "optimizer_step": optimizer_step,
        "config": resolved,
        "epoch": epoch,
    }


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resolved: dict[str, Any],
    out_dir: Path,
    epoch: int,
    optimizer_step: int,
) -> Path:
    checkpoint = out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, scheduler, resolved, epoch, optimizer_step), checkpoint)
    return checkpoint


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@contextmanager
def _validation_train_device_context(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    policy: str,
    events_path: Path,
    epoch: int,
):
    if policy not in VALIDATION_TRAIN_DEVICE_POLICY_CHOICES:
        raise ValueError(f"unsupported validation train device policy: {policy}")
    should_offload = policy == "offload-to-cpu" and device.type == "cuda"
    if not should_offload:
        yield
        return

    _append_jsonl(
        events_path,
        {
            "event": "validation_train_device_offload_start",
            "epoch": epoch,
            "policy": policy,
            "from_device": str(device),
            "to_device": "cpu",
        },
    )
    model.to(torch.device("cpu"))
    _optimizer_to_device(optimizer, torch.device("cpu"))
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        model.to(device)
        _optimizer_to_device(optimizer, device)
        torch.cuda.empty_cache()
        _append_jsonl(
            events_path,
            {
                "event": "validation_train_device_restore_end",
                "epoch": epoch,
                "policy": policy,
                "device": str(device),
            },
        )


def _assert_required_hashgrid_backend(
    model: torch.nn.Module,
    *,
    model_version: str,
    requested_backend: str,
    device: torch.device,
) -> None:
    if model_version not in HASH_MODEL_VERSIONS or device.type != "cuda":
        return
    config = model_config(model, {})
    hashgrid = config.get("hashgrid", {}) if isinstance(config, dict) else {}
    actual_backend = hashgrid.get("backend_actual")
    if actual_backend == "cuda":
        return
    fallback_reason = hashgrid.get("backend_fallback_reason")
    raise RuntimeError(
        "V2 hash CUDA training requires NAF CUDA hashgrid, "
        f"but actual backend is {actual_backend!r}. "
        f"requested backend={requested_backend!r}. "
        "Run through tools\\run_difnet_v2_hash_cuda_train.ps1 or load Visual Studio x64 C++ environment before training. "
        f"fallback reason: {fallback_reason}"
    )


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is None:
        raise ValueError(f"resume checkpoint has no optimizer_state: {checkpoint_path}")
    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler_state is None:
        raise ValueError(f"resume checkpoint has no scheduler_state: {checkpoint_path}")
    if checkpoint.get("optimizer_step") is None:
        raise ValueError(f"resume checkpoint has no optimizer_step: {checkpoint_path}")
    optimizer.load_state_dict(optimizer_state)
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(scheduler_state)
    return checkpoint if isinstance(checkpoint, dict) else {"epoch": None}


def _metric_summary_is_finite(summary: dict[str, Any]) -> None:
    for key in ("mae_norm", "mae_hu", "psnr", "ssim"):
        value = float(summary[key])
        if not math.isfinite(value):
            raise RuntimeError(f"validation metric {key} is not finite: {value}")


def _validation_plan(cache_manifest: dict[str, Any], view_key: str, args: argparse.Namespace) -> dict[str, Any]:
    split = _validation_split(cache_manifest, view_key)
    full_validation = {
        "interval_epochs": args.full_test_interval,
        "final_epoch": not args.skip_final_validation,
        "max_test_cases": args.full_test_max_cases,
    }
    return {
        "schema": "x2cbct_difnet_v0b_training_validation_plan.v1",
        "blocking": True,
        "execution_failure_policy": args.validation_failure_policy,
        "metric_quality_gate": "none; metric values are recorded and printed, not used to stop training",
        "non_finite_metrics_are_execution_failures": True,
        "validation_split": split,
        "train_device_policy": args.validation_train_device_policy,
        "full_validation": full_validation,
        "full_test": full_validation,
        "artifacts": [
            "checkpoint",
            "checkpoints/best_by_psnr.pt",
            "test/prediction_manifest.json",
            "test/metrics_summary.json",
            "test/generated_cbct",
            "test/target_cbct",
            "comparisons/visualization_summary.json",
            "viewer/index.html",
            "viewer/viewer_data.json",
            "viewer/viewer_summary.json",
        ],
    }


def _run_validation(
    args: argparse.Namespace,
    epoch: int,
    kind: str,
    checkpoint: Path,
    out_dir: Path,
    events_path: Path,
) -> dict[str, Any]:
    validation_dir = out_dir / "validation" / f"epoch_{epoch:04d}_{kind}"
    test_dir = validation_dir / "test"
    comparison_dir = validation_dir / "comparisons"
    viewer_dir = validation_dir / "viewer"
    log_dir = validation_dir / "command_logs"
    max_cases = args.full_test_max_cases
    started = time.perf_counter()
    _append_jsonl(events_path, {"event": "validation_start", "epoch": epoch, "kind": kind, "checkpoint": str(checkpoint), "output_dir": str(validation_dir)})
    print(f"[validation] epoch={epoch:04d} kind={kind} checkpoint={checkpoint.name} started", flush=True)

    test_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "test_difnet_v0b.py"),
        "--cache-root",
        str(args.validation_cache_root),
        "--split",
        args.validation_split,
        "--view-count",
        str(args.view_count),
        "--checkpoint",
        str(checkpoint),
        "--eval-grid-size",
        str(args.validation_eval_grid_size),
        "--chunk-size",
        str(args.validation_chunk_size),
        "--combine",
        args.combine,
        "--model-version",
        args.model_version,
        "--mid-channels",
        str(args.mid_channels),
        "--routed-capsule-squash",
        args.routed_capsule_squash,
        "--device",
        args.device,
        "--output-dir",
        str(test_dir),
        "--skip-visualization",
    ]
    if max_cases is not None:
        test_cmd.extend(["--max-test-cases", str(max_cases)])
    if args.capsule_hidden_dim is not None:
        test_cmd.extend(["--capsule-hidden-dim", str(args.capsule_hidden_dim)])
    if args.capsule_layers is not None:
        test_cmd.extend(["--capsule-layers", str(args.capsule_layers)])
    if args.capsule_heads is not None:
        test_cmd.extend(["--capsule-heads", str(args.capsule_heads)])
    if args.capsule_mlp_ratio is not None:
        test_cmd.extend(["--capsule-mlp-ratio", str(args.capsule_mlp_ratio)])
    if args.capsule_dropout is not None:
        test_cmd.extend(["--capsule-dropout", str(args.capsule_dropout)])
    if args.capsule_valid_mask:
        test_cmd.append("--capsule-valid-mask")
    _run_command(test_cmd, log_dir, "test")

    viz_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "visualize_difnet_v0b_predictions.py"),
        "--prediction-manifest",
        str(test_dir / "prediction_manifest.json"),
        "--output-dir",
        str(comparison_dir),
        "--slice-mode",
        "all",
        "--num-slices",
        "1",
    ]
    if args.visualization_max_cases is not None:
        viz_cmd.extend(["--max-cases", str(args.visualization_max_cases)])
    _run_command(viz_cmd, log_dir, "visualize")

    viewer_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "view_difnet_v0b_prediction.py"),
        "--prediction-manifest",
        str(test_dir / "prediction_manifest.json"),
        "--output-dir",
        str(viewer_dir),
        "--slice-stride",
        str(args.viewer_slice_stride),
    ]
    if args.viewer_max_cases is not None:
        viewer_cmd.extend(["--max-cases", str(args.viewer_max_cases)])
    _run_command(viewer_cmd, log_dir, "viewer")

    required = [
        test_dir / "prediction_manifest.json",
        test_dir / "metrics_summary.json",
        comparison_dir / "visualization_summary.json",
        viewer_dir / "index.html",
        viewer_dir / "viewer_data.json",
        viewer_dir / "viewer_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"validation artifact missing: {missing}")
    metrics_summary = _json_read(test_dir / "metrics_summary.json")
    _metric_summary_is_finite(metrics_summary)
    elapsed = time.perf_counter() - started
    result = {
        "event": "validation_end",
        "ok": True,
        "epoch": epoch,
        "kind": kind,
        "elapsed_sec": elapsed,
        "elapsed": _format_duration(elapsed),
        "checkpoint": str(checkpoint),
        "output_dir": str(validation_dir),
        "metrics_summary": metrics_summary,
        "prediction_manifest": str(test_dir / "prediction_manifest.json"),
        "comparison_summary": str(comparison_dir / "visualization_summary.json"),
        "viewer_html": str(viewer_dir / "index.html"),
    }
    _append_jsonl(events_path, result)
    print(
        "[validation] "
        f"epoch={epoch:04d} kind={kind} cases={metrics_summary['case_count']} "
        f"mae_norm={metrics_summary['mae_norm']:.6f} mae_hu={metrics_summary['mae_hu']:.3f} "
        f"psnr={metrics_summary['psnr']:.3f} ssim={metrics_summary['ssim']:.6f} "
        f"elapsed={_format_duration(elapsed)} output={validation_dir}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DIF-Net V0b from preprocessing cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--validation-cache-root", type=Path, help="Independent test cache root for periodic full test runs.")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--constant-lr-steps", type=int, default=DEFAULT_CONSTANT_LR_STEPS)
    parser.add_argument("--combine", choices=COMBINE_CHOICES, default="mlp")
    parser.add_argument("--mid-channels", type=int, default=128)
    parser.add_argument("--routed-capsule-squash", choices=CAPSULE_SQUASH_MODE_CHOICES, default="stable")
    parser.add_argument("--model-version", choices=MODEL_VERSION_CHOICES, default="v0b")
    parser.add_argument("--hashgrid-backend", choices=HASHGRID_BACKEND_CHOICES, default="auto")
    parser.add_argument("--capsule-hidden-dim", type=int)
    parser.add_argument("--capsule-layers", type=int)
    parser.add_argument("--capsule-heads", type=int)
    parser.add_argument("--capsule-mlp-ratio", type=float)
    parser.add_argument("--capsule-dropout", type=float)
    capsule_mask_group = parser.add_mutually_exclusive_group()
    capsule_mask_group.add_argument("--capsule-valid-mask", dest="capsule_valid_mask", action="store_true")
    capsule_mask_group.add_argument("--no-capsule-valid-mask", dest="capsule_valid_mask", action="store_false")
    parser.set_defaults(capsule_valid_mask=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument("--full-test-interval", type=int, default=50)
    parser.add_argument("--full-test-max-cases", type=int)
    parser.add_argument("--skip-final-validation", action="store_true")
    parser.add_argument("--validation-eval-grid-size", type=int, default=DEFAULT_VALIDATION_EVAL_GRID_SIZE)
    parser.add_argument("--validation-chunk-size", type=int, default=32768)
    parser.add_argument(
        "--validation-train-device-policy",
        choices=VALIDATION_TRAIN_DEVICE_POLICY_CHOICES,
        default="keep-on-device",
        help=(
            "keep-on-device preserves the original validation behavior; offload-to-cpu moves "
            "the training model and optimizer state to CPU before full test and restores them afterward."
        ),
    )
    parser.add_argument("--validation-failure-policy", choices=("fail-fast", "continue"), default="fail-fast")
    parser.add_argument("--visualization-max-cases", type=int)
    parser.add_argument("--viewer-max-cases", type=int)
    parser.add_argument("--viewer-slice-stride", type=int, default=4)
    parser.add_argument("--loss-valid-policy", choices=("any", "all", "none"), default="any")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-start-epoch", type=int, help="First epoch to run after loading --resume-checkpoint. Defaults to checkpoint epoch + 1.")
    args = parser.parse_args()
    set_capsule_squash_mode(args.routed_capsule_squash)

    if args.lr <= 0.0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if not 0.0 <= args.min_lr <= args.lr:
        raise ValueError(f"expected 0 <= --min-lr <= --lr, got min_lr={args.min_lr}, lr={args.lr}")
    if args.constant_lr_steps < 0:
        raise ValueError(f"--constant-lr-steps must be >= 0, got {args.constant_lr_steps}")
    if args.full_test_interval < 0:
        raise ValueError(f"--full-test-interval must be >= 0, got {args.full_test_interval}")
    validate_capsule_kwargs_for_combine(
        args.combine,
        capsule_hidden_dim=args.capsule_hidden_dim,
        capsule_layers=args.capsule_layers,
        capsule_heads=args.capsule_heads,
        capsule_mlp_ratio=args.capsule_mlp_ratio,
        capsule_dropout=args.capsule_dropout,
        capsule_valid_mask=args.capsule_valid_mask,
    )

    cache_manifest = load_cache_manifest(args.cache_root)
    view_key = view_key_from_count(args.view_count)
    if view_key not in cache_manifest["views"]:
        raise ValueError(f"{view_key} is absent from cache")
    args.validation_cache_root = (args.validation_cache_root or args.cache_root).resolve()
    validation_cache_manifest = load_cache_manifest(args.validation_cache_root)
    if view_key not in validation_cache_manifest["views"]:
        raise ValueError(f"{view_key} is absent from validation cache")
    validation_split = _validation_split(validation_cache_manifest, view_key)
    args.validation_split = validation_split
    out_dir = args.output_dir or PROJECT_ROOT / "runs" / "difnet_v0b" / f"train-{view_key}-{datetime.now():%Y%m%d-%H%M%S}"
    _prepare_output_dir(out_dir, allow_existing_nonempty=args.resume_checkpoint is not None)
    device = _resolve_device(args.device)

    train_dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split="train",
        view_key=view_key,
        num_points=args.num_points,
        max_cases=args.max_train_cases,
        c2rv_grids=args.combine == "c2rv" or args.model_version == "c2rv-style",
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = None
    if args.validation_steps is not None:
        test_dataset = CachedDifNetV0bDataset(
            args.validation_cache_root,
            split=validation_split,
            view_key=view_key,
            num_points=args.num_points,
            max_cases=args.max_test_cases,
            deterministic=True,
            c2rv_grids=args.combine == "c2rv" or args.model_version == "c2rv-style",
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=args.model_version,
        num_views=args.view_count,
        combine=args.combine,
        mid_channels=args.mid_channels,
        hashgrid_backend=args.hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
        capsule_hidden_dim=args.capsule_hidden_dim,
        capsule_layers=args.capsule_layers,
        capsule_heads=args.capsule_heads,
        capsule_mlp_ratio=args.capsule_mlp_ratio,
        capsule_dropout=args.capsule_dropout,
        capsule_valid_mask=args.capsule_valid_mask,
    ).to(device)
    _assert_required_hashgrid_backend(
        model,
        model_version=args.model_version,
        requested_backend=args.hashgrid_backend,
        device=device,
    )
    full_train_steps = len(train_loader)
    if args.steps_per_epoch is not None:
        full_train_steps = min(full_train_steps, args.steps_per_epoch)
    total_optimizer_steps = args.epochs * full_train_steps
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_index: _learning_rate_for_step(
            step_index,
            total_optimizer_steps,
            args.constant_lr_steps,
            args.lr,
            args.min_lr,
        )
        / args.lr,
    )
    resume_info: dict[str, Any] | None = None
    start_epoch = 1
    optimizer_step = 0
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.resolve()
        resume_info = _load_resume_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
        )
        checkpoint_epoch = resume_info.get("epoch")
        if args.resume_start_epoch is not None:
            start_epoch = args.resume_start_epoch
        elif checkpoint_epoch is not None:
            start_epoch = int(checkpoint_epoch) + 1
        else:
            raise ValueError("--resume-start-epoch is required when checkpoint epoch is absent")
        optimizer_step = int(resume_info.get("optimizer_step", 0))
        if start_epoch < 1:
            raise ValueError(f"resume start epoch must be >= 1, got {start_epoch}")
        if start_epoch > args.epochs:
            raise ValueError(f"resume start epoch {start_epoch} is greater than target --epochs {args.epochs}")
    resolved = {
        "cache_root": str(args.cache_root.resolve()),
        "cache_version": cache_manifest["cache_version"],
        "cbct_normalization": cache_manifest["cbct_normalization"],
        "view_count": args.view_count,
        "view_key": view_key,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_points": args.num_points,
        "mid_channels": args.mid_channels,
        "combine": args.combine,
        "routed_capsule_squash": get_capsule_squash_mode(),
        "model_version": args.model_version,
        "hashgrid_backend": args.hashgrid_backend,
        "model_config": model_config(
            model,
            {
                "model_version": "v0b",
                "num_views": args.view_count,
                "combine": args.combine,
                "mid_channels": args.mid_channels,
            },
        ),
        "lr": args.lr,
        "min_lr": args.min_lr,
        "lr_schedule": "constant_then_cosine_decay",
        "constant_lr_steps": args.constant_lr_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "scheduler_step_unit": "optimizer_step",
        "requested_device": args.device,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "validation_cache_root": str(args.validation_cache_root),
        "validation_split": validation_split,
        "test_split_role": "eval-as-validation" if validation_split == "eval" else "test-as-validation; no separate validation split in V0b",
        "has_validation_split": validation_split == "eval",
        "quick_point_validation_steps": args.validation_steps,
        "formal_full_test_interval": args.full_test_interval,
        "formal_validation_artifact_policy": "retain full test outputs and checkpoints every configured interval and at final epoch",
        "checkpoint_policy": "retain full-test epoch checkpoints and copy the highest-PSNR checkpoint to best_by_psnr.pt",
        "formal_validation_eval_grid_size": args.validation_eval_grid_size,
        "formal_validation_chunk_size": args.validation_chunk_size,
        "validation_train_device_policy": args.validation_train_device_policy,
        "viewer_slice_stride": args.viewer_slice_stride,
        "loss_valid_policy": args.loss_valid_policy,
        "validation_failure_policy": args.validation_failure_policy,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "resume_checkpoint_epoch": resume_info.get("epoch") if resume_info else None,
        "resume_start_epoch": start_epoch,
        "resume_optimizer_state": bool(args.resume_checkpoint),
        "command": " ".join(sys.argv),
    }
    _json_write(resolved, out_dir / "config_resolved.json")
    _json_write(_validation_plan(validation_cache_manifest, view_key, args), out_dir / "validation_plan.json")
    events_path = out_dir / "events.jsonl"
    device_event = {"event": "device_resolved", **{k: resolved[k] for k in ("requested_device", "device", "cuda_available", "cuda_device_name", "pin_memory", "num_workers")}}
    _append_jsonl(events_path, device_event)
    print(json.dumps(device_event, ensure_ascii=False), flush=True)
    if resume_info is not None:
        resume_event = {
            "event": "resume_loaded",
            "checkpoint": resolved["resume_checkpoint"],
            "checkpoint_epoch": resolved["resume_checkpoint_epoch"],
            "start_epoch": start_epoch,
            "optimizer_state": resolved["resume_optimizer_state"],
            "optimizer_step": optimizer_step,
            "scheduler_state": bool(resume_info.get("scheduler_state")),
        }
        _append_jsonl(events_path, resume_event)
        print(json.dumps(resume_event, ensure_ascii=False), flush=True)
    jsonl_path = out_dir / "train_log.jsonl"
    csv_path = out_dir / "train_log.csv"
    logs = [row for row in _read_jsonl(jsonl_path) if row.get("event") == "epoch_end"]
    validation_records = [
        row
        for row in _read_jsonl(events_path)
        if row.get("event") in {"validation_end", "validation_failed"}
    ]
    best_by_psnr = _existing_best_by_psnr(out_dir) if resume_info is not None else None
    best_checkpoint = out_dir / "checkpoints" / "best_by_psnr.pt"
    last_checkpoint: Path | None = None
    started_total = time.perf_counter()

    run_total_epochs = args.epochs - start_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        train_valid_ratios = []
        for step, batch in enumerate(train_loader, start=1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            item = _model_batch(batch, device)
            pred, gt = model(item)
            loss, valid_ratio = _masked_mse(pred, gt, item.get("valid_mask"), args.loss_valid_policy)
            loss_value = _assert_forward_finite(
                pred=pred,
                loss=loss,
                out_dir=out_dir,
                events_path=events_path,
                epoch=epoch,
                step=step,
                optimizer_step=optimizer_step,
                valid_ratio=valid_ratio,
            )
            loss.backward()
            grad_norm = _guarded_optimizer_step(
                model=model,
                optimizer=optimizer,
                out_dir=out_dir,
                events_path=events_path,
                epoch=epoch,
                step=step,
                optimizer_step=optimizer_step,
                loss_value=loss_value,
                valid_ratio=valid_ratio,
            )
            optimizer_step += 1
            scheduler.step()
            train_losses.append(loss_value)
            if math.isfinite(valid_ratio):
                train_valid_ratios.append(valid_ratio)
            completed_units = (epoch - start_epoch) + min(step, full_train_steps) / max(full_train_steps, 1)
            total_elapsed = time.perf_counter() - started_total
            eta = (total_elapsed / completed_units) * max(run_total_epochs - completed_units, 0.0) if completed_units > 0 else 0.0
            _append_jsonl(
                events_path,
                {
                    "event": "train_step",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "step": step,
                    "steps_per_epoch": full_train_steps,
                    "loss": loss_value,
                    "grad_norm": grad_norm,
                    "loss_valid_policy": args.loss_valid_policy,
                    "valid_ratio": valid_ratio,
                    "step_elapsed_sec": time.perf_counter() - step_started,
                    "epoch_elapsed_sec": time.perf_counter() - epoch_started,
                    "total_elapsed_sec": total_elapsed,
                    "eta_sec": eta,
                },
            )
            if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                break
        if test_loader is not None:
            val_loss, val_metrics = _evaluate(model, test_loader, device, args.validation_steps, args.loss_valid_policy)
        else:
            val_loss = float("nan")
            val_metrics = {"mae_norm": float("nan"), "mae_hu": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
        epoch_elapsed = time.perf_counter() - epoch_started
        row = {
            "event": "epoch_end",
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_valid_ratio": float(np.mean(train_valid_ratios)) if train_valid_ratios else float("nan"),
            "quick_point_validation_loss": val_loss,
            "quick_point_mae_norm": val_metrics["mae_norm"],
            "quick_point_mae_hu": val_metrics["mae_hu"],
            "quick_point_psnr": val_metrics["psnr"],
            "quick_point_ssim": val_metrics["ssim"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "optimizer_step": optimizer_step,
            "epoch_elapsed_sec": epoch_elapsed,
            "total_elapsed_sec": time.perf_counter() - started_total,
        }
        logs.append(row)
        _append_jsonl(jsonl_path, row)
        _append_jsonl(events_path, row)
        completed_epochs = max(epoch - start_epoch + 1, 1)
        total_elapsed = row["total_elapsed_sec"]
        epoch_eta = (total_elapsed / completed_epochs) * max(run_total_epochs - completed_epochs, 0) if completed_epochs else 0.0
        print(
            "[train] "
            f"epoch={epoch:04d}/{args.epochs:04d} "
            f"train_loss={row['train_loss']:.6f} "
            f"epoch_elapsed={_format_duration(epoch_elapsed)} "
            f"total_elapsed={_format_duration(total_elapsed)} "
            f"eta={_format_duration(epoch_eta)}",
            flush=True,
        )
        if train_losses:
            batch = item = pred = gt = loss = None

        kind = _validation_kind(epoch, args.epochs, args.full_test_interval, args.skip_final_validation)
        if kind is not None:
            last_checkpoint = _save_checkpoint(model, optimizer, scheduler, resolved, out_dir, epoch, optimizer_step)
            try:
                with _validation_train_device_context(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    policy=args.validation_train_device_policy,
                    events_path=events_path,
                    epoch=epoch,
                ):
                    validation_result = _run_validation(args, epoch, kind, last_checkpoint, out_dir, events_path)
                validation_records.append(validation_result)
                psnr = float(validation_result["metrics_summary"]["psnr"])
                if best_by_psnr is None or psnr > float(best_by_psnr["psnr"]):
                    shutil.copy2(last_checkpoint, best_checkpoint)
                    best_by_psnr = {
                        "psnr": psnr,
                        "checkpoint": str(last_checkpoint),
                        "best_checkpoint": str(best_checkpoint),
                        "epoch": epoch,
                        "kind": kind,
                        "metrics_summary": validation_result["metrics_summary"],
                        "output_dir": validation_result["output_dir"],
                    }
                    _append_jsonl(events_path, {"event": "best_checkpoint_updated", "metric": "psnr", **best_by_psnr})
            except Exception as exc:
                failed = {
                    "event": "validation_failed",
                    "ok": False,
                    "epoch": epoch,
                    "kind": kind,
                    "checkpoint": str(last_checkpoint),
                    "error": str(exc),
                    "policy": args.validation_failure_policy,
                }
                validation_records.append(failed)
                _append_jsonl(events_path, failed)
                print(json.dumps(failed, ensure_ascii=False), flush=True)
                if args.validation_failure_policy == "fail-fast":
                    raise

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[-1].keys()))
        writer.writeheader()
        writer.writerows(logs)
    final = {
        "ok": True,
        "checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
        "best_by_psnr": best_by_psnr,
        "validation_plan": str(out_dir / "validation_plan.json"),
        "validation_records": validation_records,
        "events_jsonl": str(events_path),
        "train_log_jsonl": str(jsonl_path),
        "train_log_csv": str(csv_path),
        "final_epoch": logs[-1],
        "validation_split": validation_split,
        "test_split_role": "eval-as-validation" if validation_split == "eval" else "test-as-validation; no separate validation split in V0b",
        "has_validation_split": validation_split == "eval",
    }
    _json_write(final, out_dir / "metrics_final.json")
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(out_dir),
                "checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
                "best_by_psnr": best_by_psnr,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
