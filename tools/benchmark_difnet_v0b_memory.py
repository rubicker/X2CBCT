from __future__ import annotations

import argparse
import json
import sys
import time
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
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.v2_hash_model import HASHGRID_BACKEND_CHOICES, MODEL_VERSION_CHOICES, build_difnet_model, model_config


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {
        "projs": batch["projs"].float().to(device, non_blocking=non_blocking),
        "points": batch["points"].float().to(device, non_blocking=non_blocking),
        "proj_points": batch["proj_points"].float().to(device, non_blocking=non_blocking),
        "p_gt": batch["p_gt"].float().to(device, non_blocking=non_blocking),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=non_blocking),
    }


def _masked_mse(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, policy: str) -> tuple[torch.Tensor, float]:
    if policy == "none":
        return torch.nn.functional.mse_loss(pred, gt), float("nan")
    point_mask = valid_mask.any(dim=1) if policy == "any" else valid_mask.all(dim=1)
    point_mask = point_mask.unsqueeze(1).to(dtype=pred.dtype)
    valid_count = point_mask.sum().clamp_min(1.0)
    loss = ((pred - gt) ** 2 * point_mask).sum() / valid_count
    return loss, float(point_mask.mean().item())


def _cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "cuda_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "cuda_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "cuda_max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one-batch DIF-Net V0b training memory and throughput.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mid-channels", type=int, default=128)
    parser.add_argument("--model-version", choices=MODEL_VERSION_CHOICES, default="v0b")
    parser.add_argument("--hashgrid-backend", choices=HASHGRID_BACKEND_CHOICES, default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--loss-valid-policy", choices=("any", "all", "none"), default="any")
    parser.add_argument("--max-cases", type=int, default=64)
    args = parser.parse_args()

    cache_manifest = load_cache_manifest(args.cache_root)
    view_key = view_key_from_count(args.view_count)
    if view_key not in cache_manifest["views"]:
        raise ValueError(f"{view_key} is absent from cache")

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split="train",
        view_key=view_key,
        num_points=args.num_points,
        max_cases=args.max_cases,
        deterministic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=args.model_version,
        num_views=args.view_count,
        combine="mlp",
        mid_channels=args.mid_channels,
        hashgrid_backend=args.hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    step_rows: list[dict[str, float]] = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_time = time.perf_counter() - started
        optimizer.zero_grad(set_to_none=True)
        item = _batch_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_started = time.perf_counter()
        pred, gt = model(item)
        loss, valid_ratio = _masked_mse(pred, gt, item["valid_mask"], args.loss_valid_policy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - compute_started
        step_rows.append(
            {
                "step": float(step),
                "data_sec": data_time,
                "compute_sec": compute_time,
                "total_sec": time.perf_counter() - started,
                "loss": float(loss.item()),
                "valid_ratio": valid_ratio,
            }
        )

    output = {
        "ok": True,
        "cache_root": str(args.cache_root.resolve()),
        "view_count": args.view_count,
        "view_key": view_key,
        "batch_size": args.batch_size,
        "num_points": args.num_points,
        "effective_points": args.batch_size * args.num_points,
        "num_workers": args.num_workers,
        "mid_channels": args.mid_channels,
        "model_version": args.model_version,
        "model_config": model_config(
            model,
            {
                "model_version": "v0b",
                "num_views": args.view_count,
                "combine": "mlp",
                "mid_channels": args.mid_channels,
            },
        ),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "steps": step_rows,
        "avg_total_sec": float(np.mean([row["total_sec"] for row in step_rows])),
        "avg_compute_sec": float(np.mean([row["compute_sec"] for row in step_rows])),
        **_cuda_memory(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
