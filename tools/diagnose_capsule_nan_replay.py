from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_difnet_v0b import _gradient_finite_summary, _masked_mse
from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset
from x2cbct_difnet.capsule_routing import (
    CAPSULE_SQUASH_MODE_CHOICES,
    capture_capsule_diagnostics,
    get_capsule_squash_mode,
    set_capsule_squash_mode,
)
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.v2_hash_model import (
    COMBINE_CHOICES,
    HASHGRID_BACKEND_CHOICES,
    MODEL_VERSION_CHOICES,
    build_difnet_model,
    resolve_checkpoint_combine,
    resolve_checkpoint_hash_backend,
    resolve_checkpoint_model_version,
    resolve_checkpoint_routed_capsule_squash,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def _json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _model_item(item: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    out = {
        "projs": torch.from_numpy(item["projs"]).unsqueeze(0).float().to(device, non_blocking=non_blocking),
        "points": torch.from_numpy(item["points"]).unsqueeze(0).float().to(device, non_blocking=non_blocking),
        "proj_points": torch.from_numpy(item["proj_points"]).unsqueeze(0).float().to(device, non_blocking=non_blocking),
        "p_gt": torch.from_numpy(item["p_gt"]).unsqueeze(0).float().to(device, non_blocking=non_blocking),
    }
    if "valid_mask" in item:
        out["valid_mask"] = torch.from_numpy(item["valid_mask"]).unsqueeze(0).bool().to(device, non_blocking=non_blocking)
    if "view_angles_deg" in item:
        out["view_angles_deg"] = torch.from_numpy(item["view_angles_deg"]).unsqueeze(0).float().to(device, non_blocking=non_blocking)
    return out


def _infer_mid_channels(checkpoint: Any, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict) and model_cfg.get("mid_channels") is not None:
                return int(model_cfg["mid_channels"])
            if config.get("mid_channels") is not None:
                return int(config["mid_channels"])
    return 128


def _sample_identity(item: dict[str, Any], index: int) -> dict[str, Any]:
    trace = dict(item.get("trace", {}))
    return {
        "index": index,
        "name": item.get("name"),
        "sample_path": trace.get("sample_path"),
        "patient_id": trace.get("patient_id"),
        "case_id": trace.get("case_id"),
        "case_key": trace.get("case_key"),
        "source_case_id": trace.get("source_case_id"),
        "augmentation_id": trace.get("augmentation_id"),
    }


def _pick_primary_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        if record.get("event") == "squash_input" and record.get("source") == "primary_capsules":
            return record.get("norm")
    return None


def _pick_parent_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record.get("norm", {})
        for record in records
        if record.get("event") == "squash_input" and record.get("source") == "efficient_parent"
    ]


def _finite_flag(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def _run_one(
    *,
    model: torch.nn.Module,
    dataset: CachedDifNetV0bDataset,
    index: int,
    device: torch.device,
    loss_valid_policy: str,
    detect_anomaly: bool,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    item = dataset[index]
    batch = _model_item(item, device)
    context = torch.autograd.detect_anomaly(check_nan=True) if detect_anomaly else torch.enable_grad()
    anomaly_error: str | None = None
    with capture_capsule_diagnostics() as capsule_records:
        try:
            with context:
                pred, gt = model(batch)
                loss, valid_ratio = _masked_mse(pred, gt, batch.get("valid_mask"), loss_valid_policy)
                loss.backward()
        except RuntimeError as exc:
            anomaly_error = str(exc)
            pred = locals().get("pred")
            gt = locals().get("gt")
            loss = locals().get("loss")
            valid_ratio = locals().get("valid_ratio", float("nan"))
    grad_summary = _gradient_finite_summary(model)
    loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) and loss.numel() == 1 else None
    return {
        "sample": _sample_identity(item, index),
        "loss": loss_value,
        "loss_finite": loss_value is not None and math.isfinite(loss_value),
        "valid_ratio": float(valid_ratio) if isinstance(valid_ratio, float) else float("nan"),
        "prediction_finite": _finite_flag(pred) if isinstance(pred, torch.Tensor) else False,
        "target_finite": _finite_flag(gt) if isinstance(gt, torch.Tensor) else False,
        "gradient": grad_summary,
        "capsule_records": capsule_records,
        "primary_norm": _pick_primary_stats(capsule_records),
        "parent_norms": _pick_parent_stats(capsule_records),
        "anomaly_error": anomaly_error,
    }


def _write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    rows = report["samples"]
    bad = [row for row in rows if not row["gradient"]["finite"] or not row["loss_finite"]]
    top = sorted(
        rows,
        key=lambda row: float((row.get("primary_norm") or {}).get("max", float("-inf"))),
        reverse=True,
    )[:10]
    lines = [
        "# Capsule NaN Replay Summary",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- cache_root: `{report['cache_root']}`",
        f"- squash: `{report['routed_capsule_squash']}`",
        f"- device: `{report['device']}`",
        f"- sample_count: `{len(rows)}`",
        f"- bad_sample_count: `{len(bad)}`",
        "",
        "## Top Primary Norm Samples",
        "",
        "| index | loss | primary_norm_max | count_gt_88 | gradient_finite | first_bad_parameter |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in top:
        primary = row.get("primary_norm") or {}
        gradient = row["gradient"]
        lines.append(
            "| {index} | {loss} | {max_norm} | {gt88} | {grad_ok} | {first_bad} |".format(
                index=row["sample"]["index"],
                loss=row["loss"],
                max_norm=primary.get("max"),
                gt88=primary.get("count_gt_88_0"),
                grad_ok=gradient["finite"],
                first_bad=gradient.get("first_bad_parameter"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay routed-capsule train samples and inspect finite gradients.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--combine", choices=("auto", *COMBINE_CHOICES), default="auto")
    parser.add_argument("--mid-channels", type=int)
    parser.add_argument("--model-version", choices=("auto", *MODEL_VERSION_CHOICES), default="auto")
    parser.add_argument("--hashgrid-backend", choices=HASHGRID_BACKEND_CHOICES, default="auto")
    parser.add_argument("--routed-capsule-squash", choices=("auto", *CAPSULE_SQUASH_MODE_CHOICES), default="auto")
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--loss-valid-policy", choices=("any", "all", "none"), default="any")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_version = resolve_checkpoint_model_version(checkpoint, requested=args.model_version)
    combine = resolve_checkpoint_combine(checkpoint, requested=args.combine)
    squash = resolve_checkpoint_routed_capsule_squash(checkpoint, requested=args.routed_capsule_squash)
    set_capsule_squash_mode(squash)
    mid_channels = _infer_mid_channels(checkpoint, args.mid_channels)
    hashgrid_backend = resolve_checkpoint_hash_backend(checkpoint, requested=args.hashgrid_backend)
    view_key = view_key_from_count(args.view_count)
    dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split=args.split,
        view_key=view_key,
        num_points=args.num_points,
        deterministic=True,
    )
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=model_version,
        num_views=args.view_count,
        combine=combine,
        mid_channels=mid_channels,
        hashgrid_backend=hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.train()
    if args.sample_index is not None:
        indices = [int(args.sample_index)]
    else:
        count = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
        indices = list(range(count))
    samples = [
        _run_one(
            model=model,
            dataset=dataset,
            index=index,
            device=device,
            loss_valid_policy=args.loss_valid_policy,
            detect_anomaly=args.detect_anomaly,
        )
        for index in indices
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bad_count = sum(1 for row in samples if not row["gradient"]["finite"] or not row["loss_finite"])
    report = {
        "schema": "x2cbct_capsule_nan_replay.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok": bad_count == 0,
        "bad_sample_count": bad_count,
        "checkpoint": str(args.checkpoint.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "view_count": args.view_count,
        "view_key": view_key,
        "num_points": args.num_points,
        "combine": combine,
        "mid_channels": mid_channels,
        "model_version": model_version,
        "routed_capsule_squash": get_capsule_squash_mode(),
        "device": str(device),
        "detect_anomaly": bool(args.detect_anomaly),
        "samples": samples,
    }
    _json_write(report, args.output)
    summary_path = args.output.with_suffix(".md")
    _write_markdown_summary(report, summary_path)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "bad_sample_count": bad_count,
                "report": str(args.output),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
