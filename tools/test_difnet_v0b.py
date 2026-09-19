from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset, load_cache_manifest
from x2cbct_difnet.capsule_routing import CAPSULE_SQUASH_MODE_CHOICES, get_capsule_squash_mode, set_capsule_squash_mode
from x2cbct_difnet.metrics import mean_metrics, metric_dict
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.v2_hash_model import (
    COMBINE_CHOICES,
    HASHGRID_BACKEND_CHOICES,
    MODEL_VERSION_CHOICES,
    build_difnet_model,
    model_config,
    resolve_checkpoint_capsule_kwargs,
    resolve_checkpoint_combine,
    resolve_checkpoint_hash_backend,
    resolve_checkpoint_mid_channels,
    resolve_checkpoint_model_version,
    resolve_checkpoint_routed_capsule_squash,
    resolve_checkpoint_view_count,
    validate_capsule_kwargs_for_combine,
)


DEFAULT_TEST_EVAL_GRID_SIZE = 128
DEFAULT_TEST_VISUALIZATION_MAX_CASES = 50
DEFAULT_TEST_VIEWER_MAX_CASES = 50
DEFAULT_TEST_VIEWER_SLICE_STRIDE = 4


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _write_json(data: dict[str, Any], path: Path) -> None:
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
    if "view_angles_deg" in item:
        out["view_angles_deg"] = torch.from_numpy(item["view_angles_deg"]).unsqueeze(0).float().to(device, non_blocking=non_blocking)
    if "valid_mask" in item:
        out["valid_mask"] = torch.from_numpy(item["valid_mask"]).unsqueeze(0).bool().to(device, non_blocking=non_blocking)
    for key, value in item.items():
        if key.startswith("c2rv_grid_"):
            tensor = torch.from_numpy(value).unsqueeze(0)
            if key.endswith("_valid_mask"):
                out[key] = tensor.bool().to(device, non_blocking=non_blocking)
            else:
                out[key] = tensor.float().to(device, non_blocking=non_blocking)
    return out


def _patient_round_robin_samples(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        patient_id = str(sample.get("patient_id", sample.get("sample_id", "")))
        grouped.setdefault(patient_id, []).append(sample)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        added = False
        for patient_id in sorted(grouped):
            bucket = grouped[patient_id]
            if bucket:
                selected.append(bucket.pop(0))
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def _shard_samples(samples: list[dict[str, Any]], shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    if shard_count == 1:
        return list(samples)
    return [sample for index, sample in enumerate(samples) if index % shard_count == shard_index]


def _case_limit_arg(value: int | None) -> list[str]:
    if value is None or value <= 0:
        return []
    return ["--max-cases", str(value)]


def _run_artifact_command(cmd: list[str], label: str) -> None:
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} export failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr, flush=True)


def _export_visual_artifacts(args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    manifest_path = out_dir / "prediction_manifest.json"
    comparisons_dir = args.comparisons_dir or out_dir / "comparisons"
    viewer_dir = args.viewer_dir or out_dir / "viewer"
    visualize_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "visualize_difnet_v0b_predictions.py"),
        "--prediction-manifest",
        str(manifest_path),
        "--output-dir",
        str(comparisons_dir),
        "--slice-mode",
        args.visualization_slice_mode,
        "--num-slices",
        str(args.visualization_num_slices),
    ]
    visualize_cmd.extend(_case_limit_arg(args.visualization_max_cases))
    _run_artifact_command(visualize_cmd, "comparison")

    viewer_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "view_difnet_v0b_prediction.py"),
        "--prediction-manifest",
        str(manifest_path),
        "--output-dir",
        str(viewer_dir),
        "--slice-stride",
        str(args.viewer_slice_stride),
    ]
    viewer_cmd.extend(_case_limit_arg(args.viewer_max_cases))
    _run_artifact_command(viewer_cmd, "viewer")

    required = [
        comparisons_dir / "visualization_summary.json",
        viewer_dir / "index.html",
        viewer_dir / "viewer_data.json",
        viewer_dir / "viewer_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"test visualization artifact missing: {missing}")
    return {
        "comparisons_dir": str(comparisons_dir),
        "viewer_dir": str(viewer_dir),
        "viewer_html": str(viewer_dir / "index.html"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DIF-Net V0b test split inference.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-grid-size", type=int, default=DEFAULT_TEST_EVAL_GRID_SIZE)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--combine", choices=("auto", *COMBINE_CHOICES), default="auto")
    parser.add_argument("--mid-channels", default="auto")
    parser.add_argument("--routed-capsule-squash", choices=("auto", *CAPSULE_SQUASH_MODE_CHOICES), default="auto")
    parser.add_argument("--model-version", choices=("auto", *MODEL_VERSION_CHOICES), default="auto")
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
    parser.add_argument("--sample-strategy", choices=("first-n", "patient-round-robin"), default="first-n")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-visualization", action="store_true")
    volume_group = parser.add_mutually_exclusive_group()
    volume_group.add_argument("--save-volumes", dest="save_volumes", action="store_true")
    volume_group.add_argument("--no-save-volumes", dest="save_volumes", action="store_false")
    parser.set_defaults(save_volumes=True)
    parser.add_argument("--comparisons-dir", type=Path)
    parser.add_argument("--viewer-dir", type=Path)
    parser.add_argument("--visualization-max-cases", type=int, default=DEFAULT_TEST_VISUALIZATION_MAX_CASES)
    parser.add_argument("--visualization-slice-mode", choices=("axial", "coronal", "sagittal", "all"), default="all")
    parser.add_argument("--visualization-num-slices", type=int, default=1)
    parser.add_argument("--viewer-max-cases", type=int, default=DEFAULT_TEST_VIEWER_MAX_CASES)
    parser.add_argument("--viewer-slice-stride", type=int, default=DEFAULT_TEST_VIEWER_SLICE_STRIDE)
    args = parser.parse_args()
    if not args.save_volumes and not args.skip_visualization:
        parser.error("--no-save-volumes requires --skip-visualization")

    cache_manifest = load_cache_manifest(args.cache_root)
    cbct_normalization = cache_manifest["cbct_normalization"]
    view_key = view_key_from_count(args.view_count)
    out_dir = args.output_dir or PROJECT_ROOT / "runs" / "difnet_v0b" / f"test-{view_key}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(
        json.dumps(
            {
                "event": "device_resolved",
                "requested_device": args.device,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_view_count = resolve_checkpoint_view_count(checkpoint)
    if checkpoint_view_count is not None and checkpoint_view_count != args.view_count:
        raise ValueError(
            "checkpoint view_count does not match the requested cache branch: "
            f"checkpoint_view_count={checkpoint_view_count}, requested_view_count={args.view_count}. "
            "Use the checkpoint's trained view count and matching cache view branch."
        )
    model_version = resolve_checkpoint_model_version(checkpoint, requested=args.model_version)
    combine = resolve_checkpoint_combine(checkpoint, requested=args.combine)
    mid_channels = resolve_checkpoint_mid_channels(checkpoint, requested=args.mid_channels)
    routed_capsule_squash = resolve_checkpoint_routed_capsule_squash(checkpoint, requested=args.routed_capsule_squash)
    set_capsule_squash_mode(routed_capsule_squash)
    hashgrid_backend = resolve_checkpoint_hash_backend(checkpoint, requested=args.hashgrid_backend)
    capsule_kwargs = resolve_checkpoint_capsule_kwargs(
        checkpoint,
        {
            "capsule_hidden_dim": args.capsule_hidden_dim,
            "capsule_layers": args.capsule_layers,
            "capsule_heads": args.capsule_heads,
            "capsule_mlp_ratio": args.capsule_mlp_ratio,
            "capsule_dropout": args.capsule_dropout,
            "capsule_valid_mask": args.capsule_valid_mask,
        },
    )
    validate_capsule_kwargs_for_combine(combine, **capsule_kwargs)
    dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split=args.split,
        view_key=view_key,
        max_cases=args.max_test_cases if args.sample_strategy == "first-n" else None,
        deterministic=True,
        eval_grid_size=args.eval_grid_size,
        c2rv_grids=combine == "c2rv" or model_version == "c2rv-style",
    )
    if args.sample_strategy == "patient-round-robin" and args.max_test_cases is not None:
        dataset.samples = _patient_round_robin_samples(dataset.samples, args.max_test_cases)
    dataset.samples = _shard_samples(dataset.samples, args.shard_count, args.shard_index)
    if not dataset.samples:
        raise ValueError(f"no samples selected for shard_index={args.shard_index}, shard_count={args.shard_count}")
    model = build_difnet_model(
        PROJECT_ROOT,
        model_version=model_version,
        num_views=args.view_count,
        combine=combine,
        mid_channels=mid_channels,
        hashgrid_backend=hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
        **capsule_kwargs,
    ).to(device)
    try:
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint model state does not match the requested model structure: "
            f"model_version={model_version}, combine={combine}, mid_channels={mid_channels}. "
            "Use auto checkpoint config recovery or pass matching model parameters."
        ) from exc
    model.eval()
    resolved_model_config = model_config(
        model,
        {
            "model_version": "v0b",
            "num_views": args.view_count,
            "combine": combine,
            "mid_channels": mid_channels,
            "routed_capsule_squash": get_capsule_squash_mode(),
        },
    )

    prediction_rows = []
    metrics_rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            batch = _model_item(item, device)
            pred = model(batch, is_eval=True, eval_npoint=args.chunk_size)
            generated = pred[0, 0].detach().cpu().numpy().reshape(item["image"].shape).astype(np.float32)
            generated = np.clip(generated, 0.0, 1.0)
            target = np.asarray(item["image"], dtype=np.float32)
            metrics = metric_dict(generated, target, normalization=cbct_normalization)
            case_id = item["trace"]["case_key"]
            generated_path = None
            target_path = None
            if args.save_volumes:
                generated_path = out_dir / "generated_cbct" / f"{case_id}_{view_key}.npy"
                target_path = out_dir / "target_cbct" / f"{case_id}_{view_key}.npy"
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(generated_path, generated)
                np.save(target_path, target)
            trace = item["trace"]
            row = {
                "patient_id": trace["patient_id"],
                "case_id": trace["case_id"],
                "case_key": case_id,
                "augmentation_id": trace.get("augmentation_id"),
                "translation_mm": trace.get("translation_mm"),
                "rotation_deg": trace.get("rotation_deg"),
                "source_patient_id": trace.get("source_patient_id"),
                "source_case_id": trace.get("source_case_id"),
                "source_case_key": trace.get("source_case_key"),
                "view_count": args.view_count,
                "source_cache_item_path": trace["sample_path"],
                "generated_cbct_path": str(generated_path) if generated_path is not None else None,
                "target_cbct_path": str(target_path) if target_path is not None else None,
                "geometry_branch": trace.get("geometry_branch", cache_manifest.get("geometry_branch")),
                "processed_shape_zyx": trace.get("processed_shape_zyx"),
                "processed_spacing_xyz": trace.get("processed_spacing_xyz"),
                "processed_origin_xyz": trace.get("processed_origin_xyz"),
                "processed_direction": trace.get("processed_direction"),
                "drr_config": trace.get("drr_config"),
                "normalization": cbct_normalization,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "model_version": model_version,
                "model_config": resolved_model_config,
                "eval_grid_size": args.eval_grid_size,
                "metrics": metrics,
            }
            prediction_rows.append(row)
            metrics_rows.append(
                {
                    "case_key": case_id,
                    "augmentation_id": trace.get("augmentation_id"),
                    "translation_mm": trace.get("translation_mm"),
                    "rotation_deg": trace.get("rotation_deg"),
                    **metrics,
                }
            )

    summary = mean_metrics([{key: float(row[key]) for key in ("mae_norm", "mae_hu", "psnr", "ssim")} for row in metrics_rows])
    summary.update(
        {
            "case_count": len(metrics_rows),
            "split": args.split,
            "view_count": args.view_count,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "cache_version": cache_manifest["cache_version"],
            "geometry_branch": cache_manifest.get("geometry_branch"),
        }
    )
    summary["cbct_normalization"] = cbct_normalization
    manifest = {
        "schema": "x2cbct_difnet_v0b_predictions.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "cache_version": cache_manifest["cache_version"],
        "geometry_branch": cache_manifest.get("geometry_branch"),
        "axis_convention": cache_manifest.get("axis_convention"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "view_count": args.view_count,
        "view_key": view_key,
        "save_volumes": args.save_volumes,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "normalization": cbct_normalization,
        "model_version": model_version,
        "routed_capsule_squash": get_capsule_squash_mode(),
        "model_config": resolved_model_config,
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
