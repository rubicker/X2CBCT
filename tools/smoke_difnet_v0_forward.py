from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from x2cbct_difnet.capsule_routing import CAPSULE_SQUASH_MODE_CHOICES, get_capsule_squash_mode, set_capsule_squash_mode
from x2cbct_difnet.official import synthetic_difnet_item, write_json_report
from x2cbct_difnet.v2_hash_model import COMBINE_CHOICES, HASHGRID_BACKEND_CHOICES, MODEL_VERSION_CHOICES, build_difnet_model, model_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official DIF-Net synthetic forward smoke.")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--mid-channels", type=int, default=128)
    parser.add_argument("--combine", choices=COMBINE_CHOICES, default="mlp")
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
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "difnet_v0" / "official_forward_smoke.json")
    args = parser.parse_args()
    set_capsule_squash_mode(args.routed_capsule_squash)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_difnet_model(
        PROJECT_ROOT,
        num_views=args.num_views,
        combine=args.combine,
        mid_channels=args.mid_channels,
        model_version=args.model_version,
        hashgrid_backend=args.hashgrid_backend,
        prefer_cuda_hashgrid=device.type == "cuda",
        capsule_hidden_dim=args.capsule_hidden_dim,
        capsule_layers=args.capsule_layers,
        capsule_heads=args.capsule_heads,
        capsule_mlp_ratio=args.capsule_mlp_ratio,
        capsule_dropout=args.capsule_dropout,
        capsule_valid_mask=args.capsule_valid_mask,
    )
    model.to(device)
    model.train(mode=args.backward)
    item = {key: value.to(device) for key, value in synthetic_difnet_item(args.num_views, args.image_size, args.num_points).items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(args.backward):
        pred, gt = model(item)
        loss_tensor = torch.nn.functional.mse_loss(pred, gt)
        if args.backward:
            loss_tensor.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loss = float(loss_tensor.detach().item())
    report = {
        "ok": True,
        "model_version": args.model_version,
        "model_config": model_config(
            model,
            {
                "model_version": "v0b",
                "num_views": args.num_views,
                "combine": args.combine,
                "mid_channels": args.mid_channels,
            },
        ),
        "num_views": args.num_views,
        "image_size": args.image_size,
        "num_points": args.num_points,
        "mid_channels": args.mid_channels,
        "combine": args.combine,
        "routed_capsule_squash": get_capsule_squash_mode(),
        "output_shape": list(pred.shape),
        "target_shape": list(gt.shape),
        "loss": loss,
        "device": str(next(model.parameters()).device),
        "backward": bool(args.backward),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_max_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3) if device.type == "cuda" else None
        ),
    }
    output = write_json_report(report, args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(output),
                "model_version": args.model_version,
                "loss": report["loss"],
                "device": report["device"],
                "hashgrid_backend": report["model_config"].get("hashgrid", {}).get("backend_actual"),
                "backward": report["backward"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
