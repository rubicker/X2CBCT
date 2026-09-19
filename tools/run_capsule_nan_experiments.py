from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.capsule_routing import (
    EfficientCapsuleFusion,
    hinton_squash,
    legacy_efficient_squash,
    set_capsule_squash_mode,
    stable_efficient_squash,
)


def _json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _finite(tensor: torch.Tensor | None) -> bool | None:
    if tensor is None:
        return None
    return bool(torch.isfinite(tensor.detach()).all().item())


def _math_experiment() -> dict[str, Any]:
    fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "legacy": legacy_efficient_squash,
        "stable": stable_efficient_squash,
        "hinton": hinton_squash,
    }
    rows = []
    for norm_target in [70.0, 80.0, 85.0, 88.0, 88.5, 89.0, 90.0, 100.0, 200.0]:
        for name, fn in fns.items():
            x = torch.full((1, 4), norm_target / 2.0, dtype=torch.float32, requires_grad=True)
            anomaly_error = None
            y = None
            try:
                y = fn(x)
                y.sum().backward()
            except RuntimeError as exc:
                anomaly_error = str(exc).splitlines()[0]
            rows.append(
                {
                    "formula": name,
                    "norm_target": norm_target,
                    "forward_finite": _finite(y) if isinstance(y, torch.Tensor) else False,
                    "backward_finite": _finite(x.grad),
                    "output_norm": (
                        float(torch.linalg.vector_norm(y.detach(), dim=-1).max().item())
                        if isinstance(y, torch.Tensor) and _finite(y)
                        else None
                    ),
                    "gradient_max_abs": (
                        float(torch.nan_to_num(x.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                        if x.grad is not None
                        else None
                    ),
                    "anomaly_error": anomaly_error,
                }
            )
    return {"experiment": "math", "rows": rows}


def _fusion_experiment(device: torch.device) -> dict[str, Any]:
    rows = []
    for mode in ["legacy", "stable", "hinton"]:
        set_capsule_squash_mode(mode)
        for scale in [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]:
            torch.manual_seed(20260617)
            fusion = EfficientCapsuleFusion(feature_channels=64, num_views=5, combine="eff1_sar_mlp").to(device)
            p_feats = torch.full((1, 64, 256, 5), float(scale), dtype=torch.float32, device=device, requires_grad=True)
            raw_primary = fusion._raw_primary_capsules(p_feats)
            loss = fusion(p_feats).square().mean()
            try:
                loss.backward()
                error = None
            except RuntimeError as exc:
                error = str(exc).splitlines()[0]
            rows.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "primary_norm_max": float(torch.linalg.vector_norm(raw_primary.detach(), dim=-1).max().item()),
                    "loss": float(loss.detach().item()) if _finite(loss) else None,
                    "loss_finite": _finite(loss),
                    "p_feats_grad_finite": _finite(p_feats.grad),
                    "parameter_grad_finite": all(
                        _finite(parameter.grad) is not False for parameter in fusion.parameters() if parameter.grad is not None
                    ),
                    "error": error,
                }
            )
    return {"experiment": "fusion", "device": str(device), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run routed-capsule NaN micro experiments.")
    parser.add_argument("--experiment", choices=("math", "fusion"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(args.device)
    if args.experiment == "math":
        report = _math_experiment()
    else:
        report = _fusion_experiment(device)
    report.update({"schema": "x2cbct_capsule_nan_micro_experiment.v1", "generated_at": datetime.now().isoformat(timespec="seconds")})
    _json_write(report, args.output)
    print(json.dumps({"ok": True, "report": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
