from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Iterator

import torch
from torch import nn


EFFICIENT_COMBINE_CHOICES = (
    "eff1_flat",
    "eff2_flat",
    "eff1_sar_mlp",
    "eff2_sar_mlp",
    "eff1_sar_hu",
    "eff2_sar_hu",
)
GAZE_COMBINE_CHOICES = ("gaze_sar_mlp", "gaze_sar_hu")
ROUTED_CAPSULE_COMBINE_CHOICES = (*EFFICIENT_COMBINE_CHOICES, *GAZE_COMBINE_CHOICES)
CAPSULE_FEATURE_PLANS = {
    32: {"primary_capsules_per_view": 8, "primary_capsule_dim": 4},
    64: {"primary_capsules_per_view": 16, "primary_capsule_dim": 4},
    128: {"primary_capsules_per_view": 16, "primary_capsule_dim": 8},
}
CAPSULE_SQUASH_MODE_CHOICES = ("stable", "legacy", "hinton")
_CAPSULE_SQUASH_MODE = "stable"
_CAPSULE_DIAGNOSTIC_STACK: list[list[dict[str, Any]]] = []


def get_capsule_squash_mode() -> str:
    return _CAPSULE_SQUASH_MODE


def set_capsule_squash_mode(mode: str) -> None:
    if mode not in CAPSULE_SQUASH_MODE_CHOICES:
        joined = ", ".join(CAPSULE_SQUASH_MODE_CHOICES)
        raise ValueError(f"unsupported capsule squash mode {mode!r}; expected one of: {joined}")
    global _CAPSULE_SQUASH_MODE
    _CAPSULE_SQUASH_MODE = mode


def _sampled_tensor_stats(tensor: torch.Tensor, *, thresholds: tuple[float, ...] = ()) -> dict[str, Any]:
    with torch.no_grad():
        detached = tensor.detach()
        flat = detached.reshape(-1)
        total_count = int(flat.numel())
        if total_count == 0:
            return {"total_count": 0, "finite": True}
        sample = flat
        max_samples = 1_000_000
        if total_count > max_samples:
            indices = (torch.arange(max_samples, device=flat.device, dtype=torch.long) * total_count) // max_samples
            sample = flat[indices]
        finite_sample = sample[torch.isfinite(sample)]
        out: dict[str, Any] = {
            "shape": list(detached.shape),
            "total_count": total_count,
            "sample_count": int(sample.numel()),
            "finite": bool(torch.isfinite(detached).all().item()),
        }
        if finite_sample.numel() > 0:
            values = finite_sample.float()
            out.update(
                {
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "mean": float(values.mean().item()),
                    "p99": float(torch.quantile(values, 0.99).item()),
                    "p999": float(torch.quantile(values, 0.999).item()),
                }
            )
        for threshold in thresholds:
            key = str(threshold).replace(".", "_")
            out[f"count_gt_{key}"] = int((detached > threshold).sum().item())
        return out


def _record_capsule_diagnostic(record: dict[str, Any]) -> None:
    if _CAPSULE_DIAGNOSTIC_STACK:
        _CAPSULE_DIAGNOSTIC_STACK[-1].append(record)


@contextmanager
def capture_capsule_diagnostics() -> Iterator[list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    _CAPSULE_DIAGNOSTIC_STACK.append(records)
    try:
        yield records
    finally:
        _CAPSULE_DIAGNOSTIC_STACK.pop()


def stable_efficient_squash(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return (1.0 - torch.exp(-norm)) * (x / (norm + eps))


def legacy_efficient_squash(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return (1.0 - 1.0 / (torch.exp(norm) + eps)) * (x / (norm + eps))


def hinton_squash(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    norm_sq = norm.square()
    return (norm_sq / (1.0 + norm_sq)) * (x / (norm + eps))


def efficient_squash(x: torch.Tensor, eps: float = 1e-8, *, source: str = "efficient_squash") -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    _record_capsule_diagnostic(
        {
            "event": "squash_input",
            "source": source,
            "mode": _CAPSULE_SQUASH_MODE,
            "dtype": str(x.dtype),
            "device": str(x.device),
            "norm": _sampled_tensor_stats(norm, thresholds=(80.0, 88.0, 89.0, 100.0)),
        }
    )
    if _CAPSULE_SQUASH_MODE == "legacy":
        return (1.0 - 1.0 / (torch.exp(norm) + eps)) * (x / (norm + eps))
    if _CAPSULE_SQUASH_MODE == "hinton":
        norm_sq = norm.square()
        return (norm_sq / (1.0 + norm_sq)) * (x / (norm + eps))
    return (1.0 - torch.exp(-norm)) * (x / (norm + eps))


def _capsule_plan(num_views: int, feature_channels: int) -> dict[str, int]:
    if int(num_views) not in (2, 5, 8, 10):
        raise ValueError(f"new capsule routing supports view_count 2, 5, 8, or 10, got {num_views}")
    feature_channels = int(feature_channels)
    if feature_channels not in CAPSULE_FEATURE_PLANS:
        raise ValueError(f"new capsule routing supports feature_channels 32, 64, or 128, got {feature_channels}")
    feature_plan = CAPSULE_FEATURE_PLANS[feature_channels]
    primary_capsules_per_view = int(feature_plan["primary_capsules_per_view"])
    primary_capsule_dim = int(feature_plan["primary_capsule_dim"])
    first_capsules_per_view = primary_capsules_per_view // 2
    first_capsule_dim = primary_capsule_dim * 2
    second_capsules_per_view = first_capsules_per_view // 2
    second_capsule_dim = first_capsule_dim * 2
    return {
        "feature_channels": feature_channels,
        "input_capsules_per_view": primary_capsules_per_view,
        "primary_capsule_dim": primary_capsule_dim,
        "input_capsules": int(num_views) * primary_capsules_per_view,
        "first_capsules_per_view": first_capsules_per_view,
        "first_capsule_dim": first_capsule_dim,
        "first_capsules": int(num_views) * first_capsules_per_view,
        "second_capsules_per_view": second_capsules_per_view,
        "second_capsule_dim": second_capsule_dim,
        "second_capsules": int(num_views) * second_capsules_per_view,
    }


class EfficientCapsuleRoutingLayer(nn.Module):
    def __init__(
        self,
        *,
        input_capsules: int,
        input_dim: int,
        output_capsules: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_capsules = int(input_capsules)
        self.input_dim = int(input_dim)
        self.output_capsules = int(output_capsules)
        self.output_dim = int(output_dim)
        self.W = nn.Parameter(torch.empty(self.output_capsules, self.input_capsules, self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.zeros(self.output_capsules, self.input_capsules, 1))
        nn.init.kaiming_normal_(self.W)

    def votes(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, I, d_in], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.input_capsules, self.input_dim):
            raise ValueError(
                f"expected input capsules [{self.input_capsules}, {self.input_dim}], got {tuple(x.shape[-2:])}"
            )
        return torch.einsum("bnid,jido->bnjio", x, self.W)

    def coupling_coefficients_from_votes(self, votes: torch.Tensor) -> torch.Tensor:
        logits = votes.square().sum(dim=-1, keepdim=True) / math.sqrt(float(self.output_dim))
        logits = logits + self.b.view(1, 1, self.output_capsules, self.input_capsules, 1)
        _record_capsule_diagnostic(
            {
                "event": "routing_logits",
                "source": "efficient_routing",
                "input_capsules": self.input_capsules,
                "output_capsules": self.output_capsules,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "dtype": str(logits.dtype),
                "device": str(logits.device),
                "logits": _sampled_tensor_stats(logits),
            }
        )
        # Parent-axis softmax makes each child capsule distribute its vote across parent capsules.
        return torch.softmax(logits, dim=2)

    def coupling_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        return self.coupling_coefficients_from_votes(self.votes(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        votes = self.votes(x)
        coupling = self.coupling_coefficients_from_votes(votes)
        parent_input = (coupling * votes).sum(dim=3)
        return efficient_squash(parent_input, source="efficient_parent")

    def info(self) -> dict[str, int]:
        return {
            "input_capsules": self.input_capsules,
            "input_dim": self.input_dim,
            "output_capsules": self.output_capsules,
            "output_dim": self.output_dim,
        }


class GazeSarLayer(nn.Module):
    def __init__(self, *, input_dim: int, output_capsules: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_capsules = int(output_capsules)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.empty(self.output_capsules, self.input_dim, self.output_dim))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, I, d_in], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")
        pred = torch.einsum("bnid,jdo->bnjio", x, self.weight)
        logits = torch.matmul(pred, pred.transpose(-1, -2)) / math.sqrt(float(self.output_dim))
        _record_capsule_diagnostic(
            {
                "event": "gaze_sar_logits",
                "source": "gaze_sar",
                "input_dim": self.input_dim,
                "output_capsules": self.output_capsules,
                "output_dim": self.output_dim,
                "dtype": str(logits.dtype),
                "device": str(logits.device),
                "logits": _sampled_tensor_stats(logits),
            }
        )
        attention = torch.softmax(logits, dim=-1)
        contextual = torch.matmul(attention, pred)
        return contextual.mean(dim=3)

    def info(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "output_capsules": self.output_capsules,
            "output_dim": self.output_dim,
        }


class GazeSarReadout(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer = GazeSarLayer(input_dim=input_dim, output_capsules=1, output_dim=output_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x).squeeze(2)

    def info(self) -> dict[str, int]:
        return {"input_dim": self.layer.input_dim, "output_dim": self.output_dim}


class _PrimaryCapsulesMixin:
    feature_channels: int
    num_views: int

    def _raw_primary_capsules(self, p_feats: torch.Tensor) -> torch.Tensor:
        if p_feats.ndim != 4:
            raise ValueError(f"p_feats must have shape [B, C, N, M], got {tuple(p_feats.shape)}")
        batch, channels, num_points, num_views = p_feats.shape
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels, got {channels}")
        if num_views != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {num_views}")
        capsules = (
            p_feats.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(
                batch,
                num_points,
                num_views * int(self.plan["input_capsules_per_view"]),
                int(self.plan["primary_capsule_dim"]),
            )
        )
        return capsules

    def _primary_capsules(self, p_feats: torch.Tensor) -> torch.Tensor:
        return efficient_squash(self._raw_primary_capsules(p_feats), source="primary_capsules")


class EfficientCapsuleFusion(nn.Module, _PrimaryCapsulesMixin):
    def __init__(self, *, feature_channels: int, num_views: int, combine: str) -> None:
        super().__init__()
        if combine not in EFFICIENT_COMBINE_CHOICES:
            raise ValueError(f"unsupported Efficient capsule combine: {combine}")
        self.feature_channels = int(feature_channels)
        self.num_views = int(num_views)
        self.combine = combine
        plan = _capsule_plan(num_views, feature_channels)
        self.plan = plan
        self.layer1 = EfficientCapsuleRoutingLayer(
            input_capsules=plan["input_capsules"],
            input_dim=plan["primary_capsule_dim"],
            output_capsules=plan["first_capsules"],
            output_dim=plan["first_capsule_dim"],
        )
        self.layer2: EfficientCapsuleRoutingLayer | None = None
        if combine.startswith("eff2_"):
            self.layer2 = EfficientCapsuleRoutingLayer(
                input_capsules=plan["first_capsules"],
                input_dim=plan["first_capsule_dim"],
                output_capsules=plan["second_capsules"],
                output_dim=plan["second_capsule_dim"],
            )
        readout_input_dim = plan["second_capsule_dim"] if self.layer2 is not None else plan["first_capsule_dim"]
        self.readout: GazeSarReadout | None = None
        if combine.endswith("_sar_mlp"):
            self.readout = GazeSarReadout(input_dim=readout_input_dim, output_dim=128)
            self.output_kind = "feature_128"
            self.output_channels = 128
        elif combine.endswith("_sar_hu"):
            self.readout = GazeSarReadout(input_dim=readout_input_dim, output_dim=1)
            self.output_kind = "direct_scalar"
            self.output_channels = 1
        else:
            self.output_kind = "flat_feature"
            self.output_channels = (
                plan["second_capsules"] * plan["second_capsule_dim"]
                if self.layer2 is not None
                else plan["first_capsules"] * plan["first_capsule_dim"]
            )

    def forward(self, p_feats: torch.Tensor, **_: Any) -> torch.Tensor:
        x = self.layer1(self._primary_capsules(p_feats))
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.readout is not None:
            out = self.readout(x).permute(0, 2, 1).contiguous()
            return out
        return x.flatten(2).permute(0, 2, 1).contiguous()

    def info(self) -> dict[str, Any]:
        layers = [self.layer1.info()]
        if self.layer2 is not None:
            layers.append(self.layer2.info())
        return {
            "family": "efficient_capsule_routing",
            "combine": self.combine,
            "feature_channels": self.feature_channels,
            "input_capsules": self.plan["input_capsules"],
            "input_capsules_per_view": self.plan["input_capsules_per_view"],
            "primary_capsule_dim": self.plan["primary_capsule_dim"],
            "backbone_layers": layers,
            "readout": self.output_kind,
            "output_channels": self.output_channels,
            "squash": get_capsule_squash_mode(),
            "uses_valid_mask": False,
            "uses_geometry_tokens": False,
        }


class GazeSarFusion(nn.Module, _PrimaryCapsulesMixin):
    def __init__(self, *, feature_channels: int, num_views: int, combine: str) -> None:
        super().__init__()
        if combine not in GAZE_COMBINE_CHOICES:
            raise ValueError(f"unsupported Gaze SAR combine: {combine}")
        self.feature_channels = int(feature_channels)
        self.num_views = int(num_views)
        self.combine = combine
        self.plan = _capsule_plan(num_views, feature_channels)
        self.layer1 = GazeSarLayer(
            input_dim=self.plan["primary_capsule_dim"],
            output_capsules=self.plan["first_capsules"],
            output_dim=self.plan["first_capsule_dim"],
        )
        readout_dim = 128 if combine == "gaze_sar_mlp" else 1
        self.readout = GazeSarReadout(input_dim=self.plan["first_capsule_dim"], output_dim=readout_dim)
        self.output_kind = "feature_128" if readout_dim == 128 else "direct_scalar"
        self.output_channels = readout_dim

    def forward(self, p_feats: torch.Tensor, **_: Any) -> torch.Tensor:
        x = efficient_squash(self.layer1(self._primary_capsules(p_feats)))
        return self.readout(x).permute(0, 2, 1).contiguous()

    def info(self) -> dict[str, Any]:
        return {
            "family": "gaze_sar",
            "combine": self.combine,
            "feature_channels": self.feature_channels,
            "input_capsules": self.plan["input_capsules"],
            "input_capsules_per_view": self.plan["input_capsules_per_view"],
            "primary_capsule_dim": self.plan["primary_capsule_dim"],
            "backbone_layers": [self.layer1.info()],
            "readout": self.output_kind,
            "output_channels": self.output_channels,
            "squash": get_capsule_squash_mode(),
            "uses_valid_mask": False,
            "uses_geometry_tokens": False,
        }
