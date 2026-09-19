from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .capsule_routing import (
    CAPSULE_SQUASH_MODE_CHOICES,
    EFFICIENT_COMBINE_CHOICES,
    GAZE_COMBINE_CHOICES,
    ROUTED_CAPSULE_COMBINE_CHOICES,
    EfficientCapsuleFusion,
    GazeSarFusion,
)
from .dufal_model import DuFalStyleDifNet

HASH_MODEL_VERSIONS = (
    "v2-hash-a1",
    "v2-hash-a1a",
    "v2-hash-a1b",
    "v2-hash-a2",
    "v2-hash-a2a",
    "v2-hash-a2b",
)
C2RV_MODEL_VERSION = "c2rv-style"
C2RV_COMBINE = "c2rv"
C2RV_GRID_SCALES = (16, 8, 4)
DUFAL_MODEL_VERSION = "dufal-style"
DUFAL_COMBINE = "dufal"

MODEL_VERSION_CHOICES = ("v0b", *HASH_MODEL_VERSIONS, C2RV_MODEL_VERSION, DUFAL_MODEL_VERSION)
HASHGRID_BACKEND_CHOICES = ("auto", "cuda", "torch")
COMBINE_CHOICES = ("mlp", "max", "capsule-a1", C2RV_COMBINE, DUFAL_COMBINE, *ROUTED_CAPSULE_COMBINE_CHOICES)


@dataclass(frozen=True)
class CapsuleConfig:
    hidden_dim: int
    layers: int = 2
    heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    valid_mask: bool = False
    proj_encoding: str = "fourier"
    angle_encoding: str = "circular_fourier"
    proj_fourier_bands: int = 8
    angle_fourier_bands: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "capsule-a1",
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "head_dim": self.hidden_dim // self.heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "valid_mask": self.valid_mask,
            "proj_encoding": self.proj_encoding,
            "angle_encoding": self.angle_encoding,
            "proj_fourier_bands": self.proj_fourier_bands,
            "angle_fourier_bands": self.angle_fourier_bands,
        }


def resolve_capsule_config(
    mid_channels: int,
    *,
    capsule_hidden_dim: int | None = None,
    capsule_layers: int | None = None,
    capsule_heads: int | None = None,
    capsule_mlp_ratio: float | None = None,
    capsule_dropout: float | None = None,
    capsule_valid_mask: bool | None = None,
) -> CapsuleConfig:
    hidden_dim = int(capsule_hidden_dim or mid_channels)
    default_heads = 8 if hidden_dim >= 256 else 4
    heads = int(capsule_heads or default_heads)
    if hidden_dim % heads != 0:
        raise ValueError(f"capsule hidden_dim {hidden_dim} must be divisible by heads {heads}")
    return CapsuleConfig(
        hidden_dim=hidden_dim,
        layers=int(capsule_layers or 2),
        heads=heads,
        mlp_ratio=float(capsule_mlp_ratio if capsule_mlp_ratio is not None else 2.0),
        dropout=float(capsule_dropout if capsule_dropout is not None else 0.0),
        valid_mask=bool(capsule_valid_mask) if capsule_valid_mask is not None else False,
    )


def validate_capsule_kwargs_for_combine(
    combine: str,
    *,
    capsule_hidden_dim: int | None,
    capsule_layers: int | None,
    capsule_heads: int | None,
    capsule_mlp_ratio: float | None,
    capsule_dropout: float | None,
    capsule_valid_mask: bool | None,
) -> None:
    if combine not in ROUTED_CAPSULE_COMBINE_CHOICES:
        return
    legacy_args = {
        "--capsule-hidden-dim": capsule_hidden_dim,
        "--capsule-layers": capsule_layers,
        "--capsule-heads": capsule_heads,
        "--capsule-mlp-ratio": capsule_mlp_ratio,
        "--capsule-dropout": capsule_dropout,
        "--capsule-valid-mask/--no-capsule-valid-mask": capsule_valid_mask,
    }
    used = [name for name, value in legacy_args.items() if value is not None]
    if used:
        joined = ", ".join(used)
        raise ValueError(f"combine={combine} does not support legacy --capsule-* arguments: {joined}")


@dataclass(frozen=True)
class HashGridConfig:
    num_levels: int = 16
    level_dim: int = 2
    base_resolution: int = 16
    log2_hashmap_size: int = 19
    input_size: float = 1.0

    @property
    def output_dim(self) -> int:
        return self.num_levels * self.level_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_levels": self.num_levels,
            "level_dim": self.level_dim,
            "base_resolution": self.base_resolution,
            "log2_hashmap_size": self.log2_hashmap_size,
            "input_size": self.input_size,
            "output_dim": self.output_dim,
            "input_range": [-self.input_size, self.input_size],
        }


class TorchHashGridEncoder(nn.Module):
    """Pure PyTorch compatibility backend for NAF-style hashgrid encoding."""

    def __init__(self, config: HashGridConfig) -> None:
        super().__init__()
        self.config = config
        self.max_params = 2 ** int(config.log2_hashmap_size)
        self.embeddings = nn.ParameterList()
        self.level_meta: list[tuple[int, int]] = []
        for level in range(config.num_levels):
            resolution = int(config.base_resolution) * (2**level)
            table_size = min(self.max_params, (resolution + 1) ** 3)
            emb = nn.Parameter(torch.empty(table_size, config.level_dim))
            nn.init.uniform_(emb, -1e-4, 1e-4)
            self.embeddings.append(emb)
            self.level_meta.append((resolution, table_size))

    @staticmethod
    def _hash_index(coords: torch.Tensor, table_size: int) -> torch.Tensor:
        primes = coords.new_tensor([1, 2654435761, 805459861], dtype=torch.long)
        hashed = torch.bitwise_xor(
            torch.bitwise_xor(coords[:, 0] * primes[0], coords[:, 1] * primes[1]),
            coords[:, 2] * primes[2],
        )
        return torch.remainder(hashed, int(table_size))

    @staticmethod
    def _dense_index(coords: torch.Tensor, resolution: int) -> torch.Tensor:
        side = int(resolution) + 1
        return coords[:, 0] + side * (coords[:, 1] + side * coords[:, 2])

    def _level_encode(self, points01: torch.Tensor, level: int) -> torch.Tensor:
        resolution, table_size = self.level_meta[level]
        emb = self.embeddings[level]
        scaled = points01 * float(resolution)
        base = torch.floor(scaled).to(torch.long)
        frac = scaled - base.to(dtype=scaled.dtype)
        out = torch.zeros(points01.shape[0], self.config.level_dim, device=points01.device, dtype=points01.dtype)
        use_dense_index = table_size == (resolution + 1) ** 3
        for corner in itertools.product((0, 1), repeat=3):
            offset = base.new_tensor(corner)
            coords = torch.clamp(base + offset, min=0, max=resolution)
            weight = torch.ones(points01.shape[0], device=points01.device, dtype=points01.dtype)
            for dim, bit in enumerate(corner):
                weight = weight * (frac[:, dim] if bit else (1.0 - frac[:, dim]))
            if use_dense_index:
                index = self._dense_index(coords, resolution)
            else:
                index = self._hash_index(coords, table_size)
            out = out + emb[index] * weight[:, None]
        return out

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[-1] != 3:
            raise ValueError(f"points must have shape [..., 3], got {tuple(points.shape)}")
        size = float(self.config.input_size)
        min_value = float(points.detach().min().item())
        max_value = float(points.detach().max().item())
        if min_value < -size - 1e-5 or max_value > size + 1e-5:
            raise ValueError(f"hashgrid input range [{min_value}, {max_value}] is outside [{-size}, {size}]")
        prefix = points.shape[:-1]
        flat = ((points.reshape(-1, 3) + size) / (2.0 * size)).clamp(0.0, 1.0)
        encoded = [self._level_encode(flat, level) for level in range(self.config.num_levels)]
        out = torch.cat(encoded, dim=1)
        return out.reshape(*prefix, self.config.output_dim)


class CudaHashGridEncoder(nn.Module):
    def __init__(self, project_root: str | Path, config: HashGridConfig) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA hashgrid backend was requested, but torch.cuda.is_available() is false")
        src_root = Path(project_root).resolve() / "external" / "naf_cbct_official" / "src"
        if not src_root.exists():
            raise FileNotFoundError(f"NAF-CBCT source tree not found: {src_root}")
        src_text = str(src_root)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)
        try:
            from encoder.hashencoder.hashgrid import HashEncoder  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import NAF CUDA hashgrid backend. Use --hashgrid-backend torch for compatibility "
                "smoke tests, or install/compile the NAF hashgrid CUDA extension for full V2 runs."
            ) from exc
        self.config = config
        self.encoder = HashEncoder(
            input_dim=3,
            num_levels=config.num_levels,
            level_dim=config.level_dim,
            base_resolution=config.base_resolution,
            log2_hashmap_size=config.log2_hashmap_size,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points, size=float(self.config.input_size))


class HashPositionEncoder(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        *,
        backend: str = "auto",
        config: HashGridConfig | None = None,
        prefer_cuda: bool = False,
    ) -> None:
        super().__init__()
        if backend not in HASHGRID_BACKEND_CHOICES:
            raise ValueError(f"unsupported hashgrid backend: {backend}")
        self.config = config or HashGridConfig()
        self.requested_backend = backend
        self.fallback_reason: str | None = None
        self.backend = self._build_backend(project_root, backend, prefer_cuda=prefer_cuda)

    def _build_backend(self, project_root: str | Path, backend: str, *, prefer_cuda: bool) -> nn.Module:
        if backend == "torch":
            self.actual_backend = "torch"
            return TorchHashGridEncoder(self.config)
        if backend == "cuda":
            self.actual_backend = "cuda"
            return CudaHashGridEncoder(project_root, self.config)
        if prefer_cuda and torch.cuda.is_available():
            try:
                self.actual_backend = "cuda"
                return CudaHashGridEncoder(project_root, self.config)
            except Exception as exc:
                self.fallback_reason = repr(exc)
                self.actual_backend = "torch"
                print(f"[hashgrid] auto backend fell back to torch: {self.fallback_reason}", file=sys.stderr)
                return TorchHashGridEncoder(self.config)
        self.actual_backend = "torch"
        return TorchHashGridEncoder(self.config)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        encoded = self.backend(points)
        if encoded.shape[-1] != self.config.output_dim:
            raise RuntimeError(f"hashgrid output dim mismatch: got {encoded.shape[-1]}, expected {self.config.output_dim}")
        return encoded.permute(0, 2, 1).contiguous()

    def info(self) -> dict[str, Any]:
        out = self.config.to_dict()
        out["backend_requested"] = self.requested_backend
        out["backend_actual"] = self.actual_backend
        out["backend_fallback_reason"] = self.fallback_reason
        return out


class HashPositionInjection(nn.Module):
    def __init__(self, method: str, mid_channels: int, hash_dim: int) -> None:
        super().__init__()
        if method not in HASH_MODEL_VERSIONS:
            raise ValueError(f"unsupported hash method: {method}")
        self.method = method
        self.mid_channels = int(mid_channels)
        self.hash_dim = int(hash_dim)
        if method in {"v2-hash-a1a"}:
            self.hash_gate = nn.Parameter(torch.ones(1, hash_dim, 1))
        elif method in {"v2-hash-a1b"}:
            self.point_gate = nn.Conv1d(hash_dim, 1, kernel_size=1)
            nn.init.zeros_(self.point_gate.weight)
            nn.init.zeros_(self.point_gate.bias)
        elif method in {"v2-hash-a2", "v2-hash-a2a", "v2-hash-a2b"}:
            self.position_map = nn.Conv1d(hash_dim, mid_channels, kernel_size=1)
            if method == "v2-hash-a2a":
                self.position_gate = nn.Parameter(torch.ones(1, mid_channels, 1))
            elif method == "v2-hash-a2b":
                self.point_gate = nn.Conv1d(hash_dim, 1, kernel_size=1)
                nn.init.zeros_(self.point_gate.weight)
                nn.init.zeros_(self.point_gate.bias)

    @property
    def classifier_input_channels(self) -> int:
        if self.method in {"v2-hash-a1", "v2-hash-a1a", "v2-hash-a1b"}:
            return self.mid_channels + self.hash_dim
        return self.mid_channels

    def forward(self, projection_feature: torch.Tensor, hash_feature: torch.Tensor) -> torch.Tensor:
        if self.method == "v2-hash-a1":
            return torch.cat([projection_feature, hash_feature], dim=1)
        if self.method == "v2-hash-a1a":
            return torch.cat([projection_feature, hash_feature * self.hash_gate], dim=1)
        if self.method == "v2-hash-a1b":
            gate = 1.0 + self.point_gate(hash_feature)
            return torch.cat([projection_feature, hash_feature * gate], dim=1)
        position_feature = self.position_map(hash_feature)
        if self.method == "v2-hash-a2a":
            position_feature = position_feature * self.position_gate
        elif self.method == "v2-hash-a2b":
            position_feature = position_feature * (1.0 + self.point_gate(hash_feature))
        return projection_feature + position_feature

    def info(self) -> dict[str, Any]:
        gate_type = {
            "v2-hash-a1": "none",
            "v2-hash-a1a": "hash-channel",
            "v2-hash-a1b": "point-scalar",
            "v2-hash-a2": "none",
            "v2-hash-a2a": "position-channel",
            "v2-hash-a2b": "point-scalar",
        }[self.method]
        return {
            "method": self.method,
            "hash_dim": self.hash_dim,
            "mid_channels": self.mid_channels,
            "classifier_input_channels": self.classifier_input_channels,
            "gate_type": gate_type,
            "gate_initial_value": 1.0 if gate_type != "none" else None,
        }


class CapsuleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(round(hidden_dim * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        residual = tokens
        x = self.norm1(tokens)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        tokens = residual + self.drop1(x)
        return tokens + self.mlp(self.norm2(tokens))


class CapsuleViewFusion(nn.Module):
    def __init__(self, feature_channels: int, num_views: int, config: CapsuleConfig) -> None:
        super().__init__()
        if config.proj_encoding != "fourier":
            raise ValueError(f"unsupported capsule proj encoding: {config.proj_encoding}")
        if config.angle_encoding != "circular_fourier":
            raise ValueError(f"unsupported capsule angle encoding: {config.angle_encoding}")
        self.feature_channels = int(feature_channels)
        self.num_views = int(num_views)
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.image_proj = nn.Linear(feature_channels, hidden_dim)
        self.proj_proj = nn.Linear(4 * config.proj_fourier_bands, hidden_dim)
        self.angle_proj = nn.Linear(2 * config.angle_fourier_bands, hidden_dim)
        self.proj_gate = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.angle_gate = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.summary_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.layers = nn.ModuleList(
            [CapsuleTransformerLayer(hidden_dim, config.heads, config.mlp_ratio, config.dropout) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, feature_channels)

    @staticmethod
    def _fourier_features(values: torch.Tensor, bands: int) -> torch.Tensor:
        freq = torch.pow(
            values.new_tensor(2.0),
            torch.arange(bands, device=values.device, dtype=values.dtype),
        )
        angles = values.unsqueeze(-1) * freq * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)

    def _proj_encoding(self, proj_points: torch.Tensor) -> torch.Tensor:
        if proj_points.shape[-1] != 2:
            raise ValueError(f"proj_points must have shape [B, M, N, 2], got {tuple(proj_points.shape)}")
        points = proj_points.permute(0, 2, 1, 3).contiguous()
        return self._fourier_features(points, self.config.proj_fourier_bands)

    def _angle_encoding(self, view_angles_deg: torch.Tensor, num_points: int) -> torch.Tensor:
        if view_angles_deg.ndim != 2:
            raise ValueError(f"view_angles_deg must have shape [B, M], got {tuple(view_angles_deg.shape)}")
        radians = view_angles_deg / 180.0 * math.pi
        freq = torch.pow(
            radians.new_tensor(2.0),
            torch.arange(self.config.angle_fourier_bands, device=radians.device, dtype=radians.dtype),
        )
        angles = radians.unsqueeze(-1) * freq
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoded[:, None, :, :].expand(-1, num_points, -1, -1)

    def _summary_attention_mask(self, valid_mask: torch.Tensor | None, *, batch: int, num_points: int, num_views: int, device: torch.device) -> torch.Tensor | None:
        if not self.config.valid_mask:
            return None
        if valid_mask is None:
            return None
        if valid_mask.shape != (batch, num_views, num_points):
            raise ValueError(f"valid_mask must have shape [B, M, N], got {tuple(valid_mask.shape)}")
        invalid = ~valid_mask.permute(0, 2, 1).reshape(batch * num_points, num_views).bool()
        token_count = num_views + 1
        mask = torch.zeros(batch * num_points, token_count, token_count, device=device, dtype=torch.bool)
        mask[:, 0, 1:] = invalid
        return mask.repeat_interleave(self.config.heads, dim=0)

    def forward(
        self,
        p_feats: torch.Tensor,
        *,
        proj_points: torch.Tensor,
        view_angles_deg: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if p_feats.ndim != 4:
            raise ValueError(f"p_feats must have shape [B, C, N, M], got {tuple(p_feats.shape)}")
        batch, channels, num_points, num_views = p_feats.shape
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels, got {channels}")
        if num_views != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {num_views}")
        image_token = self.image_proj(p_feats.permute(0, 2, 3, 1).contiguous())
        proj_token = self.proj_proj(self._proj_encoding(proj_points))
        angle_token = self.angle_proj(self._angle_encoding(view_angles_deg, num_points))
        view_tokens = image_token + self.proj_gate * proj_token + self.angle_gate * angle_token
        summary = self.summary_token.expand(batch, num_points, 1, -1)
        tokens = torch.cat([summary, view_tokens], dim=2).reshape(batch * num_points, num_views + 1, -1)
        attn_mask = self._summary_attention_mask(
            valid_mask,
            batch=batch,
            num_points=num_points,
            num_views=num_views,
            device=tokens.device,
        )
        for layer in self.layers:
            tokens = layer(tokens, attn_mask)
        summary_out = self.norm(tokens[:, 0, :]).reshape(batch, num_points, -1)
        fused = self.out_proj(summary_out)
        return fused.permute(0, 2, 1).contiguous()

    def info(self) -> dict[str, Any]:
        out = self.config.to_dict()
        out["feature_channels"] = self.feature_channels
        out["num_views"] = self.num_views
        out["gate_type"] = "encoding-class-scalar"
        return out


def _load_official_components(project_root: str | Path):
    code_path = Path(project_root).resolve() / "external" / "dif_net_official" / "code"
    if not code_path.exists():
        raise FileNotFoundError(f"official DIF-Net code not found: {code_path}")
    code_path_str = str(code_path)
    if code_path_str not in sys.path:
        sys.path.insert(0, code_path_str)
    from models.model import MLP, index_2d  # type: ignore
    from models.point_classifier import SurfaceClassifier  # type: ignore
    from models.unet import UNet  # type: ignore

    return UNet, SurfaceClassifier, MLP, index_2d


def _divisible_heads(channels: int) -> int:
    for heads in (8, 4, 2, 1):
        if int(channels) % heads == 0:
            return heads
    return 1


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class C2RVResidual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class C2RVScaleViewAttentionLayer(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm_views = nn.LayerNorm(channels)
        self.view_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_query = nn.LayerNorm(channels)
        self.norm_key = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        hidden = int(round(channels * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, voxel_feature: torch.Tensor, view_features: torch.Tensor) -> torch.Tensor:
        if voxel_feature.ndim != 3:
            raise ValueError(f"voxel_feature must have shape [B, C, N], got {tuple(voxel_feature.shape)}")
        if view_features.ndim != 4:
            raise ValueError(f"view_features must have shape [B, C, N, M], got {tuple(view_features.shape)}")
        batch, channels, num_points = voxel_feature.shape
        if view_features.shape[:3] != (batch, channels, num_points):
            raise ValueError(
                "view_features shape must match voxel feature batch/channel/point dimensions: "
                f"voxel={tuple(voxel_feature.shape)}, views={tuple(view_features.shape)}"
            )
        num_views = view_features.shape[3]
        views = view_features.permute(0, 2, 3, 1).contiguous().reshape(batch * num_points, num_views, channels)
        views_norm = self.norm_views(views)
        view_delta, _ = self.view_attn(views_norm, views_norm, views_norm, need_weights=False)
        views = views + view_delta

        query = voxel_feature.permute(0, 2, 1).contiguous().reshape(batch * num_points, 1, channels)
        cross_delta, _ = self.cross_attn(
            self.norm_query(query),
            self.norm_key(views),
            self.norm_key(views),
            need_weights=False,
        )
        fused = query + cross_delta
        fused = fused + self.ffn(self.norm_ffn(fused))
        return fused.reshape(batch, num_points, channels).permute(0, 2, 1).contiguous()


class C2RVScaleViewFusion(nn.Module):
    def __init__(self, channels: int, layers: int = 2, heads: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.layers = int(layers)
        self.heads = int(heads or _divisible_heads(channels))
        self.blocks = nn.ModuleList(
            [C2RVScaleViewAttentionLayer(self.channels, self.heads) for _ in range(self.layers)]
        )

    def forward(self, voxel_feature: torch.Tensor, view_features: torch.Tensor) -> torch.Tensor:
        out = voxel_feature
        for block in self.blocks:
            out = block(out, view_features)
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": "c2rv-scale-view-attention",
            "feature_channels": self.channels,
            "layers": self.layers,
            "heads": self.heads,
        }


class C2RVStyleDifNet(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        *,
        num_views: int,
        mid_ch: int = 128,
        grid_scales: tuple[int, ...] = C2RV_GRID_SCALES,
    ) -> None:
        super().__init__()
        UNet, SurfaceClassifier, _MLP, index_2d = _load_official_components(project_root)
        self.index_2d = index_2d
        self.model_version = C2RV_MODEL_VERSION
        self.combine = C2RV_COMBINE
        self.num_views = int(num_views)
        self.mid_ch = int(mid_ch)
        self.grid_scales = tuple(int(item) for item in grid_scales)
        self.image_encoder = UNet(1, mid_ch)
        self.ms3dv_blocks = nn.ModuleList([C2RVResidual3DBlock(mid_ch) for _ in self.grid_scales])
        self.ms3dv_project = nn.Sequential(
            nn.Conv1d(mid_ch * len(self.grid_scales), mid_ch, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )
        self.scale_view_fusion = C2RVScaleViewFusion(mid_ch, layers=2)
        self.point_classifier = SurfaceClassifier([mid_ch, 256, 64, 16, 1], no_residual=False)
        print(f"C2RVStyleDifNet, mid_ch: {mid_ch}, views: {self.num_views}, grid_scales: {self.grid_scales}")

    def _encode_projections(self, projs: torch.Tensor) -> list[torch.Tensor]:
        batch, num_views, channels, width, height = projs.shape
        flat = projs.reshape(batch * num_views, channels, width, height)
        proj_feats = self.image_encoder(flat)
        proj_feats = list(proj_feats) if isinstance(proj_feats, tuple) else [proj_feats]
        for i, feat in enumerate(proj_feats):
            _, feat_channels, feat_width, feat_height = feat.shape
            proj_feats[i] = feat.reshape(batch, num_views, feat_channels, feat_width, feat_height)
        return proj_feats

    def _projection_scale_features(self, proj_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        base = proj_feats[0]
        batch, num_views, channels, width, height = base.shape
        flat = base.reshape(batch * num_views, channels, width, height)
        out: list[torch.Tensor] = []
        for scale_index in range(len(self.grid_scales)):
            factor = 2 ** scale_index
            feat = flat if factor == 1 else F.avg_pool2d(flat, kernel_size=factor, stride=factor)
            _, feat_channels, feat_width, feat_height = feat.shape
            out.append(feat.reshape(batch, num_views, feat_channels, feat_width, feat_height))
        return out

    def _require_tensor(self, data: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in data:
            raise ValueError(
                f"combine=c2rv requires data['{key}']; construct CachedDifNetV0bDataset with c2rv_grids=True"
            )
        return data[key]

    def _query_view_features(
        self,
        features: torch.Tensor,
        proj_points: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_views = features.shape[1]
        p_list = []
        for view_index in range(num_views):
            p_list.append(self.index_2d(features[:, view_index, ...], proj_points[:, view_index, ...]))
        out = torch.stack(p_list, dim=-1)
        if valid_mask is not None:
            mask = valid_mask.permute(0, 2, 1).unsqueeze(1).bool()
            out = torch.where(mask, out, torch.zeros_like(out))
        return out

    def _back_project_scale(
        self,
        features: torch.Tensor,
        proj_points: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        view_features = self._query_view_features(features, proj_points, valid_mask)
        mask = valid_mask.permute(0, 2, 1).unsqueeze(1).bool()
        neg = -torch.finfo(view_features.dtype).max
        pooled = view_features.masked_fill(~mask, neg).max(dim=-1).values
        any_valid = mask.any(dim=-1)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return pooled.reshape(pooled.shape[0], pooled.shape[1], grid_size, grid_size, grid_size).contiguous()

    def _build_ms3dv_volumes(
        self,
        proj_feats: list[torch.Tensor],
        data: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        scale_features = self._projection_scale_features(proj_feats)
        volumes = []
        for scale_index, grid_size in enumerate(self.grid_scales):
            proj_points = self._require_tensor(data, f"c2rv_grid_proj_points_s{scale_index}")
            valid_mask = self._require_tensor(data, f"c2rv_grid_valid_mask_s{scale_index}").bool()
            volume = self._back_project_scale(scale_features[scale_index], proj_points, valid_mask, grid_size)
            volumes.append(self.ms3dv_blocks[scale_index](volume))
        return volumes

    @staticmethod
    def _query_volume_feature(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        grid = points.reshape(points.shape[0], points.shape[1], 1, 1, 3)
        sampled = F.grid_sample(volume, grid, align_corners=True, padding_mode="zeros")
        return sampled[:, :, :, 0, 0]

    def _ms3dv_features(
        self,
        proj_feats: list[torch.Tensor],
        data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        volumes = data.get("_c2rv_volume_features")
        if volumes is None:
            volumes = self._build_ms3dv_volumes(proj_feats, data)
        queried = [self._query_volume_feature(volume, data["points"]) for volume in volumes]
        return self.ms3dv_project(torch.cat(queried, dim=1))

    def forward_points(self, proj_feats: list[torch.Tensor], data: dict[str, torch.Tensor]) -> torch.Tensor:
        scale_features = self._projection_scale_features(proj_feats)
        view_features = self._query_view_features(
            scale_features[0],
            data["proj_points"],
            data.get("valid_mask"),
        )
        voxel_feature = self._ms3dv_features(proj_feats, data)
        point_feature = self.scale_view_fusion(voxel_feature, view_features)
        return self.point_classifier(point_feature)

    def forward(self, data: dict[str, torch.Tensor], is_eval: bool = False, eval_npoint: int = 100000):
        proj_feats = self._encode_projections(data["projs"])
        if not is_eval:
            return self.forward_points(proj_feats, data), data["p_gt"]
        volumes = self._build_ms3dv_volumes(proj_feats, data)
        total_npoint = data["proj_points"].shape[2]
        pred_list = []
        for left in range(0, total_npoint, eval_npoint):
            right = min(left + eval_npoint, total_npoint)
            chunk = {
                "proj_points": data["proj_points"][..., left:right, :],
                "points": data["points"][..., left:right, :],
                "_c2rv_volume_features": volumes,
            }
            if "valid_mask" in data:
                chunk["valid_mask"] = data["valid_mask"][..., left:right]
            pred_list.append(self.forward_points(proj_feats, chunk))
        return torch.cat(pred_list, dim=2)

    def model_config(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "num_views": self.num_views,
            "combine": self.combine,
            "mid_channels": self.mid_ch,
            "c2rv": {
                "grid_scales": list(self.grid_scales),
                "ms3dv": "back_project_max_pool_3d_residual",
                "scale_view_attention": self.scale_view_fusion.info(),
            },
        }


class V2HashDifNet(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        *,
        num_views: int,
        combine: str,
        mid_ch: int = 128,
        method: str,
        hashgrid_backend: str = "auto",
        prefer_cuda_hashgrid: bool = False,
        hashgrid_config: HashGridConfig | None = None,
        capsule_config: CapsuleConfig | None = None,
    ) -> None:
        super().__init__()
        if method not in MODEL_VERSION_CHOICES:
            raise ValueError(f"unsupported model_version: {method}")
        if combine not in COMBINE_CHOICES:
            raise ValueError(f"unsupported combine mode: {combine}")
        UNet, SurfaceClassifier, MLP, index_2d = _load_official_components(project_root)
        self.index_2d = index_2d
        self.model_version = method
        self.num_views = int(num_views)
        self.combine = combine
        self.mid_ch = int(mid_ch)
        self.image_encoder = UNet(1, mid_ch)
        self.routed_capsule_fusion: EfficientCapsuleFusion | GazeSarFusion | None = None
        if combine == "mlp":
            self.view_mixer = MLP([num_views, max(num_views // 2, 1), 1])
        elif combine == "capsule-a1":
            self.view_fusion = CapsuleViewFusion(
                mid_ch,
                num_views,
                capsule_config or resolve_capsule_config(mid_ch),
            )
        elif combine in EFFICIENT_COMBINE_CHOICES:
            self.routed_capsule_fusion = EfficientCapsuleFusion(
                feature_channels=mid_ch,
                num_views=num_views,
                combine=combine,
            )
        elif combine in GAZE_COMBINE_CHOICES:
            self.routed_capsule_fusion = GazeSarFusion(
                feature_channels=mid_ch,
                num_views=num_views,
                combine=combine,
            )
        self.position_encoder: HashPositionEncoder | None = None
        self.position_injection: HashPositionInjection | None = None
        classifier_input_channels = mid_ch
        if method in HASH_MODEL_VERSIONS:
            self.position_encoder = HashPositionEncoder(
                project_root,
                backend=hashgrid_backend,
                config=hashgrid_config,
                prefer_cuda=prefer_cuda_hashgrid,
            )
            self.position_injection = HashPositionInjection(method, mid_ch, self.position_encoder.config.output_dim)
            classifier_input_channels = self.position_injection.classifier_input_channels
        if self.routed_capsule_fusion is not None:
            classifier_input_channels = self.routed_capsule_fusion.output_channels
        self.point_classifier = None
        if self.routed_capsule_fusion is None or self.routed_capsule_fusion.output_kind != "direct_scalar":
            self.point_classifier = SurfaceClassifier(
                [classifier_input_channels, 256, 64, 16, 1],
                no_residual=False,
            )
        hash_backend = self.position_encoder.actual_backend if self.position_encoder is not None else "none"
        print(
            f"V2HashDifNet, method: {method}, mid_ch: {mid_ch}, combine: {combine}, "
            f"hashgrid_backend: {hash_backend}"
        )

    def _encode_projections(self, projs: torch.Tensor) -> list[torch.Tensor]:
        b, m, c, w, h = projs.shape
        flat = projs.reshape(b * m, c, w, h)
        proj_feats = self.image_encoder(flat)
        proj_feats = list(proj_feats) if isinstance(proj_feats, tuple) else [proj_feats]
        for i, feat in enumerate(proj_feats):
            _, c_, w_, h_ = feat.shape
            proj_feats[i] = feat.reshape(b, m, c_, w_, h_)
        return proj_feats

    def _fuse_view_features(self, proj_feats: list[torch.Tensor], data: dict[str, torch.Tensor]) -> torch.Tensor:
        n_view = proj_feats[0].shape[1]
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...]
                p = data["proj_points"][:, i, ...]
                f_list.append(self.index_2d(feat, p))
            p_list.append(torch.cat(f_list, dim=1))
        p_feats = torch.stack(p_list, dim=-1)
        if self.combine == "max":
            return F.max_pool2d(p_feats, (1, n_view)).squeeze(-1)
        if self.combine == "mlp":
            return self.view_mixer(p_feats.permute(0, 3, 1, 2)).squeeze(1)
        if self.combine == "capsule-a1":
            if "view_angles_deg" not in data:
                raise ValueError("combine=capsule-a1 requires data['view_angles_deg'] with shape [B, M]")
            return self.view_fusion(
                p_feats,
                proj_points=data["proj_points"],
                view_angles_deg=data["view_angles_deg"],
                valid_mask=data.get("valid_mask"),
            )
        if self.routed_capsule_fusion is not None:
            return self.routed_capsule_fusion(
                p_feats,
                valid_mask=data.get("valid_mask"),
                view_angles_deg=data.get("view_angles_deg"),
            )
        raise NotImplementedError

    def forward_points(self, proj_feats: list[torch.Tensor], data: dict[str, torch.Tensor]) -> torch.Tensor:
        projection_feature = self._fuse_view_features(proj_feats, data)
        if self.routed_capsule_fusion is not None and self.routed_capsule_fusion.output_kind == "direct_scalar":
            return projection_feature
        if self.position_encoder is not None and self.position_injection is not None:
            hash_feature = self.position_encoder(data["points"])
            point_feature = self.position_injection(projection_feature, hash_feature)
        else:
            point_feature = projection_feature
        if self.point_classifier is None:
            raise RuntimeError("point classifier is unavailable for direct scalar capsule output")
        return self.point_classifier(point_feature)

    def forward(self, data: dict[str, torch.Tensor], is_eval: bool = False, eval_npoint: int = 100000):
        proj_feats = self._encode_projections(data["projs"])
        if not is_eval:
            return self.forward_points(proj_feats, data), data["p_gt"]
        total_npoint = data["proj_points"].shape[2]
        pred_list = []
        for left in range(0, total_npoint, eval_npoint):
            right = min(left + eval_npoint, total_npoint)
            pred_list.append(
                self.forward_points(
                    proj_feats,
                    {
                        "proj_points": data["proj_points"][..., left:right, :],
                        "points": data["points"][..., left:right, :],
                        **(
                            {"view_angles_deg": data["view_angles_deg"]}
                            if "view_angles_deg" in data
                            else {}
                        ),
                        **(
                            {"valid_mask": data["valid_mask"][..., left:right]}
                            if "valid_mask" in data
                            else {}
                        ),
                    },
                )
            )
        return torch.cat(pred_list, dim=2)

    def model_config(self) -> dict[str, Any]:
        out = {
            "model_version": self.model_version,
            "num_views": self.num_views,
            "combine": self.combine,
            "mid_channels": self.mid_ch,
        }
        if self.position_encoder is not None:
            out["hashgrid"] = self.position_encoder.info()
        if self.position_injection is not None:
            out["position_injection"] = self.position_injection.info()
        if self.combine == "capsule-a1":
            out["capsule"] = self.view_fusion.info()
        if self.routed_capsule_fusion is not None:
            out["capsule_routing"] = self.routed_capsule_fusion.info()
        return out


def build_difnet_model(
    project_root: str | Path,
    *,
    model_version: str,
    num_views: int,
    combine: str,
    mid_channels: int,
    hashgrid_backend: str = "auto",
    prefer_cuda_hashgrid: bool = False,
    capsule_hidden_dim: int | None = None,
    capsule_layers: int | None = None,
    capsule_heads: int | None = None,
    capsule_mlp_ratio: float | None = None,
    capsule_dropout: float | None = None,
    capsule_valid_mask: bool | None = None,
) -> nn.Module:
    if model_version not in MODEL_VERSION_CHOICES:
        raise ValueError(f"unsupported model_version: {model_version}")
    if combine not in COMBINE_CHOICES:
        raise ValueError(f"unsupported combine mode: {combine}")
    validate_capsule_kwargs_for_combine(
        combine,
        capsule_hidden_dim=capsule_hidden_dim,
        capsule_layers=capsule_layers,
        capsule_heads=capsule_heads,
        capsule_mlp_ratio=capsule_mlp_ratio,
        capsule_dropout=capsule_dropout,
        capsule_valid_mask=capsule_valid_mask,
    )
    if combine == C2RV_COMBINE:
        if model_version not in {"v0b", C2RV_MODEL_VERSION}:
            raise ValueError(f"combine={C2RV_COMBINE} requires model_version={C2RV_MODEL_VERSION}")
        return C2RVStyleDifNet(
            project_root,
            num_views=num_views,
            mid_ch=mid_channels,
        )
    if model_version == C2RV_MODEL_VERSION:
        raise ValueError(f"model_version={C2RV_MODEL_VERSION} requires combine={C2RV_COMBINE}")
    if combine == DUFAL_COMBINE:
        if model_version != DUFAL_MODEL_VERSION:
            raise ValueError(f"combine={DUFAL_COMBINE} requires model_version={DUFAL_MODEL_VERSION}")
        return DuFalStyleDifNet(
            num_views=num_views,
            mid_ch=mid_channels,
        )
    if model_version == DUFAL_MODEL_VERSION:
        raise ValueError(f"model_version={DUFAL_MODEL_VERSION} requires combine={DUFAL_COMBINE}")
    if combine in ROUTED_CAPSULE_COMBINE_CHOICES:
        if model_version != "v0b":
            raise ValueError(f"combine={combine} requires model_version=v0b")
        if int(mid_channels) not in (32, 64, 128):
            raise ValueError(f"combine={combine} requires mid_channels 32, 64, or 128")
    if model_version == "v0b" and combine != "capsule-a1":
        if combine in ROUTED_CAPSULE_COMBINE_CHOICES:
            return V2HashDifNet(
                project_root,
                num_views=num_views,
                combine=combine,
                mid_ch=mid_channels,
                method=model_version,
                hashgrid_backend=hashgrid_backend,
                prefer_cuda_hashgrid=prefer_cuda_hashgrid,
            )
        from .official import load_official_difnet

        DIF_Net = load_official_difnet(project_root)
        return DIF_Net(num_views=num_views, combine=combine, mid_ch=mid_channels)
    capsule_config = (
        resolve_capsule_config(
            mid_channels,
            capsule_hidden_dim=capsule_hidden_dim,
            capsule_layers=capsule_layers,
            capsule_heads=capsule_heads,
            capsule_mlp_ratio=capsule_mlp_ratio,
            capsule_dropout=capsule_dropout,
            capsule_valid_mask=capsule_valid_mask,
        )
        if combine == "capsule-a1"
        else None
    )
    return V2HashDifNet(
        project_root,
        num_views=num_views,
        combine=combine,
        mid_ch=mid_channels,
        method=model_version,
        hashgrid_backend=hashgrid_backend,
        prefer_cuda_hashgrid=prefer_cuda_hashgrid,
        capsule_config=capsule_config,
    )


def model_config(model: nn.Module, fallback: dict[str, Any]) -> dict[str, Any]:
    if hasattr(model, "model_config"):
        return getattr(model, "model_config")()
    return dict(fallback)


def resolve_checkpoint_model_version(checkpoint: Any, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in MODEL_VERSION_CHOICES:
            raise ValueError(f"unsupported model_version: {requested}")
        return requested
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict):
                value = str(model_cfg.get("model_version", ""))
                if value in MODEL_VERSION_CHOICES:
                    return value
            value = str(config.get("model_version", "v0b"))
            if value in MODEL_VERSION_CHOICES:
                return value
    return "v0b"


def resolve_checkpoint_combine(checkpoint: Any, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in COMBINE_CHOICES:
            raise ValueError(f"unsupported combine mode: {requested}")
        return requested
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict):
                value = str(model_cfg.get("combine", ""))
                if value in COMBINE_CHOICES:
                    return value
            value = str(config.get("combine", "mlp"))
            if value in COMBINE_CHOICES:
                return value
    return "mlp"


def resolve_checkpoint_mid_channels(checkpoint: Any, requested: str | int | None = "auto") -> int:
    if requested is not None and str(requested) != "auto":
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


def resolve_checkpoint_view_count(checkpoint: Any) -> int | None:
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict) and model_cfg.get("num_views") is not None:
                return int(model_cfg["num_views"])
            if config.get("view_count") is not None:
                return int(config["view_count"])
    return None


def resolve_checkpoint_routed_capsule_squash(checkpoint: Any, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in CAPSULE_SQUASH_MODE_CHOICES:
            raise ValueError(f"unsupported routed capsule squash mode: {requested}")
        return requested
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            value = str(config.get("routed_capsule_squash", ""))
            if value in CAPSULE_SQUASH_MODE_CHOICES:
                return value
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict):
                capsule_routing = model_cfg.get("capsule_routing")
                if isinstance(capsule_routing, dict):
                    value = str(capsule_routing.get("squash", ""))
                    if value in CAPSULE_SQUASH_MODE_CHOICES:
                        return value
    return "stable"


def resolve_checkpoint_capsule_kwargs(checkpoint: Any, requested: dict[str, Any] | None = None) -> dict[str, Any]:
    requested = dict(requested or {})
    capsule_cfg: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict):
                raw_capsule = model_cfg.get("capsule")
                if isinstance(raw_capsule, dict):
                    capsule_cfg = raw_capsule
    field_map = {
        "capsule_hidden_dim": "hidden_dim",
        "capsule_layers": "layers",
        "capsule_heads": "heads",
        "capsule_mlp_ratio": "mlp_ratio",
        "capsule_dropout": "dropout",
        "capsule_valid_mask": "valid_mask",
    }
    out: dict[str, Any] = {}
    for arg_name, cfg_name in field_map.items():
        if requested.get(arg_name) is not None:
            out[arg_name] = requested[arg_name]
        elif cfg_name in capsule_cfg:
            out[arg_name] = capsule_cfg[cfg_name]
        else:
            out[arg_name] = None
    return out


def resolve_checkpoint_hash_backend(checkpoint: Any, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in HASHGRID_BACKEND_CHOICES:
            raise ValueError(f"unsupported hashgrid backend: {requested}")
        return requested
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            model_cfg = config.get("model_config")
            if isinstance(model_cfg, dict):
                model_hashgrid = model_cfg.get("hashgrid")
                if isinstance(model_hashgrid, dict):
                    value = str(model_hashgrid.get("backend_actual", "auto"))
                    if value in HASHGRID_BACKEND_CHOICES:
                        return value
            hashgrid = config.get("hashgrid")
            if isinstance(hashgrid, dict):
                value = str(hashgrid.get("backend_requested", "auto"))
                if value in HASHGRID_BACKEND_CHOICES:
                    return value
            value = str(config.get("hashgrid_backend", "auto"))
            if value in HASHGRID_BACKEND_CHOICES:
                return value
    return "auto"
