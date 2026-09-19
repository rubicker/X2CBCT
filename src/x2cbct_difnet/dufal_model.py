from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def index_2d(feat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    uv = uv.unsqueeze(2)
    feat = feat.transpose(2, 3)
    sampled = F.grid_sample(feat, uv, align_corners=True, padding_mode="zeros")
    return sampled[:, :, :, 0]


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCFSpectralHighPass2d(nn.Module):
    def __init__(self, channels: int, modes: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, channels)
        self.channel_mix = nn.Parameter(torch.randn(channels, channels) * scale)
        self.spectrum_weight = nn.Parameter(torch.randn(channels, self.modes, self.modes, 2) * scale)
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SCF spectral block expects [B, C, H, W], got {tuple(x.shape)}")
        residual = x - self.low_pass(x)
        b, _c, height, width = residual.shape
        width_freq = width // 2 + 1
        h_modes = min(self.modes, height)
        w_modes = min(self.modes, width_freq)
        residual_ft = torch.fft.rfft2(residual, norm="ortho")
        out_ft = torch.zeros(
            b,
            self.channels,
            height,
            width_freq,
            dtype=residual_ft.dtype,
            device=residual.device,
        )
        weight = torch.view_as_complex(self.spectrum_weight[:, :h_modes, :w_modes, :].contiguous())
        channel_mix = self.channel_mix.to(dtype=residual_ft.dtype)
        mixed = torch.einsum("bcxy,oc,oxy->boxy", residual_ft[:, :, :h_modes, :w_modes], channel_mix, weight)
        out_ft[:, :, :h_modes, :w_modes] = mixed
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")

    def info(self) -> dict[str, Any]:
        return {
            "name": "scf-spectral-high-pass-2d",
            "channels": self.channels,
            "modes": self.modes,
        }


class LocalPatchHighPassFNO(nn.Module):
    def __init__(self, channels: int, modes: int = 16, patch_size: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.spectral = SCFSpectralHighPass2d(channels, modes=modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, height, width = x.shape
        patch = max(1, min(self.patch_size, height, width))
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = padded.shape[-2:]
        patches = padded.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.contiguous().view(b, c, -1, patch, patch).permute(0, 2, 1, 3, 4)
        patch_count = patches.shape[1]
        patch_out = self.spectral(patches.reshape(b * patch_count, c, patch, patch))
        patch_out = patch_out.reshape(b, patch_count, c, patch, patch).permute(0, 2, 1, 3, 4)
        columns = patch_out.contiguous().view(b, c * patch * patch, patch_count)
        out = F.fold(columns, output_size=(padded_h, padded_w), kernel_size=patch, stride=patch)
        return out[:, :, :height, :width]

    def info(self) -> dict[str, Any]:
        return {
            "name": "local-patch-high-pass-fno",
            "channels": self.channels,
            "patch_size": self.patch_size,
            "spectral": self.spectral.info(),
        }


class HiLocFFNOBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 16, patch_size: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        self.patch_size = int(patch_size)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.global_fno = SCFSpectralHighPass2d(channels, modes=modes)
        self.local_fno = LocalPatchHighPassFNO(channels, modes=modes, patch_size=patch_size)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.norm(x)
        fused = self.fuse(torch.cat([base, self.global_fno(base), self.local_fno(base)], dim=1))
        return x + fused

    def info(self) -> dict[str, Any]:
        return {
            "name": "hiloc-ffno",
            "channels": self.channels,
            "modes": self.modes,
            "patch_size": self.patch_size,
            "global_fno": self.global_fno.info(),
            "local_fno": self.local_fno.info(),
        }


class CAFF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.spatial_query = nn.Conv2d(channels, channels, kernel_size=1)
        self.frequency_key = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = ConvNormAct2d(channels, channels, kernel_size=3)

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        query = self.spatial_query(spatial)
        key = self.frequency_key(frequency)
        query_ft = torch.fft.rfft2(query, norm="ortho")
        key_ft = torch.fft.rfft2(key, norm="ortho")
        gate = torch.sigmoid(key_ft.abs())
        fused = torch.fft.irfft2(query_ft * gate + key_ft, s=query.shape[-2:], norm="ortho")
        return self.out(fused)

    def info(self) -> dict[str, Any]:
        return {"name": "caff-frequency-domain-fusion", "channels": self.channels}


class FeatureDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct2d(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointClassifier(nn.Module):
    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(feature_channels, 256, kernel_size=1),
                nn.Conv1d(256 + feature_channels, 64, kernel_size=1),
                nn.Conv1d(64 + feature_channels, 16, kernel_size=1),
                nn.Conv1d(16 + feature_channels, 1, kernel_size=1),
            ]
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        y = feature
        base = feature
        for index, layer in enumerate(self.layers):
            y = layer(y if index == 0 else torch.cat([y, base], dim=1))
            if index != len(self.layers) - 1:
                y = F.leaky_relu(y, inplace=True)
        return y


@dataclass(frozen=True)
class DuFalConfig:
    feature_channels: int
    blocks: int = 2
    modes: int = 16
    patch_size: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "dufal-projection-feature-extractor",
            "feature_channels": self.feature_channels,
            "blocks": self.blocks,
            "modes": self.modes,
            "patch_size": self.patch_size,
            "spatial_branch": "conv-norm-gelu",
            "frequency_branch": "hiloc-ffno",
            "fusion": "caff",
            "feature_decoder": "conv-norm-gelu",
            "view_fusion": "max",
        }


class DuFalProjectionFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        feature_channels: int = 128,
        blocks: int = 2,
        modes: int = 16,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.config = DuFalConfig(
            feature_channels=int(feature_channels),
            blocks=int(blocks),
            modes=int(modes),
            patch_size=int(patch_size),
        )
        channels = self.config.feature_channels
        self.spatial_stem = nn.Sequential(ConvNormAct2d(in_channels, channels), ConvNormAct2d(channels, channels))
        self.frequency_stem = ConvNormAct2d(in_channels, channels)
        self.frequency_blocks = nn.ModuleList(
            [HiLocFFNOBlock(channels, modes=self.config.modes, patch_size=self.config.patch_size) for _ in range(self.config.blocks)]
        )
        self.fusion_blocks = nn.ModuleList([CAFF(channels) for _ in range(self.config.blocks)])
        self.decoder = FeatureDecoder(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_stem(x)
        frequency = self.frequency_stem(x)
        for frequency_block, fusion in zip(self.frequency_blocks, self.fusion_blocks):
            frequency = frequency_block(frequency)
            spatial = spatial + fusion(spatial, frequency)
        return self.decoder(spatial)

    def info(self) -> dict[str, Any]:
        out = self.config.to_dict()
        out["hiloc_blocks"] = [block.info() for block in self.frequency_blocks]
        out["caff_blocks"] = [block.info() for block in self.fusion_blocks]
        return out


class DuFalStyleDifNet(nn.Module):
    def __init__(self, *, num_views: int, mid_ch: int = 128, blocks: int = 2, modes: int = 16, patch_size: int = 8) -> None:
        super().__init__()
        self.model_version = "dufal-style"
        self.combine = "dufal"
        self.num_views = int(num_views)
        self.mid_ch = int(mid_ch)
        self.projection_encoder = DuFalProjectionFeatureExtractor(
            in_channels=1,
            feature_channels=self.mid_ch,
            blocks=blocks,
            modes=modes,
            patch_size=patch_size,
        )
        self.point_classifier = PointClassifier(self.mid_ch)
        print(f"DuFalStyleDifNet, mid_ch: {mid_ch}, views: {self.num_views}, combine: max")

    def _encode_projections(self, projs: torch.Tensor) -> list[torch.Tensor]:
        batch, num_views, channels, width, height = projs.shape
        flat = projs.reshape(batch * num_views, channels, width, height)
        feat = self.projection_encoder(flat)
        feat = feat.reshape(batch, num_views, feat.shape[1], feat.shape[2], feat.shape[3])
        return [feat]

    def _fuse_view_features(self, proj_feats: list[torch.Tensor], data: dict[str, torch.Tensor]) -> torch.Tensor:
        features = proj_feats[0]
        num_views = features.shape[1]
        p_list = []
        for view_index in range(num_views):
            p_list.append(index_2d(features[:, view_index, ...], data["proj_points"][:, view_index, ...]))
        p_feats = torch.stack(p_list, dim=-1)
        valid_mask = data.get("valid_mask")
        if valid_mask is None:
            return F.max_pool2d(p_feats, (1, num_views)).squeeze(-1)
        mask = valid_mask.permute(0, 2, 1).unsqueeze(1).bool()
        neg = -torch.finfo(p_feats.dtype).max
        pooled = p_feats.masked_fill(~mask, neg).max(dim=-1).values
        any_valid = mask.any(dim=-1)
        return torch.where(any_valid, pooled, torch.zeros_like(pooled))

    def forward_points(self, proj_feats: list[torch.Tensor], data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.point_classifier(self._fuse_view_features(proj_feats, data))

    def forward(self, data: dict[str, torch.Tensor], is_eval: bool = False, eval_npoint: int = 100000):
        proj_feats = self._encode_projections(data["projs"])
        if not is_eval:
            return self.forward_points(proj_feats, data), data["p_gt"]
        total_npoint = data["proj_points"].shape[2]
        pred_list = []
        for left in range(0, total_npoint, eval_npoint):
            right = min(left + eval_npoint, total_npoint)
            chunk = {
                "proj_points": data["proj_points"][..., left:right, :],
                "points": data["points"][..., left:right, :],
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
            "dufal": self.projection_encoder.info(),
        }
