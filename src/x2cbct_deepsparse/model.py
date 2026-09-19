from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def _official_code_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve() / "external" / "deepsparse_official" / "code"
    if not (root / "models" / "model_v7.py").exists():
        raise FileNotFoundError(f"official DeepSparse code is missing: {root}")
    return root


def masked_query_view_feats(
    view_feats: torch.Tensor,
    points_proj: torch.Tensor,
    fusion: str | None = "max",
    view_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepSparse view query with support for per-point validity masks.

    Official DeepSparse accepts a [B,M] padded-view mask.  Clinical projections
    additionally need [B,M,N] validity because a 3-D point can fall outside the
    detector in only some views.  All-invalid points return zeros instead of
    propagating ``-inf`` into the decoder.
    """

    if view_feats.ndim != 5:
        raise ValueError(f"view_feats must be [B,M,C,H,W], got {tuple(view_feats.shape)}")
    if points_proj.ndim != 4 or points_proj.shape[-1] != 2:
        raise ValueError(f"points_proj must be [B,M,N,2], got {tuple(points_proj.shape)}")
    if view_feats.shape[:2] != points_proj.shape[:2]:
        raise ValueError("view and projection batch/view dimensions do not match")

    sampled = []
    for view_index in range(view_feats.shape[1]):
        feat = view_feats[:, view_index].transpose(2, 3)
        grid = points_proj[:, view_index].unsqueeze(2)
        values = F.grid_sample(feat, grid, align_corners=True, padding_mode="zeros")
        sampled.append(values[:, :, :, 0])
    stacked = torch.stack(sampled, dim=-1)  # [B,C,N,M]

    if fusion is None:
        return stacked
    if fusion != "max":
        raise NotImplementedError(f"unsupported DeepSparse view fusion: {fusion}")
    if view_mask is None:
        return stacked.max(dim=-1).values

    mask = view_mask.bool()
    if mask.ndim == 2:
        if tuple(mask.shape) != tuple(view_feats.shape[:2]):
            raise ValueError(f"view_mask must be [B,M], got {tuple(mask.shape)}")
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        expected = (view_feats.shape[0], view_feats.shape[1], points_proj.shape[2])
        if tuple(mask.shape) != expected:
            raise ValueError(f"point view mask must be [B,M,N]={expected}, got {tuple(mask.shape)}")
        mask = mask.permute(0, 2, 1).unsqueeze(1)
    else:
        raise ValueError(f"view_mask must be [B,M] or [B,M,N], got {tuple(mask.shape)}")

    masked = stacked.masked_fill(~mask, float("-inf"))
    pooled = masked.max(dim=-1).values
    any_valid = mask.any(dim=-1)
    return torch.where(any_valid, pooled, torch.zeros_like(pooled))


def _load_official_modules(project_root: str | Path) -> tuple[Any, Any, Any]:
    code_root = _official_code_root(project_root)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from models import encoder as encoder_module  # type: ignore[import-not-found]
    from models import model_v7 as model_module  # type: ignore[import-not-found]
    from utils import load_config  # type: ignore[import-not-found]

    # EncoderV_mv imported the official function by name, so replace that
    # module binding without editing the vendored repository.
    encoder_module.query_view_feats = masked_query_view_feats
    return model_module, encoder_module, load_config


def _model_class(project_root: str | Path) -> type[torch.nn.Module]:
    model_module, _, _ = _load_official_modules(project_root)

    class _X2CBCTDeepSparse(model_module.Model_mv):
        def encode(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            low_res_mask = data.get("points_lr_valid_mask", data.get("view_mask"))
            feats_3d, feats_2d = self.encoder(
                data,
                require_2d=True,
                view_masks=[low_res_mask],
            )
            loss_vq_all: torch.Tensor | float = 0.0
            feats_out = None
            for layer_index, feats in enumerate(feats_3d):
                feats, loss_vq = self.cb[layer_index](feats)
                loss_vq_all = loss_vq_all + loss_vq
                if layer_index > 0:
                    feats = torch.cat([feats, feats_out], dim=1)
                feats_out = self.decoders[layer_index](feats)
            if feats_out is None:
                raise RuntimeError("DeepSparse encoder returned no feature levels")
            return {
                "feats_2d": feats_2d,
                "feats_3d": feats_out,
                "loss_vq": loss_vq_all / len(feats_3d),
            }

        def forward_points(
            self,
            feats_dict: dict[str, Any],
            data: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            p_feats = model_module.index_3d(feats_dict["feats_3d"], data["points_ct"])
            point_mask = data.get("points_valid_mask", data.get("view_mask"))
            for feats_2d in feats_dict["feats_2d"]:
                p_feats_2d = masked_query_view_feats(
                    view_feats=feats_2d,
                    points_proj=data["points_proj"],
                    fusion="max",
                    view_mask=point_mask,
                )
                p_feats = torch.cat([p_feats_2d, p_feats], dim=1)
            return self.point_decoder(p_feats)

        def forward(
            self,
            data: dict[str, torch.Tensor],
            is_eval: bool = False,
            eval_npoint: int = 100_000,
        ) -> dict[str, torch.Tensor]:
            feats_dict = self.encode(data)
            if not is_eval:
                return {
                    "points_pred": self.forward_points(feats_dict, data),
                    "loss_vq": feats_dict["loss_vq"],
                }

            total = int(data["points_ct"].shape[1])
            predictions = []
            for left in range(0, total, int(eval_npoint)):
                right = min(left + int(eval_npoint), total)
                chunk = dict(data)
                chunk["points_ct"] = data["points_ct"][:, left:right, :]
                chunk["points_proj"] = data["points_proj"][:, :, left:right, :]
                if "points_valid_mask" in data:
                    chunk["points_valid_mask"] = data["points_valid_mask"][:, :, left:right]
                predictions.append(self.forward_points(feats_dict, chunk))
            return {"points_pred": torch.cat(predictions, dim=2)}

    _X2CBCTDeepSparse.__name__ = "X2CBCTDeepSparse"
    return _X2CBCTDeepSparse


def build_deepsparse_model(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    root = Path(project_root).resolve()
    _, _, load_config = _load_official_modules(root)
    cfg_path = (
        Path(config_path).resolve()
        if config_path is not None
        else root / "external" / "deepsparse_official" / "configs" / "finetune_s1.yaml"
    )
    cfg = load_config(str(cfg_path))
    model_cls = _model_class(root)
    model = model_cls(cfg.model)
    if checkpoint_path is not None:
        checkpoint = torch.load(str(Path(checkpoint_path).resolve()), map_location="cpu")
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
    return model
