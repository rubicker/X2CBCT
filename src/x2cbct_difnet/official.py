from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


def _official_code_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "external" / "dif_net_official" / "code"


def load_official_difnet(project_root: str | Path):
    code_path = _official_code_path(project_root)
    if not code_path.exists():
        raise FileNotFoundError(f"official DIF-Net code not found: {code_path}")
    code_path_str = str(code_path)
    if code_path_str not in sys.path:
        sys.path.insert(0, code_path_str)
    from models.model import DIF_Net  # type: ignore

    return DIF_Net


def synthetic_difnet_item(num_views: int, image_size: int, num_points: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260522)
    projs = torch.rand((1, num_views, 1, image_size, image_size), generator=generator)
    points = torch.rand((1, num_points, 3), generator=generator) * 2.0 - 1.0
    proj_points = torch.rand((1, num_views, num_points, 2), generator=generator) * 2.0 - 1.0
    p_gt = torch.rand((1, 1, num_points), generator=generator)
    view_angles_deg = torch.linspace(0.0, 360.0, steps=num_views + 1)[:-1].reshape(1, num_views)
    valid_mask = torch.ones((1, num_views, num_points), dtype=torch.bool)
    item = {
        "projs": projs,
        "points": points,
        "proj_points": proj_points,
        "p_gt": p_gt,
        "view_angles_deg": view_angles_deg,
        "valid_mask": valid_mask,
    }
    for scale_index, grid_size in enumerate((16, 8, 4)):
        grid_count = int(grid_size) ** 3
        item[f"c2rv_grid_points_s{scale_index}"] = torch.rand((1, grid_count, 3), generator=generator) * 2.0 - 1.0
        item[f"c2rv_grid_proj_points_s{scale_index}"] = (
            torch.rand((1, num_views, grid_count, 2), generator=generator) * 2.0 - 1.0
        )
        item[f"c2rv_grid_valid_mask_s{scale_index}"] = torch.ones((1, num_views, grid_count), dtype=torch.bool)
    return item


def run_official_forward_smoke(
    project_root: str | Path,
    num_views: int = 2,
    image_size: int = 64,
    num_points: int = 128,
    mid_channels: int = 128,
    combine: str = "mlp",
) -> dict[str, Any]:
    DIF_Net = load_official_difnet(project_root)
    model = DIF_Net(num_views=num_views, combine=combine, mid_ch=mid_channels)
    model.eval()
    item = synthetic_difnet_item(num_views, image_size, num_points)
    with torch.no_grad():
        pred, gt = model(item)
        loss = torch.nn.functional.mse_loss(pred, gt).item()
    return {
        "ok": True,
        "num_views": num_views,
        "image_size": image_size,
        "num_points": num_points,
        "mid_channels": mid_channels,
        "combine": combine,
        "output_shape": list(pred.shape),
        "target_shape": list(gt.shape),
        "loss": float(loss),
        "device": str(next(model.parameters()).device),
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out
