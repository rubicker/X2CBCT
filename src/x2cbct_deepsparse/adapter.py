from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


class DeepSparseCachedDataset(Dataset):
    """Expose an existing X2CBCT cache using DeepSparse's tensor contract.

    The wrapped cache remains the source of truth for case splits, fixed views,
    projection normalization, CT normalization, and clinical geometry.  A
    single 32^3 back-projection grid is requested through the existing C2RV
    grid path because it has exactly the coordinates needed by DeepSparse's
    low-resolution 3-D feature volume.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        view_count: int,
        num_points: int = 10_000,
        lr_res: int = 32,
        max_cases: int | None = None,
        seed: int = 20260721,
        deterministic: bool = False,
        eval_grid_size: int | None = None,
        projection_multiplier: float = 1.0,
    ) -> None:
        if int(view_count) not in SUPPORTED_VIEW_COUNTS:
            choices = "/".join(str(item) for item in SUPPORTED_VIEW_COUNTS)
            raise ValueError(f"DeepSparse deployment only supports project views {choices}, got {view_count}")
        if int(lr_res) < 2:
            raise ValueError(f"lr_res must be >= 2, got {lr_res}")
        if not np.isfinite(projection_multiplier) or float(projection_multiplier) <= 0.0:
            raise ValueError("projection_multiplier must be a finite positive number")

        self.view_count = int(view_count)
        self.view_key = view_key_from_count(self.view_count)
        self.lr_res = int(lr_res)
        self.projection_multiplier = float(projection_multiplier)
        self.base = CachedDifNetV0bDataset(
            cache_root,
            split=split,
            view_key=self.view_key,
            num_points=int(num_points),
            max_cases=max_cases,
            seed=int(seed),
            deterministic=bool(deterministic),
            eval_grid_size=eval_grid_size,
            c2rv_grids=True,
            c2rv_grid_scales=(self.lr_res,),
        )
        self.cache_root = self.base.cache_root
        self.cache_manifest = self.base.cache_manifest
        self.samples = self.base.samples
        self.split = split
        self.eval_grid_size = eval_grid_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        projs = np.asarray(item["projs"], dtype=np.float32)
        if self.projection_multiplier != 1.0:
            projs = projs * self.projection_multiplier

        if projs.ndim != 4 or projs.shape[0] != self.view_count or projs.shape[1] != 1:
            raise ValueError(
                f"expected projections [M,1,H,W] with M={self.view_count}, got {tuple(projs.shape)}"
            )

        points_proj = np.asarray(item["proj_points"], dtype=np.float32)
        points_valid = np.asarray(item["valid_mask"], dtype=bool)
        points_lr_proj = np.asarray(item["c2rv_grid_proj_points_s0"], dtype=np.float32)
        points_lr_valid = np.asarray(item["c2rv_grid_valid_mask_s0"], dtype=bool)
        expected_lr_points = self.lr_res**3
        if points_lr_proj.shape != (self.view_count, expected_lr_points, 2):
            raise ValueError(
                "unexpected low-resolution projection grid shape: "
                f"{tuple(points_lr_proj.shape)} != {(self.view_count, expected_lr_points, 2)}"
            )

        out: dict[str, Any] = {
            "dst_name": str(self.cache_manifest.get("dataset_group", self.cache_root.name)),
            "name": item["name"],
            "projs": projs,
            "points_ct": np.asarray(item["points"], dtype=np.float32),
            "points_gt": np.asarray(item["p_gt"], dtype=np.float32),
            "points_proj": points_proj,
            "points_lr_proj": points_lr_proj,
            # Exact-M experiments contain no padded views.  Per-point masks
            # separately exclude detector-outside samples during max pooling.
            "view_mask": np.ones((self.view_count,), dtype=bool),
            "points_valid_mask": points_valid,
            "points_lr_valid_mask": points_lr_valid,
            "angles": np.asarray(item.get("view_angles_deg", []), dtype=np.float32),
            "trace": item["trace"],
        }
        if "image" in item:
            out["image"] = np.asarray(item["image"], dtype=np.float32)
        return out
