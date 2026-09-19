from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OFFICIAL_GRID_SIZE = 256
OFFICIAL_STRIDE = 4


@dataclass(frozen=True)
class OfficialLatticeBlock:
    offset_zyx: tuple[int, int, int]
    grid_size: int = OFFICIAL_GRID_SIZE
    stride: int = OFFICIAL_STRIDE

    @property
    def lattice_size(self) -> int:
        return self.grid_size // self.stride

    @property
    def point_count(self) -> int:
        return self.lattice_size**3

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_zyx": list(self.offset_zyx),
            "grid_size": self.grid_size,
            "stride": self.stride,
            "lattice_size_zyx": [self.lattice_size, self.lattice_size, self.lattice_size],
            "point_count": self.point_count,
        }


def official_lattice_blocks(grid_size: int = OFFICIAL_GRID_SIZE, stride: int = OFFICIAL_STRIDE) -> list[OfficialLatticeBlock]:
    if grid_size % stride != 0:
        raise ValueError("official lattice grid_size must be divisible by stride")
    return [
        OfficialLatticeBlock(offset_zyx=(z, y, x), grid_size=grid_size, stride=stride)
        for z in range(stride)
        for y in range(stride)
        for x in range(stride)
    ]


def official_lattice_metadata(grid_size: int = OFFICIAL_GRID_SIZE, stride: int = OFFICIAL_STRIDE) -> dict[str, Any]:
    blocks = official_lattice_blocks(grid_size, stride)
    return {
        "name": "official_interleaved_lattice",
        "grid_size": int(grid_size),
        "stride": int(stride),
        "block_count": len(blocks),
        "points_per_block": blocks[0].point_count if blocks else 0,
        "blocks": [block.to_dict() for block in blocks],
    }


def sample_official_lattice_indices(
    num_points: int,
    rng: np.random.Generator,
    *,
    grid_size: int = OFFICIAL_GRID_SIZE,
    stride: int = OFFICIAL_STRIDE,
    block_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    blocks = official_lattice_blocks(grid_size, stride)
    if block_index is None:
        block_index = int(rng.integers(0, len(blocks)))
    block = blocks[int(block_index)]
    lattice_size = block.lattice_size
    replace = int(num_points) > block.point_count
    flat = rng.choice(block.point_count, size=int(num_points), replace=replace)
    z_base, rem = divmod(flat, lattice_size * lattice_size)
    y_base, x_base = divmod(rem, lattice_size)
    offset = np.asarray(block.offset_zyx, dtype=np.int64)
    indices = np.stack([z_base, y_base, x_base], axis=1).astype(np.int64) * stride + offset[None, :]
    trace = {
        "point_sampling": "official_interleaved_lattice",
        "block_index": int(block_index),
        "block_offset_zyx": list(block.offset_zyx),
        "replace": bool(replace),
        "grid_size": int(grid_size),
        "stride": int(stride),
    }
    return indices.astype(np.int64), trace
