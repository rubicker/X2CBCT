from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json
from x2cbct_difnet.synthrad2025 import build_synthrad_view_caches


def _parse_views(value: str) -> list[str]:
    views = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if token in {"2", "5", "10"}:
            token = f"view_{token}"
        views.append(token)
    return views


def _infer_target_size(drr_manifest: dict) -> int:
    for row in drr_manifest.get("cases", []):
        row_config = row.get("drr_config") or {}
        if row_config.get("voxel_size"):
            return int(row_config["voxel_size"])
        shape = row.get("processed_shape_zyx") or row.get("target_shape_zyx")
        if shape and len(shape) == 3 and len(set(int(item) for item in shape)) == 1:
            return int(shape[0])
    config = drr_manifest.get("drr_config") or {}
    if config.get("voxel_size"):
        return int(config["voxel_size"])
    raise ValueError("cannot infer target size from DRR manifest; pass --target-size")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SynthRAD2025 chest/abdomen 2/5/10-view cache manifests from 50-view DRR data.")
    parser.add_argument("--drr-manifest", type=Path, default=PROJECT_ROOT / "data_cache" / "synthrad2025_task2" / "drr_128" / "drr_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data_cache" / "synthrad2025_task2" / "view_cache_128")
    parser.add_argument("--views", default="view_2,view_5,view_8,view_10")
    parser.add_argument("--target-size", type=int)
    parser.add_argument("--asset-mode", choices=("reference", "copy"), default="reference")
    args = parser.parse_args()

    drr_manifest = read_json(args.drr_manifest)
    target_size = int(args.target_size) if args.target_size is not None else _infer_target_size(drr_manifest)
    views = _parse_views(args.views)
    outputs = build_synthrad_view_caches(
        drr_manifest,
        args.output_root,
        views=views,
        target_size=target_size,
        source_name="synthrad2025_task2",
        group_key="anatomy",
        default_group="synthrad2025_task2",
        split_policy="8:2 train/test; test also acts as validation",
        asset_mode=args.asset_mode,
    )
    print(json.dumps({"ok": True, "outputs": [str(path) for path in outputs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
