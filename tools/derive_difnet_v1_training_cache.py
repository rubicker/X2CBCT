from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.v1_training_cache import derive_v1_training_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive a DIF-Net V1 training cache from a training-base cache and a fixed derivative manifest."
    )
    parser.add_argument("--base-cache-root", type=Path, required=True)
    parser.add_argument("--derivative-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--views", default="2,5,8,10")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = derive_v1_training_cache(
        base_cache_root=args.base_cache_root,
        derivative_manifest_path=args.derivative_manifest,
        output_root=args.output_root,
        views=args.views,
        overwrite=args.overwrite,
        command=" ".join(sys.argv),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(args.output_root.resolve()),
                "view_sample_counts": manifest.get("view_sample_counts"),
                "counts": manifest.get("counts"),
                "writes_body_or_projection_assets": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
