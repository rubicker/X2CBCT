from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DIF-Net V0b cached geometry projection points.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=2)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = CachedDifNetV0bDataset(
        args.cache_root,
        split=args.split,
        view_key=view_key_from_count(args.view_count),
        num_points=args.num_points,
        max_cases=args.max_cases,
        deterministic=True,
    )
    cases = []
    for index in range(len(dataset)):
        item = dataset[index]
        mask = np.asarray(item["valid_mask"])
        proj_points = np.asarray(item["proj_points"])
        cases.append(
            {
                "name": item["name"],
                "trace": item["trace"],
                "proj_points_shape": list(proj_points.shape),
                "proj_points_finite": bool(np.isfinite(proj_points).all()),
                "valid_mask_shape": list(mask.shape),
                "valid_mask_dtype": str(mask.dtype),
                "valid_ratio": float(mask.mean()),
            }
        )
    report = {
        "ok": True,
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "view_count": args.view_count,
        "case_count": len(cases),
        "cases": cases,
    }
    output = args.output or args.cache_root / "reports" / f"geometry_audit_{args.split}_view{args.view_count}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"ok": True, "report": str(output), "case_count": len(cases)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
