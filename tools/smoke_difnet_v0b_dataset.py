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

from x2cbct_difnet.dataset import DifNetV0bDataset


def _array_summary(array: np.ndarray) -> dict[str, object]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load one V0b local DIF-Net dataset item.")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--view-key", choices=("view_2", "view_5", "view_10"), default="view_2")
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--projection-size", type=int, default=64)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = DifNetV0bDataset(
        PROJECT_ROOT,
        split=args.split,
        view_key=args.view_key,
        num_points=args.num_points,
        projection_size=args.projection_size,
    )
    item = dataset[args.index]
    valid_mask = np.asarray(item["valid_mask"])
    report = {
        "ok": True,
        "split": args.split,
        "view_key": args.view_key,
        "dataset_length": len(dataset),
        "index": args.index,
        "name": item["name"],
        "projs": _array_summary(item["projs"]),
        "points": _array_summary(item["points"]),
        "proj_points": _array_summary(item["proj_points"]),
        "p_gt": _array_summary(item["p_gt"]),
        "valid_ratio": float(valid_mask.mean()),
        "trace": item["trace"],
    }
    output = args.output or PROJECT_ROOT / "runs" / "difnet_v0" / f"v0b_dataset_smoke_{args.split}_{args.view_key}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"ok": True, "report": str(output), "valid_ratio": report["valid_ratio"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
