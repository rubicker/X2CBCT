from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.synthrad2025 import assign_split, find_cbct_files, write_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SynthRAD2025 Task 2 chest/abdomen CBCT registry.")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "dataset_list" / "synthrad2025_task2" / "raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data_manifest" / "synthrad2025_task2_chest_abdomen_registry.json")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    records = find_cbct_files(args.raw_root)
    assigned = assign_split(records, seed=args.seed, train_ratio=args.train_ratio)
    registry = write_registry(assigned, args.output, args.raw_root)
    print(json.dumps({"ok": True, "output": str(args.output), "counts": registry["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
