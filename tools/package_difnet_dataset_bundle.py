from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.dataset_bundle import bundle_sources, create_dataset_bundle
from x2cbct_difnet.preprocess_pipeline import DATASET_ID


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the complete fixed DIF-Net dataset family for incremental transfer."
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "transfer_packages")
    parser.add_argument("--plan", action="store_true", help="Print the exact incremental package scope.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.plan:
        print(
            json.dumps(
                {
                    "dataset_id": DATASET_ID,
                    "source_roots": [str(path) for path in bundle_sources(PROJECT_ROOT)],
                    "output_root": str(args.output_root.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    result = create_dataset_bundle(PROJECT_ROOT, args.output_root, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
