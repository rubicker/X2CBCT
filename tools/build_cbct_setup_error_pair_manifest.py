from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_error_manifest import build_pair_manifest, write_pair_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build same-patient cross-case CBCT setup-error pair manifest.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset_list" / "self_projection1208")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "pair_manifest")
    parser.add_argument("--no-headers", action="store_true", help="Skip SimpleITK header reads for a faster path-only scan.")
    args = parser.parse_args()

    manifest = build_pair_manifest(args.dataset_root, read_headers=not args.no_headers)
    outputs = write_pair_manifest(manifest, args.output_dir)
    print(json.dumps({"ok": True, **outputs, "pair_count": manifest["pair_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
