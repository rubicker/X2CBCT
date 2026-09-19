from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.synthrad2025 import SYNTHRAD2025_TASK2_DOWNLOADS, download_with_resume, extract_zip


def _default_download_dir() -> Path:
    return PROJECT_ROOT / "dataset_list" / "synthrad2025_task2" / "downloads"


def _default_extract_dir() -> Path:
    return PROJECT_ROOT / "dataset_list" / "synthrad2025_task2" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SynthRAD2025 Task 2 training and validation input packages.")
    parser.add_argument("--download-dir", type=Path, default=_default_download_dir())
    parser.add_argument("--extract-dir", type=Path, default=_default_extract_dir())
    parser.add_argument("--files", default="task2-train,task2-val-input")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    selected = [part.strip() for part in args.files.split(",") if part.strip()]
    results = []
    for key in selected:
        if key not in SYNTHRAD2025_TASK2_DOWNLOADS:
            raise ValueError(f"unknown file key: {key}")
        spec = SYNTHRAD2025_TASK2_DOWNLOADS[key]
        result = download_with_resume(spec, args.download_dir, proxy=None if args.no_proxy else args.proxy)
        if args.extract and result.get("complete"):
            result["extract"] = extract_zip(Path(result["path"]), args.extract_dir)
        results.append({"key": key, **result})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    print(json.dumps({"ok": all(item.get("ok") for item in results), "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
