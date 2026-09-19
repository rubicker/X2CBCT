from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.translation_probe import build_translation_probe


def _parse_shifts(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one shift is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V1 self_projection1208 +8/+12mm translation pre-experiment.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset_list" / "self_projection1208")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "v1_translation_range_probe")
    parser.add_argument("--shifts-mm", type=_parse_shifts, default=(8.0, 12.0))
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--target-angle-deg", type=float, default=290.0)
    args = parser.parse_args()

    report = build_translation_probe(
        args.dataset_root,
        args.output_dir,
        shifts_mm=args.shifts_mm,
        max_cases=args.max_cases,
        target_angle_deg=args.target_angle_deg,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "case_count": report["case_count"],
                "report": str(args.output_dir / "translation_range_probe_report.json"),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
