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
from x2cbct_difnet.setup_error_roi import generate_roi_candidate, write_roi_candidates


def unique_cases(manifest: dict) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    rows = []
    for case in manifest["cases"]:
        key = (str(case["patient_id"]), str(case["case_id"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(case)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cervical spine ROI candidates for setup-error registration.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset_list" / "self_projection1208")
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "cervical_roi_review")
    parser.add_argument("--threshold-low", type=float, default=700.0)
    parser.add_argument("--threshold-high", type=float, default=1700.0)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pair_manifest:
        manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    else:
        manifest = build_pair_manifest(args.dataset_root, read_headers=True)
        write_pair_manifest(manifest, args.output_dir)

    cases = unique_cases(manifest)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    candidates = [
        generate_roi_candidate(
            str(case["patient_id"]),
            str(case["case_id"]),
            Path(case["cbct_path"]),
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
        )
        for case in cases
    ]
    csv_path = args.output_dir / "roi_candidates.csv"
    json_path = args.output_dir / "roi_candidates.json"
    write_roi_candidates(candidates, csv_path, json_path)
    print(
        json.dumps(
            {
                "ok": True,
                "roi_candidates_csv": str(csv_path),
                "roi_candidates_json": str(json_path),
                "case_count": len(candidates),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
