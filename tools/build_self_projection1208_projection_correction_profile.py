from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import (
    build_self_projection1208_projection_correction_profile,
    default_projection_correction_profile_path,
)
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the global self_projection1208 per-patient projection brightness correction profile."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_projection_correction_profile_path(PROJECT_ROOT),
    )
    parser.add_argument("--projection-size", type=int, default=256)
    parser.add_argument("--projection-correction-view", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--allow-missing-cases",
        action="store_true",
        help="Allow profile generation to skip fixed-view cases with missing required projection files.",
    )
    args = parser.parse_args()

    profile = build_self_projection1208_projection_correction_profile(
        project_root=PROJECT_ROOT,
        output_path=args.output_path,
        projection_size=args.projection_size,
        view_key=view_key_from_count(args.projection_correction_view),
        progress_interval=args.progress_interval,
        allow_missing_required_case_files=args.allow_missing_cases,
        command=" ".join(sys.argv),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "profile_path": str(args.output_path.resolve()),
                "source_case_count": profile["source_case_count"],
                "dark_case_count": profile["dark_case_count_in_profile"],
                "patient_count": profile["patient_count_in_profile"],
                "projection_size": profile["source_projection_size"],
                "view_key": profile["source_view_key"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
