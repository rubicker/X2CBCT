from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.preprocess_pipeline import (
    build_pipeline_stages,
    describe_pipeline,
    pipeline_layout,
    run_pipeline,
    validate_pipeline_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the complete fixed self_projection1208 and LUNA16 DIF-Net dataset matrix."
    )
    parser.add_argument("--plan", action="store_true", help="Print the fixed stage plan without generating data.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Back up and rebuild every stage, including outputs that already pass validation.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    layout = pipeline_layout(PROJECT_ROOT)
    stages = build_pipeline_stages(
        PROJECT_ROOT,
        python_executable=sys.executable,
        progress_interval=args.progress_interval,
    )
    plan = describe_pipeline(stages, layout)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    validate_pipeline_inputs(PROJECT_ROOT)
    summary = run_pipeline(stages, layout, force_rebuild=args.force_rebuild)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
