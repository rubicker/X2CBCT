from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.derivative_manifests import build_derivative_manifest_set


def _parse_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fixed DIF-Net V1 derivative manifests before any real preprocessing work."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data_manifest" / "difnet_v1_derivative_manifests",
    )
    parser.add_argument(
        "--fixed-view-manifest",
        type=Path,
        default=PROJECT_ROOT / "data_manifest" / "fixed_view_sampling_v0b.json",
    )
    parser.add_argument(
        "--patient-split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data_manifest" / "splits_v0b_train_test.json",
    )
    parser.add_argument(
        "--luna16-registry",
        type=Path,
        default=PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json",
    )
    parser.add_argument("--mixed-train-ratio", type=float, default=0.8)
    parser.add_argument("--mixed-seed", type=int, default=20260517)
    parser.add_argument("--translation-mm", type=_parse_ints, default=(-10, 0, 10))
    parser.add_argument("--rotation-deg", type=_parse_ints, default=(-10, 0, 10))
    parser.add_argument(
        "--skip-local-case-path-check",
        action="store_true",
        help="Skip checking that fixed-view self_projection1208 case files still exist on disk.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index = build_derivative_manifest_set(
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        fixed_view_manifest_path=args.fixed_view_manifest,
        patient_split_manifest_path=args.patient_split_manifest,
        luna16_registry_path=args.luna16_registry,
        mixed_train_ratio=args.mixed_train_ratio,
        mixed_seed=args.mixed_seed,
        translation_mm=args.translation_mm,
        rotation_deg=args.rotation_deg,
        command=" ".join(sys.argv),
        overwrite=args.overwrite,
        validate_case_paths=not args.skip_local_case_path_check,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "index_path": index["index_path"],
                "derivative_manifest_count": len(index["derivative_manifests"]),
                "training_base_manifests": index["training_base_manifests"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
