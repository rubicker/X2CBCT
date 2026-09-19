from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import write_json
from x2cbct_difnet.normalization_profile import estimate_grouped_hu_profiles, parse_percentiles
from x2cbct_difnet.split_policy import PATIENT_LEVEL_POLICY, assign_train_test_split


LUNA16_C2RV_SPLIT_POLICY = "luna16_c2rv_deepsparse_splits"


def _parse_pair(text: str) -> tuple[int, int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return values[0], values[1]


def _parse_mhd(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split())


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split())


def _is_identity(values: tuple[float, ...], eps: float = 1e-6) -> bool:
    return len(values) == 9 and all(abs(values[index] - (1.0 if index in {0, 4, 8} else 0.0)) <= eps for index in range(9))


def _rounded(values: tuple[float, ...], digits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(item), digits) for item in values)


def _subset_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("subset") and part[6:].isdigit():
            return part
    return "unknown"


def _find_ct_mhds(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.mhd") if "seg-lungs-LUNA16" not in str(path))


def _row_for(path: Path, root: Path) -> dict[str, Any]:
    header = _parse_mhd(path)
    dim_size_xyz = _int_tuple(header["DimSize"])
    spacing_xyz = _float_tuple(header["ElementSpacing"])
    raw_path = path.parent / header.get("ElementDataFile", "")
    mask_path = root / "seg-lungs-LUNA16" / path.name
    transform = _float_tuple(header.get("TransformMatrix", ""))
    return {
        "patient_id": path.stem,
        "seriesuid": path.stem,
        "subset": _subset_name(path),
        "ct_path": str(path.resolve()),
        "relative_ct_path": str(path.resolve().relative_to(root.resolve())),
        "raw_path": str(raw_path.resolve()),
        "relative_raw_path": str(raw_path.resolve().relative_to(root.resolve())) if raw_path.exists() else None,
        "mask_path": str(mask_path.resolve()) if mask_path.exists() else None,
        "relative_mask_path": str(mask_path.resolve().relative_to(root.resolve())) if mask_path.exists() else None,
        "dim_size_xyz": list(dim_size_xyz),
        "shape_zyx": [dim_size_xyz[2], dim_size_xyz[1], dim_size_xyz[0]],
        "spacing_xyz": list(spacing_xyz),
        "element_type": header.get("ElementType"),
        "anatomical_orientation": header.get("AnatomicalOrientation"),
        "transform_is_identity": _is_identity(transform),
        "raw_exists": raw_path.exists(),
        "mask_exists": mask_path.exists(),
    }


def _compatibility_reasons(row: dict[str, Any], expected_inplane: tuple[int, int]) -> list[str]:
    reasons = []
    dim_x, dim_y, _ = [int(item) for item in row["dim_size_xyz"]]
    spacing_x, spacing_y, spacing_z = [float(item) for item in row["spacing_xyz"]]
    if (dim_x, dim_y) != expected_inplane:
        reasons.append("unexpected_inplane_shape")
    if not row["raw_exists"]:
        reasons.append("missing_raw")
    if not row["mask_exists"]:
        reasons.append("missing_lung_mask")
    if row["element_type"] != "MET_SHORT":
        reasons.append("unexpected_element_type")
    if not row["transform_is_identity"]:
        reasons.append("non_identity_transform")
    if abs(spacing_x - spacing_y) > 1e-6:
        reasons.append("non_square_inplane_spacing")
    if spacing_x <= 0 or spacing_y <= 0 or spacing_z <= 0:
        reasons.append("non_positive_spacing")
    return reasons


def _counter(items: list[Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for key, count in Counter(items).most_common(limit):
        rows.append({"key": key, "count": count})
    return rows


def _split_cases(cases: list[dict[str, Any]], train_ratio: float, seed: int, policy: str) -> dict[str, Any]:
    split_result = assign_train_test_split(cases, policy=policy, train_ratio=train_ratio, seed=seed)
    train_ids = {id(row) for row in split_result.train}
    assigned = []
    for row in sorted(cases, key=lambda item: item["patient_id"]):
        split = "train" if id(row) in train_ids else "test"
        assigned.append(
            {
                **row,
                "split": split,
                "split_seed": seed,
                "split_train_ratio": train_ratio,
                "split_policy": split_result.metadata["policy"],
            }
        )
    return {**split_result.metadata, "cases": assigned}


def _split_case_keys(row: dict[str, Any]) -> list[str]:
    subset = str(row.get("subset", "")).strip()
    seriesuid = str(row.get("seriesuid", row.get("patient_id", ""))).strip()
    keys = [seriesuid]
    if subset:
        keys.insert(0, f"{subset}_{seriesuid}")
    return keys


def _split_cases_from_c2rv(cases: list[dict[str, Any]], splits_json: Path) -> dict[str, Any]:
    with splits_json.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    split_by_case: dict[str, str] = {}
    for split in ("train", "eval", "test"):
        for case_id in raw.get(split, []):
            split_by_case[str(case_id)] = split
    assigned = []
    missing = []
    for row in sorted(cases, key=lambda item: item["patient_id"]):
        matched_key = next((key for key in _split_case_keys(row) if key in split_by_case), None)
        if matched_key is None:
            missing.append(_split_case_keys(row)[0])
            continue
        split = split_by_case[matched_key]
        assigned.append(
            {
                **row,
                "split": split,
                "upstream_split": split,
                "split_policy": LUNA16_C2RV_SPLIT_POLICY,
                "split_source": str(splits_json.resolve()),
                "split_case_key": matched_key,
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} LUNA16 cases are missing from C2RV/DeepSparse splits.json: {preview}")
    train_patients = {row["patient_id"] for row in assigned if row["split"] == "train"}
    eval_patients = {row["patient_id"] for row in assigned if row["split"] == "eval"}
    test_patients = {row["patient_id"] for row in assigned if row["split"] == "test"}
    patient_overlap = sorted((train_patients & eval_patients) | (train_patients & test_patients) | (eval_patients & test_patients))
    return {
        "policy": LUNA16_C2RV_SPLIT_POLICY,
        "source": str(splits_json.resolve()),
        "allow_patient_overlap": False,
        "patient_overlap": patient_overlap,
        "case_overlap_count": 0,
        "train_case_count": sum(1 for row in assigned if row["split"] == "train"),
        "eval_case_count": sum(1 for row in assigned if row["split"] == "eval"),
        "test_case_count": sum(1 for row in assigned if row["split"] == "test"),
        "cases": assigned,
    }


def build_luna16_profile(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = args.root.resolve()
    rows = [_row_for(path, root) for path in _find_ct_mhds(root)]
    expected_inplane = args.expected_inplane
    compatible = []
    excluded = []
    for row in rows:
        reasons = _compatibility_reasons(row, expected_inplane)
        out = {**row, "compatibility_reasons": reasons}
        if reasons:
            excluded.append(out)
        else:
            compatible.append(out)

    exact_counter = Counter(
        (
            tuple(row["dim_size_xyz"]),
            _rounded(tuple(float(item) for item in row["spacing_xyz"])),
        )
        for row in compatible
    )
    dominant_exact = exact_counter.most_common(1)[0][0] if exact_counter else None
    if args.geometry_policy == "strict-exact" and dominant_exact is not None:
        included = [
            row for row in compatible
            if (tuple(row["dim_size_xyz"]), _rounded(tuple(float(item) for item in row["spacing_xyz"]))) == dominant_exact
        ]
        strict_excluded = [
            {**row, "compatibility_reasons": ["not_dominant_exact_shape_spacing"]}
            for row in compatible
            if row not in included
        ]
        excluded = excluded + strict_excluded
    else:
        included = compatible

    if args.split_policy == "c2rv":
        split = _split_cases_from_c2rv(included, args.splits_json)
    else:
        split = _split_cases(included, args.train_ratio, args.seed, PATIENT_LEVEL_POLICY)
    profiles = {}
    profile_case_stats = []
    if not args.metadata_only and included:
        profiles, profile_case_stats = estimate_grouped_hu_profiles(
            split["cases"],
            image_path=lambda item: Path(item["ct_path"]),
            mask_path=None,
            group_key=lambda _item: "luna16",
            source_split=args.hu_window_split,
            target_size=None,
            percentiles=parse_percentiles(args.hu_percentiles),
            mask_policy="finite",
            max_voxels_per_case=args.hu_sample_voxels_per_case,
        )

    shape_groups = _counter([tuple(row["dim_size_xyz"]) for row in rows], args.top)
    spacing_groups = _counter([_rounded(tuple(float(item) for item in row["spacing_xyz"])) for row in rows], args.top)
    exact_groups = _counter(
        [(tuple(row["dim_size_xyz"]), _rounded(tuple(float(item) for item in row["spacing_xyz"]))) for row in rows],
        args.top,
    )
    exclusion_counts = Counter(reason for row in excluded for reason in row["compatibility_reasons"])
    report = {
        "schema": "x2cbct_luna16_dataset_profile.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "geometry_policy": args.geometry_policy,
        "expected_inplane_xy": list(expected_inplane),
        "total_ct_count": len(rows),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "unique_patient_count": len({row["patient_id"] for row in rows}),
        "included_patient_count": len({row["patient_id"] for row in included}),
        "shape_group_count": len(Counter(tuple(row["dim_size_xyz"]) for row in rows)),
        "spacing_group_count": len(Counter(tuple(row["spacing_xyz"]) for row in rows)),
        "exact_shape_spacing_group_count": len(exact_counter),
        "dominant_exact_shape_spacing": {
            "dim_size_xyz": list(dominant_exact[0]) if dominant_exact else None,
            "spacing_xyz": list(dominant_exact[1]) if dominant_exact else None,
            "count": exact_counter[dominant_exact] if dominant_exact else 0,
        },
        "shape_groups_top": shape_groups,
        "spacing_groups_top": spacing_groups,
        "exact_shape_spacing_groups_top": exact_groups,
        "exclusion_summary": dict(sorted(exclusion_counts.items())),
        "hu_window": {
            "metadata_only": args.metadata_only,
            "source_split": args.hu_window_split,
            "percentiles": list(parse_percentiles(args.hu_percentiles)),
            "sample_voxels_per_case": args.hu_sample_voxels_per_case,
            "profiles": profiles,
            "case_stats": profile_case_stats,
        },
        "split": {key: value for key, value in split.items() if key != "cases"},
        "included_cases": included,
        "excluded_cases": excluded,
    }
    registry = {
        "schema": "x2cbct_luna16_patient_split_registry.v1",
        "generated_at": report["generated_at"],
        "source_profile": str(args.output.resolve()),
        "root": str(root),
        "geometry_policy": args.geometry_policy,
        "hu_window": report["hu_window"],
        "split": {key: value for key, value in split.items() if key != "cases"},
        "cases": split["cases"],
    }
    return report, registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan LUNA16 CT metadata, HU window, and train/test split.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset_list" / "LUNA16")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "audits" / "luna16_dataset_profile" / "luna16_profile_c2rv.json")
    parser.add_argument("--registry-output", type=Path, default=PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json")
    parser.add_argument("--geometry-policy", choices=("compatible", "strict-exact"), default="compatible")
    parser.add_argument("--expected-inplane", type=_parse_pair, default=(512, 512))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--split-policy", choices=("c2rv", "patient"), default="c2rv")
    parser.add_argument("--splits-json", type=Path, default=PROJECT_ROOT / "external" / "c2rv_cbct_official" / "data" / "LUNA16" / "splits.json")
    parser.add_argument("--hu-percentiles", default="0.5,99.5")
    parser.add_argument("--hu-window-split", choices=("train", "all"), default="train")
    parser.add_argument("--hu-sample-voxels-per-case", type=int, default=20_000)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    report, registry = build_luna16_profile(args)
    write_json(report, args.output)
    write_json(registry, args.registry_output)
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(args.output.resolve()),
                "registry": str(args.registry_output.resolve()),
                "included_count": report["included_count"],
                "excluded_count": report["excluded_count"],
                "patient_overlap": report["split"]["patient_overlap"],
                "hu_profile": report["hu_window"]["profiles"].get("luna16"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
