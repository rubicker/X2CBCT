from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json, relative_to_cache, resolve_cache_path, write_json
from x2cbct_difnet.normalization import view_key_from_count
from x2cbct_difnet.setup_augmentation import setup_augmentation_config
from x2cbct_difnet.split_policy import CASE_MIXED_POLICY, PATIENT_LEVEL_POLICY, normalize_split_policy


BASE_AUGMENTATION_ID = "si_0_rot_0"
SPLITS = ("train", "test")


def _views(raw: str, source_manifest: dict[str, Any]) -> tuple[str, ...]:
    if raw == "all":
        views = source_manifest.get("views", {})
        return tuple(views.keys()) if isinstance(views, dict) else tuple(str(item) for item in views)
    out = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        out.append(view_key_from_count(int(token)) if token in {"2", "5", "10"} else token)
    return tuple(dict.fromkeys(out))


def _source_case_key(sample: dict[str, Any]) -> str:
    setup = sample.get("setup_augmentation") or {}
    if setup.get("source_case_key"):
        return str(setup["source_case_key"])
    return str(sample.get("case_key") or f"patient{sample['patient_id']}_case{sample['case_id']}")


def _case_sort_key(case_key: str) -> tuple[int, int, str]:
    patient_part, _, case_part = case_key.partition("_case")
    try:
        patient = int(patient_part.replace("patient", ""))
        case = int(case_part)
        return patient, case, case_key
    except ValueError:
        return 10**9, 10**9, case_key


def _split_from_source_manifest(samples: list[dict[str, Any]]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for sample in samples:
        source = _source_case_key(sample)
        split = str(sample["split"])
        previous = assignment.setdefault(source, split)
        if previous != split:
            raise ValueError(f"source case {source} appears in both {previous!r} and {split!r}")
    return assignment


def _split_from_patient_manifest(split_manifest_path: Path) -> dict[str, str]:
    payload = read_json(split_manifest_path)
    assignment: dict[str, str] = {}
    for split in SPLITS:
        for row in payload.get(split, {}).get("cases", []):
            key = f"patient{row['patient_id']}_case{row['case_id']}"
            assignment[key] = split
    if not assignment:
        raise ValueError(f"patient split manifest has no train/test cases: {split_manifest_path}")
    return assignment


def _absolute_cache_path(cache_root: Path, value: str) -> str:
    return str(resolve_cache_path(cache_root, value).resolve())


def _manifest_sample(
    source_sample: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    output_sample_path: Path,
    split: str,
    sample_id: str,
    case_key: str,
    augmentation_selection: str,
) -> dict[str, Any]:
    sample = dict(source_sample)
    sample["sample_id"] = sample_id
    sample["split"] = split
    sample["case_key"] = case_key
    sample["sample_path"] = relative_to_cache(output_sample_path, output_root)
    sample["case_manifest_path"] = _absolute_cache_path(source_root, str(source_sample["case_manifest_path"]))
    sample["cbct_payload_path"] = _absolute_cache_path(source_root, str(source_sample["cbct_payload_path"]))
    sample["projection_payload_paths"] = [
        _absolute_cache_path(source_root, str(item)) for item in source_sample["projection_payload_paths"]
    ]
    selected = []
    for item in source_sample.get("selected_projections", []):
        copied = dict(item)
        for key in ("payload_path", "metadata_path"):
            if copied.get(key):
                copied[key] = _absolute_cache_path(source_root, str(copied[key]))
        selected.append(copied)
    sample["selected_projections"] = selected
    if augmentation_selection == "base":
        sample.pop("setup_augmentation", None)
    return sample


def _write_sample_npz(
    source_npz_path: Path,
    output_npz_path: Path,
    *,
    source_root: Path,
    split: str,
    sample_id: str,
    case_key: str,
    augmentation_selection: str,
) -> None:
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_npz_path, allow_pickle=False) as raw:
        payload = {key: raw[key] for key in raw.files}
        payload["sample_id"] = np.array(sample_id)
        payload["split"] = np.array(split)
        payload["case_key"] = np.array(case_key)
        payload["case_manifest_path"] = np.array(_absolute_cache_path(source_root, str(raw["case_manifest_path"].item())))
        payload["cbct_payload_path"] = np.array(_absolute_cache_path(source_root, str(raw["cbct_payload_path"].item())))
        payload["projection_payload_paths"] = np.array(
            [_absolute_cache_path(source_root, str(item)) for item in raw["projection_payload_paths"].tolist()]
        )
        if augmentation_selection == "base":
            for key in (
                "augmentation_id",
                "translation_mm",
                "rotation_deg",
                "source_patient_id",
                "source_case_id",
                "source_case_key",
            ):
                payload.pop(key, None)
        np.savez_compressed(output_npz_path, **payload)


def _split_summary(samples: list[dict[str, Any]], *, split_policy: str, split_seed: int, train_ratio: float) -> dict[str, Any]:
    by_split = {split: [item for item in samples if item["split"] == split] for split in SPLITS}
    patients = {
        split: sorted({str(item["patient_id"]) for item in rows})
        for split, rows in by_split.items()
    }
    cases = {
        split: sorted({_source_case_key(item) for item in rows})
        for split, rows in by_split.items()
    }
    patient_overlap = sorted(set(patients["train"]) & set(patients["test"]))
    case_overlap = sorted(set(cases["train"]) & set(cases["test"]))
    return {
        "policy": split_policy,
        "split_unit": "case" if split_policy == CASE_MIXED_POLICY else "patient_id",
        "seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "train_patient_count": len(patients["train"]),
        "test_patient_count": len(patients["test"]),
        "train_case_count": len(cases["train"]),
        "test_case_count": len(cases["test"]),
        "patient_overlap": patient_overlap,
        "case_overlap_count": len(case_overlap),
        "case_overlap": case_overlap,
        "allow_patient_overlap": split_policy == CASE_MIXED_POLICY,
        "train_cases": len(cases["train"]),
        "test_cases": len(cases["test"]),
        "indexed_train_cases": len(cases["train"]),
        "indexed_test_cases": len(cases["test"]),
        "has_validation_split": False,
        "test_split_role": "held-out test; no validation split",
    }


def _view_sample_counts(samples: list[dict[str, Any]], views: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        view: {
            split: sum(1 for sample in samples if sample["view_key"] == view and sample["split"] == split)
            for split in SPLITS
        }
        for view in views
    }


def derive_real_difnet_cache(
    *,
    source_cache_root: Path,
    output_root: Path,
    split_policy: str,
    augmentation_selection: str,
    views: str = "all",
    patient_split_manifest: Path = PROJECT_ROOT / "data_manifest" / "splits_v0b_train_test.json",
    overwrite: bool = False,
    command: str = "",
) -> dict[str, Any]:
    source_root = source_cache_root.resolve()
    out_root = output_root.resolve()
    source_manifest = read_json(source_root / "cache_manifest.json")
    normalized_policy = normalize_split_policy(split_policy)
    if augmentation_selection not in {"base", "all"}:
        raise ValueError("augmentation_selection must be 'base' or 'all'")
    selected_views = _views(views, source_manifest)

    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root already exists: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_samples = [
        sample
        for sample in source_manifest["samples"]
        if sample["view_key"] in selected_views
        and (
            augmentation_selection == "all"
            or (sample.get("setup_augmentation") or {}).get("augmentation_id") == BASE_AUGMENTATION_ID
        )
    ]
    if not source_samples:
        raise ValueError("no source samples matched the requested views and augmentation selection")

    split_assignment = (
        _split_from_source_manifest(source_manifest["samples"])
        if normalized_policy == CASE_MIXED_POLICY
        else _split_from_patient_manifest(patient_split_manifest.resolve())
    )

    derived_samples: list[dict[str, Any]] = []
    skipped_sources: set[str] = set()
    for source_sample in sorted(
        source_samples,
        key=lambda item: (
            _case_sort_key(_source_case_key(item)),
            str((item.get("setup_augmentation") or {}).get("augmentation_id", "")),
            str(item["view_key"]),
        ),
    ):
        source = _source_case_key(source_sample)
        split = split_assignment.get(source)
        if split is None:
            skipped_sources.add(source)
            continue
        view_key = str(source_sample["view_key"])
        if augmentation_selection == "base":
            sample_id = f"{source}_{view_key}"
            case_key = source
        else:
            sample_id = str(source_sample["sample_id"])
            suffix = f"_{view_key}"
            case_key = str(source_sample.get("case_key") or (sample_id[: -len(suffix)] if sample_id.endswith(suffix) else sample_id))
        output_sample_path = out_root / view_key / split / f"{sample_id}.npz"
        source_npz_path = resolve_cache_path(source_root, str(source_sample["sample_path"]))
        _write_sample_npz(
            source_npz_path,
            output_sample_path,
            source_root=source_root,
            split=split,
            sample_id=sample_id,
            case_key=case_key,
            augmentation_selection=augmentation_selection,
        )
        derived_samples.append(
            _manifest_sample(
                source_sample,
                source_root=source_root,
                output_root=out_root,
                output_sample_path=output_sample_path,
                split=split,
                sample_id=sample_id,
                case_key=case_key,
                augmentation_selection=augmentation_selection,
            )
        )

    train_ratio = float(source_manifest.get("split", {}).get("train_ratio", 0.8))
    split_seed = int(source_manifest.get("split", {}).get("seed", 0))
    if normalized_policy == PATIENT_LEVEL_POLICY:
        split_payload = read_json(patient_split_manifest)
        split_seed = int(split_payload.get("seed", split_seed))
        train_ratio = float(1.0 - float(split_payload.get("test_ratio", 1.0 - train_ratio)))

    manifest = dict(source_manifest)
    manifest["cache_version"] = out_root.name
    manifest["generated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["generated_command"] = command
    manifest["views"] = (
        {view: source_manifest["views"][view] for view in selected_views}
        if isinstance(source_manifest.get("views"), dict)
        else list(selected_views)
    )
    if augmentation_selection == "base":
        manifest["setup_augmentation"] = setup_augmentation_config(enabled=False).to_dict()
    manifest["split_manifest"] = (
        {
            "path": str(patient_split_manifest.resolve()),
            "protocol": read_json(patient_split_manifest).get("protocol"),
        }
        if normalized_policy == PATIENT_LEVEL_POLICY
        else None
    )
    manifest["split"] = _split_summary(
        derived_samples,
        split_policy=normalized_policy,
        split_seed=split_seed,
        train_ratio=train_ratio,
    )
    source_case_keys = sorted({_source_case_key(sample) for sample in derived_samples}, key=_case_sort_key)
    manifest["source_case_counts"] = {
        **dict(source_manifest.get("source_case_counts", {})),
        "derived_source_cases": len(source_case_keys),
        "derived_skipped_source_cases": len(skipped_sources),
        "setup_augmentation_combinations_per_source_case": 1
        if augmentation_selection == "base"
        else source_manifest.get("setup_augmentation", {}).get("combination_count_per_source_case"),
    }
    manifest["derived_cache"] = {
        "source_cache_root": str(source_root),
        "source_cache_manifest": str((source_root / "cache_manifest.json").resolve()),
        "asset_reference_mode": "absolute_paths",
        "split_policy": normalized_policy,
        "augmentation_selection": augmentation_selection,
        "base_augmentation_id": BASE_AUGMENTATION_ID if augmentation_selection == "base" else None,
        "source_case_count": len(source_case_keys),
        "skipped_source_cases": sorted(skipped_sources, key=_case_sort_key),
    }
    manifest["view_sample_counts"] = _view_sample_counts(derived_samples, selected_views)
    manifest["samples"] = derived_samples
    manifest["failed_cases"] = []

    write_json(manifest, out_root / "cache_manifest.json")
    write_json(
        {
            "ok": True,
            "cache_root": str(out_root),
            "source_cache_root": str(source_root),
            "split_policy": normalized_policy,
            "augmentation_selection": augmentation_selection,
            "view_sample_counts": manifest["view_sample_counts"],
            "source_case_count": len(source_case_keys),
            "skipped_source_case_count": len(skipped_sources),
        },
        out_root / "reports" / "cache_summary.json",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive a lightweight real_projection DIF-Net cache from a setup-augmented asset cache.")
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-policy", type=normalize_split_policy, choices=(PATIENT_LEVEL_POLICY, CASE_MIXED_POLICY), required=True)
    parser.add_argument("--augmentation-selection", choices=("base", "all"), required=True)
    parser.add_argument("--views", default="all")
    parser.add_argument("--patient-split-manifest", type=Path, default=PROJECT_ROOT / "data_manifest" / "splits_v0b_train_test.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = derive_real_difnet_cache(
        source_cache_root=args.source_cache_root,
        output_root=args.output_root,
        split_policy=args.split_policy,
        augmentation_selection=args.augmentation_selection,
        views=args.views,
        patient_split_manifest=args.patient_split_manifest,
        overwrite=args.overwrite,
        command=" ".join(sys.argv),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(args.output_root.resolve()),
                "view_sample_counts": manifest["view_sample_counts"],
                "source_case_count": manifest["derived_cache"]["source_case_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
