from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json, write_json
from x2cbct_difnet.luna16_v1_preprocess import materialize_luna16_registry
from x2cbct_difnet.setup_augmentation import bin_label


def _parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        text = item.strip()
        if text:
            out.append(int(text))
    if 0 not in out:
        raise ValueError("augmentation bins must include 0")
    return list(dict.fromkeys(out))


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _seriesuid(row: dict[str, Any]) -> str:
    for key in ("seriesuid", "source_case_id", "patient_id", "case_id", "case_key"):
        value = row.get(key)
        if value:
            text = _remove_prefix(str(value), "luna16_")
            if text.startswith("subset") and "_" in text:
                maybe_subset, rest = text.split("_", 1)
                if _remove_prefix(maybe_subset, "subset").isdigit():
                    return rest
            return text
    raise ValueError(f"cannot derive seriesuid from row: {row}")


def _augmentation_id(translation_mm: int, rotation_deg: int) -> str:
    return f"si_{bin_label(int(translation_mm))}_rot_{bin_label(int(rotation_deg))}"


def _safe_rmtree(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    target = path.resolve()
    if not str(target).lower().startswith(str(root).lower()):
        raise RuntimeError(f"refusing to remove outside workspace: {target}")
    if target in {root, root / "data_cache", root / "data_manifest", root / "dataset_list"}:
        raise RuntimeError(f"refusing to remove broad workspace path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _run(command: list[str], *, stage: str) -> None:
    print(json.dumps({"event": "stage_start", "stage": stage}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(json.dumps({"event": "stage_done", "stage": stage}, ensure_ascii=False), flush=True)


def _write_old_eval_derivative_manifest(
    *,
    registry_path: Path,
    output_path: Path,
    translation_mm: list[int],
    rotation_deg: list[int],
    command: str,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    cases = []
    for row in registry.get("cases", []):
        if str(row.get("split", "")).strip().lower() != "eval":
            continue
        case = {
            key: value
            for key, value in row.items()
            if key not in {"split", "upstream_split", "split_source", "split_policy"}
        }
        seriesuid = _seriesuid(case)
        case.setdefault("case_key", seriesuid)
        case.setdefault("seriesuid", seriesuid)
        case.setdefault("patient_id", seriesuid)
        cases.append(case)
    cases.sort(key=lambda item: str(item.get("seriesuid") or item.get("case_key")))
    if not cases:
        raise ValueError(f"no old eval cases found in {registry_path}")
    payload = {
        "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "name": output_path.stem,
        "dataset": "luna16",
        "split_policy": "temporary_old_eval_augmented_replay",
        "command": command,
        "augmentation": {
            "enabled": True,
            "mode": "materialized_eval_replay",
            "translation_mm": translation_mm,
            "rotation_deg": rotation_deg,
            "combination_count_per_source_case": len(translation_mm) * len(rotation_deg),
            "excluded_augmentation_ids": [_augmentation_id(0, 0)],
            "effective_combination_count_per_source_case": len(translation_mm) * len(rotation_deg) - 1,
        },
        "split": {
            "policy": "temporary_old_eval_augmented_replay",
            "source_registry": str(registry_path.resolve()),
            "source_old_split": "eval",
            "output_split": "eval",
            "case_overlap_count": 0,
            "patient_overlap": [],
        },
        "counts": {
            "eval_case_count": len(cases),
            "eval_patient_count": len(cases),
            "requested_combination_count_per_case": len(translation_mm) * len(rotation_deg),
            "effective_combination_count_per_case": len(translation_mm) * len(rotation_deg) - 1,
        },
        "splits": {
            "test": {
                "case_count": len(cases),
                "patient_count": len(cases),
                "source_old_split": "eval",
                "cases": cases,
            }
        },
    }
    write_json(payload, output_path)
    return payload


def _filter_drr_manifest(input_path: Path, output_path: Path, excluded_ids: set[str]) -> dict[str, Any]:
    manifest = read_json(input_path)
    cases = list(manifest.get("cases", []))
    kept = []
    removed = []
    for row in cases:
        setup = row.get("setup_augmentation") if isinstance(row.get("setup_augmentation"), dict) else {}
        aug_id = setup.get("augmentation_id")
        if aug_id in excluded_ids:
            removed.append(row)
        else:
            kept.append(row)
    manifest["cases"] = kept
    manifest["generated_count"] = len(kept)
    manifest["filtered_out_count"] = len(removed)
    manifest["filtered_out_augmentation_ids"] = sorted(excluded_ids)
    setup = manifest.get("setup_augmentation")
    if isinstance(setup, dict):
        setup["excluded_augmentation_ids"] = sorted(excluded_ids)
        setup["effective_combination_count_per_source_case"] = len(
            {
                (int((row.get("setup_augmentation") or {}).get("translation_mm", 0)), int((row.get("setup_augmentation") or {}).get("rotation_deg", 0)))
                for row in kept
            }
        )
    write_json(manifest, output_path)
    return manifest


def _patch_cache_manifest(cache_root: Path, *, split: str, source_manifest: Path, build_summary: dict[str, Any]) -> dict[str, Any]:
    manifest_path = cache_root / "cache_manifest.json"
    manifest = read_json(manifest_path)
    samples = list(manifest.get("samples", []))
    for sample in samples:
        sample["split"] = split
        sample["old_luna16_split"] = "eval"
    case_ids = {
        str(sample.get("source_case_id") or sample.get("case_id", "")).split("_si_", 1)[0]
        for sample in samples
    }
    augmentation_ids = sorted({str(sample.get("augmentation_id")) for sample in samples if sample.get("augmentation_id")})
    manifest["samples"] = samples
    manifest["cache_version"] = cache_root.parent.name
    manifest["test_is_validation"] = True
    manifest["split_policy"] = "temporary_old_eval_augmented_replay"
    manifest["counts"] = {
        "cases": len(case_ids),
        "augmented_cases": len(case_ids) * len(augmentation_ids),
        "split": {split: len(case_ids) * len(augmentation_ids)},
        "samples": len(samples),
    }
    manifest["split"] = {
        "policy": "temporary_old_eval_augmented_replay",
        "split_unit": "case",
        "source_old_split": "eval",
        "output_split": split,
        "case_count": len(case_ids),
        "augmentation_count_per_case": len(augmentation_ids),
        "augmented_case_count": len(case_ids) * len(augmentation_ids),
        "case_overlap_count": 0,
        "has_validation_split": split == "eval",
    }
    setup = manifest.get("setup_augmentation")
    if isinstance(setup, dict):
        setup["excluded_augmentation_ids"] = ["si_0_rot_0"]
        setup["effective_combination_count_per_source_case"] = len(augmentation_ids)
    manifest["old_eval_augmented_cache"] = {
        "source_manifest": str(source_manifest.resolve()),
        "source_old_split": "eval",
        "output_split": split,
        "excluded_augmentation_ids": ["si_0_rot_0"],
        "build_summary": build_summary,
    }
    write_json(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an augmented LUNA16 old-eval replay cache without the base setup sample.")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data_cache" / "luna16" / "temp_old_eval49_aug14_t6_r9_256")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset_list" / "LUNA16")
    parser.add_argument("--translation-mm", default="-6,0,6")
    parser.add_argument("--rotation-deg", default="-9,-4,0,4,9")
    parser.add_argument("--size-preset", choices=("128", "256"), default="256")
    parser.add_argument("--projector", choices=("tigre", "fallback-ray-sum"), default="tigre")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    cache_root = output_root / "luna16"
    work_root = output_root.parent / f"_{output_root.name}_build_tmp"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        _safe_rmtree(output_root)
    if work_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"work root already exists: {work_root}")
        _safe_rmtree(work_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    translation_mm = _parse_ints(args.translation_mm)
    rotation_deg = _parse_ints(args.rotation_deg)
    derivative_manifest_path = work_root / "manifests" / "luna16_old_eval49_aug15_tmp.json"
    derivative_manifest = _write_old_eval_derivative_manifest(
        registry_path=args.registry.resolve(),
        output_path=derivative_manifest_path,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        command=" ".join(sys.argv),
    )

    registry_path = work_root / "registry" / "luna16_old_eval_registry.json"
    materialize_luna16_registry(
        manifest_path=derivative_manifest_path,
        output_path=registry_path,
        source_split="test",
        output_split="test",
        root=args.root,
        command=" ".join(sys.argv),
    )

    preset = int(args.size_preset)
    spacing = "1.6,1.6,1.6" if preset == 256 else "3.2,3.2,3.2"
    preprocess_root = work_root / "preprocessed"
    drr_root = work_root / "drr"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "preprocess_ct_only_volumes.py"),
            "--registry",
            str(registry_path),
            "--output-root",
            str(preprocess_root),
            "--dataset-name",
            "luna16",
            "--image-path-key",
            "ct_path",
            "--case-id-key",
            "seriesuid",
            "--resample-spacing",
            spacing,
            "--target-shape",
            f"{preset},{preset},{preset}",
            "--hu-window-mode",
            "fixed",
            "--hu-fixed-min",
            "-1000",
            "--hu-fixed-max",
            "1024",
            "--drop-hu-after-normalization",
            "--progress-interval",
            str(args.progress_interval),
        ],
        stage="preprocess_ct",
    )

    detector_spacing = 2.4 if preset == 256 else 4.8
    drr_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "build_synthrad2025_drr.py"),
        "--preprocess-report",
        str(preprocess_root / "preprocess_report.json"),
        "--output-root",
        str(drr_root),
        "--projector",
        args.projector,
        "--detector-size",
        str(preset),
        "--detector-spacing-mm",
        str(detector_spacing),
        "--projection-count",
        "10",
        "--dso-mm",
        "800",
        "--dsd-mm",
        "1200",
        "--projection-normalization",
        "c2rv-per-case-max",
        "--projection-loader-denominator",
        "0.2",
        "--target-volume-mode",
        "base_reference",
        "--enable-setup-augmentation",
        "--progress-interval",
        str(args.progress_interval),
    ]
    for value in translation_mm:
        drr_cmd.append(f"--translation-mm={value}")
    for value in rotation_deg:
        drr_cmd.append(f"--rotation-deg={value}")
    _run(drr_cmd, stage="generate_drr")

    filtered_drr_manifest = work_root / "drr_filtered" / "drr_manifest.json"
    filtered = _filter_drr_manifest(drr_root / "drr_manifest.json", filtered_drr_manifest, {"si_0_rot_0"})
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "build_ct_only_view_cache.py"),
            "--drr-manifest",
            str(filtered_drr_manifest),
            "--output-root",
            str(output_root),
            "--dataset-name",
            "luna16",
            "--views",
            "2,5,10",
            "--target-size",
            str(preset),
            "--asset-mode",
            "copy",
        ],
        stage="build_view_cache",
    )

    build_summary = {
        "schema": "x2cbct_luna16_old_eval_augmented_cache_build.v1",
        "output_root": str(output_root),
        "cache_root": str(cache_root.resolve()),
        "registry": str(args.registry.resolve()),
        "source_old_split": "eval",
        "output_split": "eval",
        "translation_mm": translation_mm,
        "rotation_deg": rotation_deg,
        "requested_combination_count_per_source_case": len(translation_mm) * len(rotation_deg),
        "excluded_augmentation_ids": ["si_0_rot_0"],
        "effective_combination_count_per_source_case": len(translation_mm) * len(rotation_deg) - 1,
        "old_eval_case_count": int(derivative_manifest["counts"]["eval_case_count"]),
        "filtered_drr_generated_count": int(filtered["generated_count"]),
        "target_volume_mode": "base_reference",
    }
    stable_derivative_manifest = output_root / "reports" / "old_eval_aug_derivative_manifest.json"
    stable_derivative_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(derivative_manifest_path, stable_derivative_manifest)
    build_summary["derivative_manifest"] = str(stable_derivative_manifest.resolve())
    manifest = _patch_cache_manifest(cache_root, split="eval", source_manifest=stable_derivative_manifest, build_summary=build_summary)
    build_summary["sample_count"] = len(manifest.get("samples", []))
    build_summary["counts"] = manifest.get("counts")
    write_json(build_summary, output_root / "build_summary.json")
    write_json({"ok": True, **build_summary}, output_root / "reports" / "cache_summary.json")
    if not args.keep_intermediate:
        _safe_rmtree(work_root)
    print(json.dumps({"ok": True, **build_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
