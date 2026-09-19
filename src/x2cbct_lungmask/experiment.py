from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.stats
import SimpleITK as sitk

from .adapter import PreparedVolume, prepare_volume
from .manifests import VolumeRecord, load_prediction_records, load_reference_records, read_json, resolve_project_path
from .metrics import average_symmetric_surface_distance_mm, dice_coefficient
from x2cbct_difnet.volume_grid import resized_spacing_preserve_fov_xyz


RIGHT_LUNG_LABEL = 1
LEFT_LUNG_LABEL = 2


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _safe_case_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"


def _prepare(record: VolumeRecord, target_shape_zyx: tuple[int, int, int]) -> PreparedVolume:
    array = np.load(record.path, mmap_mode=None)
    return prepare_volume(
        array,
        source_shape_zyx=record.source_shape_zyx,
        source_spacing_xyz=record.source_spacing_xyz,
        target_shape_zyx=target_shape_zyx,
        source_origin_xyz=record.source_origin_xyz,
        source_direction=record.source_direction,
        normalization=record.normalization,
        value_space=record.value_space,
    )


def _mask_image(mask_zyx: np.ndarray, geometry: PreparedVolume) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_xyz)
    image.SetDirection(geometry.direction)
    return image


def _load_or_segment(
    *,
    inferer: Any,
    prepared: PreparedVolume,
    output_path: Path,
    recompute: bool,
) -> np.ndarray:
    if output_path.exists() and not recompute:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(output_path))).astype(np.uint8)
    mask = np.asarray(inferer.apply(prepared.to_sitk()), dtype=np.uint8)
    labels = set(int(value) for value in np.unique(mask))
    if not labels.issubset({0, RIGHT_LUNG_LABEL, LEFT_LUNG_LABEL}):
        raise ValueError(f"unexpected LungMask R231 labels: {sorted(labels)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_mask_image(mask, prepared), str(output_path), True)
    return mask


def _metric_row(
    *,
    case_id: str,
    method_name: str,
    prediction_mask: np.ndarray,
    reference_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    reconstruction_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "method": method_name,
        "status": "ok",
        "right_voxels": int((prediction_mask == RIGHT_LUNG_LABEL).sum()),
        "left_voxels": int((prediction_mask == LEFT_LUNG_LABEL).sum()),
    }
    for organ, label in (("right", RIGHT_LUNG_LABEL), ("left", LEFT_LUNG_LABEL)):
        pred = prediction_mask == label
        ref = reference_mask == label
        row[f"{organ}_dice_percent"] = 100.0 * dice_coefficient(pred, ref)
        row[f"{organ}_asd_mm"] = average_symmetric_surface_distance_mm(
            pred,
            ref,
            spacing_xyz=spacing_xyz,
        )
        row[f"{organ}_reference_voxels"] = int(ref.sum())
    row["bilateral_dice_percent"] = float(
        np.mean([row["left_dice_percent"], row["right_dice_percent"]])
    )
    row["bilateral_asd_mm"] = float(np.mean([row["left_asd_mm"], row["right_asd_mm"]]))
    if any(row[key] == 0 for key in ("right_voxels", "left_voxels", "right_reference_voxels", "left_reference_voxels")):
        row["status"] = "invalid_mask"
    if reconstruction_metrics:
        for key in ("psnr", "ssim", "mae_hu", "mae_norm"):
            if key in reconstruction_metrics:
                row[f"recon_{key}"] = float(reconstruction_metrics[key])
    return row


def _finite_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def summarize_rows(rows: list[dict[str, Any]], methods: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {}
    metric_keys = (
        "recon_psnr",
        "recon_ssim",
        "left_dice_percent",
        "left_asd_mm",
        "right_dice_percent",
        "right_asd_mm",
        "bilateral_dice_percent",
        "bilateral_asd_mm",
    )
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        ok_rows = [row for row in method_rows if row.get("status") == "ok"]
        metrics = {
            key: _finite_summary([float(row[key]) for row in ok_rows if row.get(key) not in (None, "")])
            for key in metric_keys
        }
        summary_json[method] = {
            "case_count": len(method_rows),
            "success_count": len(ok_rows),
            "failure_count": len(method_rows) - len(ok_rows),
            "metrics": metrics,
        }
        flat: dict[str, Any] = {
            "method": method,
            "case_count": len(method_rows),
            "success_count": len(ok_rows),
            "failure_count": len(method_rows) - len(ok_rows),
        }
        for key, value in metrics.items():
            flat[f"{key}_mean"] = value["mean"]
            flat[f"{key}_std"] = value["std"]
        summary_rows.append(flat)
    return summary_rows, summary_json


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, iterations: int = 10000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    primary_method: str,
    methods: list[str],
    seed: int = 20260724,
) -> list[dict[str, Any]]:
    metric_specs = {
        "bilateral_dice_percent": 1.0,
        "bilateral_asd_mm": -1.0,
    }
    by_method_case = {
        method: {
            str(row["case_id"]): row
            for row in rows
            if row["method"] == method and row.get("status") == "ok"
        }
        for method in methods
    }
    if primary_method not in by_method_case:
        return []
    comparisons: list[dict[str, Any]] = []
    for metric, direction in metric_specs.items():
        metric_rows: list[dict[str, Any]] = []
        for baseline in methods:
            if baseline == primary_method:
                continue
            shared = sorted(set(by_method_case[primary_method]) & set(by_method_case[baseline]))
            primary_values: list[float] = []
            baseline_values: list[float] = []
            for case_id in shared:
                primary_value = float(by_method_case[primary_method][case_id][metric])
                baseline_value = float(by_method_case[baseline][case_id][metric])
                if np.isfinite(primary_value) and np.isfinite(baseline_value):
                    primary_values.append(primary_value)
                    baseline_values.append(baseline_value)
            primary_array = np.asarray(primary_values, dtype=np.float64)
            baseline_array = np.asarray(baseline_values, dtype=np.float64)
            improvement = direction * (primary_array - baseline_array)
            if improvement.size == 0:
                p_value = float("nan")
                ci_low, ci_high = float("nan"), float("nan")
            else:
                ci_low, ci_high = _bootstrap_mean_ci(improvement, seed=seed + len(metric_rows))
                p_value = (
                    1.0
                    if np.allclose(improvement, 0.0)
                    else float(scipy.stats.wilcoxon(improvement, alternative="two-sided").pvalue)
                )
            metric_rows.append(
                {
                    "primary_method": primary_method,
                    "baseline_method": baseline,
                    "metric": metric,
                    "positive_improvement_definition": (
                        "primary-minus-baseline" if direction > 0 else "baseline-minus-primary"
                    ),
                    "paired_case_count": int(improvement.size),
                    "primary_mean": float(primary_array.mean()) if primary_array.size else None,
                    "baseline_mean": float(baseline_array.mean()) if baseline_array.size else None,
                    "mean_improvement": float(improvement.mean()) if improvement.size else None,
                    "improvement_ci95_low": ci_low if np.isfinite(ci_low) else None,
                    "improvement_ci95_high": ci_high if np.isfinite(ci_high) else None,
                    "wilcoxon_p": p_value if np.isfinite(p_value) else None,
                }
            )
        finite_order = sorted(
            range(len(metric_rows)),
            key=lambda index: (
                metric_rows[index]["wilcoxon_p"] is None,
                metric_rows[index]["wilcoxon_p"] if metric_rows[index]["wilcoxon_p"] is not None else 1.0,
            ),
        )
        count = sum(metric_rows[index]["wilcoxon_p"] is not None for index in finite_order)
        running_max = 0.0
        for rank, index in enumerate(finite_order[:count], start=1):
            adjusted = min(1.0, float(metric_rows[index]["wilcoxon_p"]) * (count - rank + 1))
            running_max = max(running_max, adjusted)
            metric_rows[index]["holm_adjusted_p"] = running_max
        for index in finite_order[count:]:
            metric_rows[index]["holm_adjusted_p"] = None
        comparisons.extend(metric_rows)
    return comparisons


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_experiment(config_path: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, VolumeRecord], dict[str, dict[str, VolumeRecord]]]:
    config = read_json(config_path)
    reference_config = config["reference"]
    reference_manifest = resolve_project_path(
        reference_config["cache_manifest"],
        project_root=project_root,
        manifest_path=config_path,
    )
    view_key = str(config["view_key"])
    reference_records = load_reference_records(
        reference_manifest,
        project_root=project_root,
        view_key=view_key,
        value_space=str(reference_config.get("value_space", "normalized")),
    )
    methods: dict[str, dict[str, VolumeRecord]] = {}
    for method in config["methods"]:
        if not bool(method.get("enabled", True)):
            continue
        name = str(method["name"])
        manifest_path = resolve_project_path(
            method["prediction_manifest"],
            project_root=project_root,
            manifest_path=config_path,
        )
        methods[name] = load_prediction_records(
            manifest_path,
            project_root=project_root,
            reference_records=reference_records,
            augmentation_id=method.get("augmentation_id"),
            value_space=str(method.get("value_space", "normalized")),
        )
    if not methods:
        raise ValueError("experiment config has no enabled methods")
    return config, reference_records, methods


def audit_experiment(config_path: Path, project_root: Path) -> dict[str, Any]:
    config, references, methods = _load_experiment(config_path, project_root)
    reference_cases = set(references)
    method_report: dict[str, Any] = {}
    errors: list[str] = []
    for name, records in methods.items():
        cases = set(records)
        missing = sorted(reference_cases - cases)
        extra = sorted(cases - reference_cases)
        missing_files = sorted(str(record.path) for record in records.values() if not record.path.exists())
        shape_counts: Counter[str] = Counter()
        for record in records.values():
            if record.path.exists():
                shape_counts[str(tuple(int(value) for value in np.load(record.path, mmap_mode="r").shape))] += 1
        method_report[name] = {
            "case_count": len(cases),
            "missing_reference_cases": missing,
            "extra_cases": extra,
            "missing_volume_files": missing_files,
            "array_shape_counts": dict(sorted(shape_counts.items())),
        }
        if missing or extra or missing_files:
            errors.append(name)
    missing_reference_files = sorted(str(record.path) for record in references.values() if not record.path.exists())
    reference_shape_counts: Counter[str] = Counter()
    for record in references.values():
        if record.path.exists():
            reference_shape_counts[str(tuple(int(value) for value in np.load(record.path, mmap_mode="r").shape))] += 1
    if missing_reference_files:
        errors.append("reference")
    target_shape = tuple(int(value) for value in config.get("target_shape_zyx", [128, 128, 128]))
    sample_reference = next(iter(references.values()))
    spacing = resized_spacing_preserve_fov_xyz(
        sample_reference.source_shape_zyx,
        sample_reference.source_spacing_xyz,
        target_shape,
    )
    return {
        "schema": "x2cbct_lungmask_audit.v1",
        "ok": not errors,
        "config": str(config_path),
        "view_key": config["view_key"],
        "reference_case_count": len(references),
        "target_shape_zyx": list(target_shape),
        "target_spacing_xyz": list(spacing),
        "reference_array_shape_counts": dict(sorted(reference_shape_counts.items())),
        "missing_reference_files": missing_reference_files,
        "methods": method_report,
        "failed_sections": errors,
        "disabled_methods": [
            {"name": method["name"], "reason": method.get("reason", "")}
            for method in config["methods"]
            if not bool(method.get("enabled", True))
        ],
    }


def run_experiment(
    config_path: Path,
    *,
    project_root: Path,
    output_dir: Path,
    max_cases: int | None,
    recompute: bool,
) -> dict[str, Any]:
    audit = audit_experiment(config_path, project_root)
    if not audit["ok"]:
        raise RuntimeError(f"experiment audit failed; inspect {output_dir / 'audit_report.json'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(audit, output_dir / "audit_report.json")
    config, references, methods = _load_experiment(config_path, project_root)
    target_shape = tuple(int(value) for value in config.get("target_shape_zyx", [128, 128, 128]))
    case_ids = sorted(references)
    if max_cases is not None:
        case_ids = case_ids[: max(0, int(max_cases))]

    from lungmask import LMInferer
    import torch

    lungmask_config = config.get("lungmask", {})
    force_cpu = bool(lungmask_config.get("force_cpu", False))
    inferer = LMInferer(
        modelname=str(lungmask_config.get("modelname", "R231")),
        force_cpu=force_cpu,
        batch_size=int(lungmask_config.get("batch_size", 20)),
        volume_postprocessing=bool(lungmask_config.get("volume_postprocessing", True)),
        tqdm_disable=bool(lungmask_config.get("tqdm_disable", False)),
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, case_id in enumerate(case_ids, start=1):
        safe_case = _safe_case_name(case_id)
        try:
            reference_prepared = _prepare(references[case_id], target_shape)
            reference_mask = _load_or_segment(
                inferer=inferer,
                prepared=reference_prepared,
                output_path=output_dir / "masks" / "reference" / f"{safe_case}.nii.gz",
                recompute=recompute,
            )
            reference_labels = set(int(value) for value in np.unique(reference_mask))
            if not {RIGHT_LUNG_LABEL, LEFT_LUNG_LABEL}.issubset(reference_labels):
                raise ValueError(f"reference mask is missing a lung label: {sorted(reference_labels)}")
        except Exception as exc:
            failures.append({"case_id": case_id, "method": "reference", "error": f"{type(exc).__name__}: {exc}"})
            continue

        for method_name, method_records in methods.items():
            try:
                prediction_prepared = _prepare(method_records[case_id], target_shape)
                if not np.allclose(prediction_prepared.spacing_xyz, reference_prepared.spacing_xyz, atol=1e-6):
                    raise ValueError(
                        f"spacing mismatch: {prediction_prepared.spacing_xyz} vs {reference_prepared.spacing_xyz}"
                    )
                prediction_mask = _load_or_segment(
                    inferer=inferer,
                    prepared=prediction_prepared,
                    output_path=output_dir / "masks" / method_name / f"{safe_case}.nii.gz",
                    recompute=recompute,
                )
                rows.append(
                    _metric_row(
                        case_id=case_id,
                        method_name=method_name,
                        prediction_mask=prediction_mask,
                        reference_mask=reference_mask,
                        spacing_xyz=reference_prepared.spacing_xyz,
                        reconstruction_metrics=method_records[case_id].reconstruction_metrics,
                    )
                )
            except Exception as exc:
                failure = {"case_id": case_id, "method": method_name, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                rows.append({"case_id": case_id, "method": method_name, "status": "failed", "error": failure["error"]})
        print(f"lungmask {position}/{len(case_ids)} case_id={case_id}", flush=True)

    method_names = list(methods)
    summary_rows, summary_json = summarize_rows(rows, method_names)
    comparisons = paired_comparisons(
        rows,
        primary_method=str(config.get("primary_method", "")),
        methods=method_names,
    )
    _write_csv(rows, output_dir / "metrics_per_case.csv")
    _write_csv(summary_rows, output_dir / "metrics_summary.csv")
    if comparisons:
        _write_csv(comparisons, output_dir / "paired_comparisons.csv")
    _write_json(summary_json, output_dir / "metrics_summary.json")
    _write_json(failures, output_dir / "failures.json")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lungmask": importlib.metadata.version("lungmask"),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(environment, output_dir / "environment.json")
    result = {
        "schema": "x2cbct_lungmask_experiment_result.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "case_count": len(case_ids),
        "method_count": len(methods),
        "metric_row_count": len(rows),
        "failure_count": len(failures),
        "paired_comparison_count": len(comparisons),
        "environment": environment,
    }
    _write_json(result, output_dir / "experiment_result.json")
    return result


def main(project_root: Path | None = None) -> int:
    root = project_root or Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Audit or run the X2CBCT LungMask downstream experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--max-cases", type=int)
    run_parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config, project_root=root)
    if args.command == "audit":
        report = audit_experiment(config_path, root)
        if args.output:
            _write_json(report, resolve_project_path(args.output, project_root=root))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2
    output_dir = resolve_project_path(args.output_dir, project_root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_experiment(
            config_path,
            project_root=root,
            output_dir=output_dir,
            max_cases=args.max_cases,
            recompute=bool(args.recompute),
        )
    except Exception:
        audit_path = output_dir / "audit_report.json"
        if not audit_path.exists():
            try:
                _write_json(audit_experiment(config_path, root), audit_path)
            except Exception as audit_exc:
                _write_json({"ok": False, "error": f"{type(audit_exc).__name__}: {audit_exc}"}, audit_path)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
