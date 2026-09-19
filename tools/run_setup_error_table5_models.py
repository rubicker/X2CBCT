from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS

AXIS_RUNNER = PROJECT_ROOT / "tools" / "run_setup_error_axis_max_experiment.py"
DEFAULT_MODEL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "setup_error"
    / "table5_models_v5_grid128.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "model_experiments"
    / "table5_view5_grid128"
)
DEFAULT_CACHE_ROOTS = [
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "synthetic_test_stress_cases"
    / "seed_20260724",
    PROJECT_ROOT
    / "data_cache"
    / "difnet_v1_si10_rot10_3x3_256"
    / "self_projection1208"
    / "train_mixed_noaug_256",
]
DEFAULT_FORMAL_RESULTS = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "synthetic_test_stress_cases"
    / "seed_20260724"
    / "registration_validation"
    / "registration_results.csv"
)
DEFAULT_ROI_REVIEWED = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "model_experiments"
    / "_shared_baseline"
    / "cervical_roi_review"
    / "roi_reviewed.csv"
)
COMMON_TRUTH_ABS_TOLERANCE = 5e-5


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_model_config(path: Path) -> dict[str, Any]:
    resolved = _project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"model config does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("models"), list):
        raise ValueError(f"model config must contain a models list: {resolved}")
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for index, raw in enumerate(config["models"]):
        if not isinstance(raw, dict):
            raise ValueError(f"models[{index}] must be an object")
        model = dict(raw)
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            raise ValueError(f"models[{index}] has no id")
        if model_id in seen:
            raise ValueError(f"duplicate model id: {model_id}")
        seen.add(model_id)
        backend = str(model.get("backend", "")).strip()
        if backend not in {"difnet", "deepsparse"}:
            raise ValueError(
                f"model {model_id} has unsupported backend={backend!r}"
            )
        checkpoint = _project_path(str(model.get("checkpoint", "")))
        model_python = (
            _project_path(str(model["python"]))
            if model.get("python")
            else None
        )
        model.update(
            {
                "id": model_id,
                "display_name": str(
                    model.get("display_name", model_id)
                ).strip(),
                "enabled": bool(model.get("enabled", True)),
                "backend": backend,
                "checkpoint": checkpoint,
                "model_label": str(
                    model.get("model_label", model_id)
                ).strip(),
                "chunk_size": int(model.get("chunk_size", 30_000)),
            }
        )
        if model_python is not None:
            model["python"] = model_python
        if model.get("mid_channels") is not None:
            model["mid_channels"] = int(model["mid_channels"])
        models.append(model)
    result = dict(config)
    result["path"] = resolved
    result["models"] = models
    return result


def _parse_assignments(
    values: list[str] | None,
    *,
    cast: Callable[[str], Any],
    option_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values or []:
        model_id, separator, raw = value.partition("=")
        model_id = model_id.strip()
        raw = raw.strip()
        if not separator or not model_id or not raw:
            raise ValueError(
                f"{option_name} must use MODEL=VALUE, got {value!r}"
            )
        if model_id in result:
            raise ValueError(
                f"{option_name} was repeated for model {model_id!r}"
            )
        result[model_id] = cast(raw)
    return result


def _selected_models(
    config: dict[str, Any],
    selectors: list[str] | None,
) -> list[dict[str, Any]]:
    models = list(config["models"])
    by_id = {str(model["id"]): model for model in models}
    if not selectors:
        selected = [model for model in models if model["enabled"]]
    else:
        requested: list[str] = []
        for selector in selectors:
            requested.extend(
                part.strip()
                for part in selector.split(",")
                if part.strip()
            )
        unknown = [model_id for model_id in requested if model_id not in by_id]
        if unknown:
            raise KeyError(
                f"unknown model ids {unknown}; available={list(by_id)}"
            )
        selected = [by_id[model_id] for model_id in requested]
    if not selected:
        raise ValueError("no models were selected")
    return [dict(model) for model in selected]


def _apply_overrides(
    models: list[dict[str, Any]],
    *,
    checkpoints: dict[str, Path],
    chunk_sizes: dict[str, int],
) -> list[dict[str, Any]]:
    selected_ids = {str(model["id"]) for model in models}
    unknown = (set(checkpoints) | set(chunk_sizes)) - selected_ids
    if unknown:
        raise KeyError(
            f"overrides target models that are not selected: {sorted(unknown)}"
        )
    result = []
    for source in models:
        model = dict(source)
        model_id = str(model["id"])
        if model_id in checkpoints:
            model["checkpoint"] = checkpoints[model_id]
        if model_id in chunk_sizes:
            model["chunk_size"] = int(chunk_sizes[model_id])
        result.append(model)
    return result


def _validate_models(models: list[dict[str, Any]]) -> None:
    for model in models:
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint for {model['id']} does not exist: {checkpoint}"
            )
        if checkpoint.stat().st_size <= 0:
            raise ValueError(
                f"checkpoint for {model['id']} is empty: {checkpoint}"
            )
        if int(model["chunk_size"]) < 1:
            raise ValueError(
                f"chunk_size for {model['id']} must be >= 1"
            )
        model_python = model.get("python")
        if model_python is not None and not Path(model_python).is_file():
            raise FileNotFoundError(
                f"Python executable for {model['id']} does not exist: "
                f"{model_python}"
            )


def _axis_command(
    args: argparse.Namespace,
    model: dict[str, Any],
    *,
    output_dir: Path,
) -> list[str]:
    command = [
        str(model.get("python", args.python)),
        str(AXIS_RUNNER),
        "--backend",
        str(model["backend"]),
        "--checkpoint",
        str(model["checkpoint"]),
        "--model-label",
        str(model["model_label"]),
        "--experiment",
        "both",
        "--output-dir",
        str(output_dir),
        "--formal-registration-results",
        str(args.formal_registration_results),
        "--roi-reviewed",
        str(args.roi_reviewed),
        "--si-pair",
        args.si_pair,
        "--yaw-pair",
        args.yaw_pair,
        "--require-inference-split",
        args.require_inference_split,
        "--view-count",
        str(args.view_count),
        "--eval-grid-size",
        str(args.eval_grid_size),
        "--chunk-size",
        str(model["chunk_size"]),
        "--grid-control-si-tolerance-mm",
        str(args.grid_control_si_tolerance_mm),
        "--grid-control-yaw-tolerance-deg",
        str(args.grid_control_yaw_tolerance_deg),
        "--device",
        args.device,
    ]
    for cache_root in args.cache_root:
        command.extend(["--cache-root", str(cache_root)])
    if model.get("mid_channels") is not None:
        command.extend(["--mid-channels", str(model["mid_channels"])])
    if args.skip_viewer:
        command.append("--skip-viewer")
    if args.enforce_grid_control:
        command.append("--enforce-grid-control")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_model_output(path: Path) -> bool:
    summary = path / "run_summary.json"
    table = path / "table_ready_results.csv"
    if not summary.is_file() or not table.is_file():
        return False
    try:
        with summary.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("ok"))


def _reusable_model_output(path: Path, checkpoint: Path) -> bool:
    if not _completed_model_output(path):
        return False
    try:
        with (path / "run_summary.json").open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        previous_checkpoint = Path(str(payload["checkpoint"]))
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    return previous_checkpoint == Path(checkpoint)


def _archive_model_output(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"model output directory does not exist: {path}")
    archive_root = path.parent / "_resume_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = archive_root / f"{path.name}_{stamp}"
    path.replace(archive_path)
    return archive_path


def _auto_resume_if_needed(args: argparse.Namespace) -> bool:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(
            f"batch output exists and is not a directory: {output_dir}"
        )
    if args.resume or not output_dir.exists() or not any(output_dir.iterdir()):
        return False
    args.resume = True
    print(
        f"[batch:auto-resume] reusing existing batch output: {output_dir}",
        flush=True,
    )
    return True


def _verify_common_truth(rows: list[dict[str, str]]) -> None:
    by_axis: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_axis.setdefault(str(row["axis_experiment"]), []).append(row)
    if set(by_axis) != {"si-max", "yaw-max"}:
        raise ValueError(
            f"cross-model table must contain si-max and yaw-max, got {set(by_axis)}"
        )
    for axis, axis_rows in by_axis.items():
        reference = axis_rows[0]
        for field in ("true_SI_mm", "true_yaw_deg"):
            expected = float(reference[field])
            for row in axis_rows[1:]:
                actual = float(row[field])
                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=COMMON_TRUTH_ABS_TOLERANCE,
                ):
                    raise ValueError(
                        f"{axis} uses inconsistent C128 {field}: "
                        f"{actual} != {expected} "
                        f"(tolerance={COMMON_TRUTH_ABS_TOLERANCE})"
                    )


def _wide_rows(
    long_rows: list[dict[str, str]],
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_model_axis = {
        (str(row["model_id"]), str(row["axis_experiment"])): row
        for row in long_rows
    }
    result = []
    for model in models:
        model_id = str(model["id"])
        si = by_model_axis[(model_id, "si-max")]
        yaw = by_model_axis[(model_id, "yaw-max")]
        result.append(
            {
                "model_id": model_id,
                "model": model["display_name"],
                "checkpoint": str(model["checkpoint"]),
                "si_pair_true_C128_SI_mm": si["true_SI_mm"],
                "si_pair_reconstructed_c128_SI_mm": si["reconstructed_SI_mm"],
                "si_pair_abs_error_SI_mm": si["abs_error_SI_mm"],
                "si_pair_true_C128_yaw_deg": si["true_yaw_deg"],
                "si_pair_reconstructed_c128_yaw_deg": si["reconstructed_yaw_deg"],
                "si_pair_abs_error_yaw_deg": si["abs_error_yaw_deg"],
                "yaw_pair_true_C128_SI_mm": yaw["true_SI_mm"],
                "yaw_pair_reconstructed_c128_SI_mm": yaw["reconstructed_SI_mm"],
                "yaw_pair_abs_error_SI_mm": yaw["abs_error_SI_mm"],
                "yaw_pair_true_C128_yaw_deg": yaw["true_yaw_deg"],
                "yaw_pair_reconstructed_c128_yaw_deg": yaw["reconstructed_yaw_deg"],
                "yaw_pair_abs_error_yaw_deg": yaw["abs_error_yaw_deg"],
                "si_grid_control_pass": si["grid_control_pass"],
                "yaw_grid_control_pass": yaw["grid_control_pass"],
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Table 5: cross-model setup-error comparison",
        "",
        "All values use the selected held-out 5-view test samples and a native "
        "128^3 registration grid. C128 is measured from resampled reference/target "
        "CBCTs; c128 is measured from each model reconstruction.",
        "",
        "| Model | S-I C128 (mm) | S-I c128 (mm) | Abs. error S-I (mm) | yaw C128 (deg) | yaw c128 (deg) | Abs. error yaw (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{float(row['si_pair_true_C128_SI_mm']):.4f} | "
            f"{float(row['si_pair_reconstructed_c128_SI_mm']):.4f} | "
            f"{float(row['si_pair_abs_error_SI_mm']):.4f} | "
            f"{float(row['yaw_pair_true_C128_yaw_deg']):.4f} | "
            f"{float(row['yaw_pair_reconstructed_c128_yaw_deg']):.4f} | "
            f"{float(row['yaw_pair_abs_error_yaw_deg']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Table-5 128-grid setup-error experiments for DIF-Net, DuFal, "
            "C2RV, DeepSparse, and CaSE-CBCT, then aggregate a cross-model table."
        )
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        help="Run only these model ids. Repeat or use comma-separated ids.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Override one checkpoint with MODEL=PATH. Repeat as needed.",
    )
    parser.add_argument(
        "--model-chunk-size",
        action="append",
        help="Override one chunk size with MODEL=INTEGER. Repeat as needed.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cache-root", type=Path, action="append")
    parser.add_argument(
        "--formal-registration-results",
        type=Path,
        default=DEFAULT_FORMAL_RESULTS,
    )
    parser.add_argument("--roi-reviewed", type=Path, default=DEFAULT_ROI_REVIEWED)
    parser.add_argument("--si-pair", default="6:0:2")
    parser.add_argument("--yaw-pair", default="0:0:6")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=5)
    parser.add_argument("--eval-grid-size", type=int, default=128)
    parser.add_argument(
        "--require-inference-split",
        choices=("any", "train", "test", "eval"),
        default="test",
    )
    parser.add_argument(
        "--grid-control-si-tolerance-mm",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--grid-control-yaw-tolerance-deg",
        type=float,
        default=0.5,
    )
    parser.add_argument("--enforce-grid-control", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _resolve_args(args: argparse.Namespace) -> None:
    args.python = _project_path(args.python)
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {args.python}")
    args.model_config = _project_path(args.model_config)
    args.output_dir = _project_path(args.output_dir)
    args.formal_registration_results = _project_path(
        args.formal_registration_results
    )
    args.roi_reviewed = _project_path(args.roi_reviewed)
    args.cache_root = [
        _project_path(path)
        for path in (
            args.cache_root if args.cache_root is not None else DEFAULT_CACHE_ROOTS
        )
    ]
    if int(args.eval_grid_size) < 2:
        raise ValueError("--eval-grid-size must be >= 2")
    if not math.isfinite(args.grid_control_si_tolerance_mm) or args.grid_control_si_tolerance_mm < 0:
        raise ValueError("--grid-control-si-tolerance-mm must be finite and non-negative")
    if not math.isfinite(args.grid_control_yaw_tolerance_deg) or args.grid_control_yaw_tolerance_deg < 0:
        raise ValueError("--grid-control-yaw-tolerance-deg must be finite and non-negative")
    required_files = [
        AXIS_RUNNER,
        args.formal_registration_results,
        args.roi_reviewed,
    ]
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")
    for path in args.cache_root:
        if not path.is_dir() or not (path / "cache_manifest.json").is_file():
            raise FileNotFoundError(f"invalid cache root: {path}")


def main() -> int:
    args = _parse_args()
    _resolve_args(args)
    config = _load_model_config(args.model_config)
    selected = _selected_models(config, args.model)
    checkpoint_overrides = _parse_assignments(
        args.checkpoint,
        cast=_project_path,
        option_name="--checkpoint",
    )
    chunk_overrides = _parse_assignments(
        args.model_chunk_size,
        cast=int,
        option_name="--model-chunk-size",
    )
    models = _apply_overrides(
        selected,
        checkpoints=checkpoint_overrides,
        chunk_sizes=chunk_overrides,
    )
    _validate_models(models)

    inventory = {
        "schema": "x2cbct_setup_error_table5_batch_plan.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": str(config["path"]),
        "output_dir": str(args.output_dir),
        "view_count": args.view_count,
        "eval_grid_size": args.eval_grid_size,
        "models": [
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in model.items()
                    if key != "checkpoint"
                },
                "checkpoint": str(model["checkpoint"]),
                "checkpoint_size_bytes": Path(model["checkpoint"]).stat().st_size,
                "output_dir": str(args.output_dir / "model_runs" / model["id"]),
            }
            for model in models
        ],
    }
    if args.list_models:
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return 0

    if args.dry_run:
        results = []
        for model in models:
            output_dir = args.output_dir / "model_runs" / str(model["id"])
            command = _axis_command(args, model, output_dir=output_dir)
            print(
                f"[batch:dry-run] model={model['id']} "
                f"command={subprocess.list2cmdline(command)}",
                flush=True,
            )
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            results.append(
                {
                    "model_id": model["id"],
                    "returncode": completed.returncode,
                }
            )
            if completed.returncode != 0 and not args.continue_on_error:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "dry_run": True,
                            "plan": inventory,
                            "results": results,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return completed.returncode
        ok = all(row["returncode"] == 0 for row in results)
        print(
            json.dumps(
                {
                    "ok": ok,
                    "dry_run": True,
                    "plan": inventory,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if ok else 1

    _auto_resume_if_needed(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "resolved_batch.json", inventory)

    run_records: list[dict[str, Any]] = []
    successful_models: list[dict[str, Any]] = []
    for model in models:
        model_dir = args.output_dir / "model_runs" / str(model["id"])
        if args.resume and _reusable_model_output(model_dir, Path(model["checkpoint"])):
            print(f"[batch:resume] model={model['id']} already complete", flush=True)
            run_records.append(
                {
                    "model_id": model["id"],
                    "status": "reused",
                    "returncode": 0,
                    "output_dir": str(model_dir),
                }
            )
            successful_models.append(model)
            continue
        archived_output: Path | None = None
        if model_dir.exists() and any(model_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(
                    f"incomplete model output is non-empty: {model_dir}; "
                    "choose a new batch path"
                )
            archived_output = _archive_model_output(model_dir)
            print(
                f"[batch:resume] model={model['id']} archived stale output "
                f"to {archived_output}",
                flush=True,
            )
        command = _axis_command(args, model, output_dir=model_dir)
        started_at = datetime.now().isoformat(timespec="seconds")
        print(
            f"[batch:start] model={model['id']} checkpoint={model['checkpoint']}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        record = {
            "model_id": model["id"],
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(model_dir),
            "command": command,
        }
        if archived_output is not None:
            record["archived_output_dir"] = str(archived_output)
        run_records.append(record)
        _write_json(
            args.output_dir / "batch_progress.json",
            {"runs": run_records},
        )
        if completed.returncode == 0:
            successful_models.append(model)
        elif not args.continue_on_error:
            return completed.returncode

    long_rows: list[dict[str, str]] = []
    for model in successful_models:
        table_path = (
            args.output_dir
            / "model_runs"
            / str(model["id"])
            / "table_ready_results.csv"
        )
        if not table_path.is_file():
            raise FileNotFoundError(
                f"successful model has no table output: {table_path}"
            )
        rows = _read_csv(table_path)
        if len(rows) != 2:
            raise ValueError(
                f"model {model['id']} must produce exactly two axis rows, "
                f"got {len(rows)}"
            )
        for row in rows:
            long_rows.append(
                {
                    "model_id": str(model["id"]),
                    "model": str(model["display_name"]),
                    **row,
                }
            )
    if len(successful_models) != len(models):
        _write_json(
            args.output_dir / "run_summary.json",
            {
                "ok": False,
                "reason": "one or more models failed",
                "runs": run_records,
            },
        )
        return 1

    _verify_common_truth(long_rows)
    wide_rows = _wide_rows(long_rows, models)
    _write_csv(
        args.output_dir / "table5_model_comparison_long.csv",
        long_rows,
    )
    _write_csv(
        args.output_dir / "table5_model_comparison.csv",
        wide_rows,
    )
    _write_json(
        args.output_dir / "table5_model_comparison.json",
        {
            "schema": "x2cbct_setup_error_table5_comparison.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": wide_rows,
        },
    )
    _write_markdown(
        args.output_dir / "table5_model_comparison.md",
        wide_rows,
    )
    summary = {
        "ok": True,
        "schema": "x2cbct_setup_error_table5_batch.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_count": len(models),
        "runs": run_records,
        "comparison_csv": str(
            args.output_dir / "table5_model_comparison.csv"
        ),
        "comparison_markdown": str(
            args.output_dir / "table5_model_comparison.md"
        ),
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
