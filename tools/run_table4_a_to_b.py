from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "table4_a_to_b.json"
EPOCH_PATTERN = re.compile(r"(?:checkpoint_)?epoch_(\d+)", re.IGNORECASE)


def _path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    f"PyYAML is required to read {path.suffix} configs; "
                    "use the default JSON registry or install PyYAML"
                ) from exc
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.strip().lower()
        if lowered in {"null", "none", "~"}:
            return None
        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False
        return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _epoch(path: Path) -> int | None:
    match = EPOCH_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def discover_checkpoints(run_dir: Path | None) -> list[Path]:
    if run_dir is None or not run_dir.is_dir():
        return []
    candidates: set[Path] = set()
    for directory in (run_dir, run_dir / "checkpoints"):
        if not directory.is_dir():
            continue
        for pattern in ("*.pt", "*.pth"):
            candidates.update(path.resolve() for path in directory.glob(pattern) if path.is_file())
    return sorted(
        candidates,
        key=lambda path: (
            _epoch(path) is None,
            _epoch(path) if _epoch(path) is not None else -1,
            path.name.lower(),
        ),
    )


def resolve_checkpoint(spec: Any, run_dir: Path | None) -> Path:
    if spec is None or str(spec).strip() == "":
        raise FileNotFoundError("checkpoint is not configured")
    value = str(spec).strip()
    normalized = value.lower()
    candidates = discover_checkpoints(run_dir)

    if normalized == "latest":
        epoch_candidates = [path for path in candidates if _epoch(path) is not None]
        if epoch_candidates:
            return max(epoch_candidates, key=lambda path: (_epoch(path), path.stat().st_mtime_ns))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime_ns)
        raise FileNotFoundError(f"no checkpoint candidates found under {run_dir}")

    if normalized == "best":
        best_candidates = [
            path
            for path in candidates
            if path.stem.lower() in {"best", "best_by_psnr", "best_by_ssim"}
        ]
        if len(best_candidates) == 1:
            return best_candidates[0]
        if not best_candidates:
            raise FileNotFoundError(f"no best checkpoint found under {run_dir}")
        raise ValueError(f"multiple best checkpoints found; use an explicit path: {best_candidates}")

    if normalized.startswith("epoch:"):
        try:
            requested_epoch = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint epoch selector: {value}") from exc
        matches = [path for path in candidates if _epoch(path) == requested_epoch]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"epoch {requested_epoch} checkpoint not found under {run_dir}")
        raise ValueError(f"multiple epoch {requested_epoch} checkpoints found; use an explicit path: {matches}")

    explicit = Path(value)
    attempts: list[Path] = []
    if explicit.is_absolute():
        attempts.append(explicit.resolve())
    else:
        attempts.append((PROJECT_ROOT / explicit).resolve())
        if run_dir is not None:
            attempts.append((run_dir / explicit).resolve())
    existing = list(dict.fromkeys(path for path in attempts if path.is_file()))
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"ambiguous checkpoint path {value}; matches: {existing}")
    raise FileNotFoundError(f"checkpoint not found: {value}; tried: {attempts}")


def _parse_assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects ID=VALUE, got: {value}")
    key, assigned = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"{option} experiment id cannot be empty: {value}")
    return key, assigned.strip()


def _parse_set_assignment(value: str) -> tuple[str, str, Any]:
    left, assigned = _parse_assignment(value, option="--set")
    if "." not in left:
        raise ValueError(f"--set expects ID.KEY=VALUE, got: {value}")
    experiment_id, key = left.split(".", 1)
    if not key:
        raise ValueError(f"--set key cannot be empty: {value}")
    return experiment_id, key, _parse_value(assigned)


def _apply_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted_key}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def _prepare_config(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    experiments = config.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("configuration requires an experiments mapping")

    for assignment in args.set_values:
        experiment_id, key, value = _parse_set_assignment(assignment)
        if experiment_id not in experiments:
            raise KeyError(f"unknown experiment in --set: {experiment_id}")
        top_level_keys = {
            "enabled",
            "model",
            "dataset",
            "backend",
            "run_dir",
            "checkpoint",
            "model_config",
            "output_subdir",
            "python_exe",
            "note",
            "view_count",
            "split",
            "cache_root",
            "extra_args",
        }
        if key.split(".", 1)[0] in top_level_keys:
            _apply_nested(experiments[experiment_id], key, value)
        elif key.startswith("params."):
            _apply_nested(experiments[experiment_id], key, value)
        else:
            experiments[experiment_id].setdefault("params", {})[key] = value

    for assignment in args.checkpoint:
        experiment_id, value = _parse_assignment(assignment, option="--checkpoint")
        if experiment_id not in experiments:
            raise KeyError(f"unknown experiment in --checkpoint: {experiment_id}")
        experiments[experiment_id]["checkpoint"] = value
    return config


def _select_experiments(
    experiments: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    for_listing: bool,
) -> list[str]:
    requested_ids = list(dict.fromkeys(args.experiment))
    unknown = [experiment_id for experiment_id in requested_ids if experiment_id not in experiments]
    if unknown:
        raise KeyError(f"unknown experiment ids: {unknown}")

    if requested_ids:
        selected = requested_ids
    else:
        selected = list(experiments)
        if args.model:
            selected = [
                experiment_id
                for experiment_id in selected
                if str(experiments[experiment_id].get("model")) in set(args.model)
            ]
        if args.dataset:
            selected = [
                experiment_id
                for experiment_id in selected
                if str(experiments[experiment_id].get("dataset")) in set(args.dataset)
            ]
        if not for_listing and not args.all and not args.model and not args.dataset:
            raise ValueError(
                "choose experiments with --all, --experiment, --model, or --dataset "
                "(use --list to inspect the registry)"
            )
        if not args.include_disabled and not for_listing:
            selected = [
                experiment_id
                for experiment_id in selected
                if bool(experiments[experiment_id].get("enabled", True))
            ]

    if not selected:
        raise ValueError("no experiments matched the selection")
    return selected


def _resolved_experiment(
    experiment_id: str,
    experiment: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    datasets = config.get("datasets", {})
    dataset_name = str(experiment.get("dataset"))
    dataset_config = datasets.get(dataset_name, {})
    if not isinstance(dataset_config, dict):
        raise ValueError(f"dataset configuration must be a mapping: {dataset_name}")

    params = copy.deepcopy(defaults.get("params", {}))
    params.update(copy.deepcopy(experiment.get("params", {})))
    global_overrides = {
        "device": args.device,
        "eval_grid_size": args.eval_grid_size,
        "chunk_size": args.chunk_size,
        "max_test_cases": args.max_test_cases,
        "save_volumes": args.save_volumes,
        "skip_visualization": args.skip_visualization,
    }
    params.update({key: value for key, value in global_overrides.items() if value is not None})

    run_dir_value = experiment.get("run_dir")
    run_dir = _path(run_dir_value) if run_dir_value else None
    cache_root_value = experiment.get("cache_root", dataset_config.get("cache_root"))
    if not cache_root_value:
        raise ValueError(f"{experiment_id}: cache_root is not configured")
    cache_root = _path(cache_root_value)
    checkpoint = resolve_checkpoint(experiment.get("checkpoint"), run_dir)

    output_root_value = args.output_root or defaults.get("output_root", "runs/output/table4_a_to_b")
    output_root = _path(output_root_value)
    if args.run_tag:
        output_root = output_root / args.run_tag
    checkpoint_tag = checkpoint.stem
    output_template = experiment.get("output_subdir", f"{experiment_id}/{checkpoint_tag}")
    output_subdir = str(output_template).format(
        experiment=experiment_id,
        model=experiment.get("model"),
        dataset=dataset_name,
        checkpoint=checkpoint_tag,
    )
    output_dir = _path(output_subdir, base=output_root)

    resolved = {
        **copy.deepcopy(experiment),
        "id": experiment_id,
        "run_dir": run_dir,
        "cache_root": cache_root,
        "checkpoint_path": checkpoint,
        "view_count": int(experiment.get("view_count", defaults.get("view_count", 5))),
        "split": str(experiment.get("split", defaults.get("split", "test"))),
        "params": params,
        "output_root": output_root,
        "output_dir": output_dir,
        "python_exe": _path(
            args.python_exe
            or experiment.get("python_exe")
            or defaults.get("python_exe")
            or Path(sys.executable)
        ),
    }
    return resolved


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _build_difnet_command(experiment: dict[str, Any], python_exe: Path) -> list[str]:
    params = experiment["params"]
    allowed = {
        "device",
        "eval_grid_size",
        "chunk_size",
        "max_test_cases",
        "save_volumes",
        "skip_visualization",
        "sample_strategy",
        "shard_count",
        "shard_index",
        "combine",
        "mid_channels",
        "routed_capsule_squash",
        "model_version",
        "hashgrid_backend",
        "capsule_hidden_dim",
        "capsule_layers",
        "capsule_heads",
        "capsule_mlp_ratio",
        "capsule_dropout",
        "capsule_valid_mask",
        "comparisons_dir",
        "viewer_dir",
        "visualization_max_cases",
        "visualization_slice_mode",
        "visualization_num_slices",
        "viewer_max_cases",
        "viewer_slice_stride",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{experiment['id']}: unsupported DIF-Net backend parameters: {unknown}")
    if not bool(params.get("save_volumes", False)) and not bool(params.get("skip_visualization", True)):
        raise ValueError(f"{experiment['id']}: visualization requires save_volumes=true")

    command = [
        str(python_exe),
        str(PROJECT_ROOT / "tools" / "test_difnet_v0b.py"),
        "--cache-root",
        str(experiment["cache_root"]),
        "--split",
        experiment["split"],
        "--view-count",
        str(experiment["view_count"]),
        "--checkpoint",
        str(experiment["checkpoint_path"]),
        "--output-dir",
        str(experiment["output_dir"]),
    ]
    scalar_flags = {
        "eval_grid_size": "--eval-grid-size",
        "chunk_size": "--chunk-size",
        "max_test_cases": "--max-test-cases",
        "device": "--device",
        "sample_strategy": "--sample-strategy",
        "shard_count": "--shard-count",
        "shard_index": "--shard-index",
        "combine": "--combine",
        "mid_channels": "--mid-channels",
        "routed_capsule_squash": "--routed-capsule-squash",
        "model_version": "--model-version",
        "hashgrid_backend": "--hashgrid-backend",
        "capsule_hidden_dim": "--capsule-hidden-dim",
        "capsule_layers": "--capsule-layers",
        "capsule_heads": "--capsule-heads",
        "capsule_mlp_ratio": "--capsule-mlp-ratio",
        "capsule_dropout": "--capsule-dropout",
        "comparisons_dir": "--comparisons-dir",
        "viewer_dir": "--viewer-dir",
        "visualization_max_cases": "--visualization-max-cases",
        "visualization_slice_mode": "--visualization-slice-mode",
        "visualization_num_slices": "--visualization-num-slices",
        "viewer_max_cases": "--viewer-max-cases",
        "viewer_slice_stride": "--viewer-slice-stride",
    }
    for key, flag in scalar_flags.items():
        value = params.get(key)
        if key in {"comparisons_dir", "viewer_dir"} and value is not None:
            value = _path(value)
        _append_option(command, flag, value)
    command.append("--save-volumes" if bool(params.get("save_volumes", False)) else "--no-save-volumes")
    if bool(params.get("skip_visualization", True)):
        command.append("--skip-visualization")
    capsule_valid_mask = params.get("capsule_valid_mask")
    if capsule_valid_mask is not None:
        command.append("--capsule-valid-mask" if bool(capsule_valid_mask) else "--no-capsule-valid-mask")
    command.extend(str(value) for value in experiment.get("extra_args", []))
    return command


def _build_deepsparse_command(experiment: dict[str, Any], python_exe: Path) -> list[str]:
    params = experiment["params"]
    allowed = {
        "device",
        "eval_grid_size",
        "chunk_size",
        "max_test_cases",
        "save_volumes",
        "skip_visualization",
        "sample_strategy",
        "shard_count",
        "shard_index",
        "mixed_precision",
        "eval_indices",
        "lr_res",
        "projection_multiplier",
        "seed",
        "official_config",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{experiment['id']}: unsupported DeepSparse backend parameters: {unknown}")

    command = [
        str(python_exe),
        str(PROJECT_ROOT / "tools" / "test_deepsparse.py"),
        "--cache-root",
        str(experiment["cache_root"]),
        "--split",
        experiment["split"],
        "--view-count",
        str(experiment["view_count"]),
        "--checkpoint",
        str(experiment["checkpoint_path"]),
        "--output-dir",
        str(experiment["output_dir"]),
    ]
    if experiment.get("model_config"):
        command.extend(["--config", str(_path(experiment["model_config"]))])
    scalar_flags = {
        "eval_grid_size": "--eval-grid-size",
        "chunk_size": "--chunk-size",
        "max_test_cases": "--max-test-cases",
        "device": "--device",
        "lr_res": "--lr-res",
        "projection_multiplier": "--projection-multiplier",
        "seed": "--seed",
        "official_config": "--official-config",
    }
    for key, flag in scalar_flags.items():
        value = params.get(key)
        if key == "official_config" and value is not None:
            value = _path(value)
        _append_option(command, flag, value)
    if params.get("eval_indices") is not None:
        command.append("--eval-indices")
        command.extend(str(index) for index in params["eval_indices"])
    command.append("--save-volumes" if bool(params.get("save_volumes", False)) else "--no-save-volumes")
    mixed_precision = params.get("mixed_precision")
    if mixed_precision is not None:
        command.append("--mixed-precision" if bool(mixed_precision) else "--no-mixed-precision")
    command.extend(str(value) for value in experiment.get("extra_args", []))
    return command


def build_command(experiment: dict[str, Any], python_exe: Path) -> list[str]:
    backend = str(experiment.get("backend"))
    if backend == "difnet":
        return _build_difnet_command(experiment, python_exe)
    if backend == "deepsparse":
        return _build_deepsparse_command(experiment, python_exe)
    raise ValueError(f"{experiment['id']}: unsupported backend: {backend}")


def _display_command(command: list[str]) -> str:
    return shlex.join(command)


def _list_registry(config: dict[str, Any], args: argparse.Namespace) -> int:
    experiments = config["experiments"]
    selected_ids = _select_experiments(experiments, args, for_listing=True)
    print("ID\tENABLED\tMODEL\tDATASET\tCHECKPOINT\tNOTE")
    for experiment_id in selected_ids:
        experiment = experiments[experiment_id]
        run_dir = _path(experiment["run_dir"]) if experiment.get("run_dir") else None
        try:
            checkpoint = resolve_checkpoint(experiment.get("checkpoint"), run_dir)
            checkpoint_text = str(checkpoint.relative_to(PROJECT_ROOT))
        except (FileNotFoundError, ValueError) as exc:
            checkpoint_text = f"MISSING ({exc})"
        print(
            "\t".join(
                [
                    experiment_id,
                    str(bool(experiment.get("enabled", True))).lower(),
                    str(experiment.get("model")),
                    str(experiment.get("dataset")),
                    checkpoint_text,
                    str(experiment.get("note", "")),
                ]
            )
        )
        if args.list_checkpoints:
            candidates = discover_checkpoints(run_dir)
            for candidate in candidates:
                epoch_text = _epoch(candidate)
                print(
                    f"  - epoch={epoch_text if epoch_text is not None else '-'} "
                    f"{candidate.relative_to(PROJECT_ROOT)}"
                )
            if not candidates:
                print("  - no checkpoint candidates")
    return 0


def _load_metrics(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "metrics_summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _run_command(command: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        stdout_available = True
        for line in process.stdout:
            if stdout_available:
                try:
                    print(line, end="", flush=True)
                except (BrokenPipeError, OSError):
                    stdout_available = False
            log.write(line)
            log.flush()
        return process.wait()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Table 4 A-to-b evaluations: noaug-trained checkpoints on aug3x3 test caches."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list", action="store_true", help="List registry entries without running them.")
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List checkpoint candidates for the selected registry entries.",
    )
    parser.add_argument("--all", action="store_true", help="Select every enabled experiment.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--experiment", action="append", default=[], help="Select an exact experiment id; repeatable.")
    parser.add_argument("--model", action="append", default=[], help="Filter by model name; repeatable.")
    parser.add_argument("--dataset", action="append", default=[], help="Filter by dataset name; repeatable.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="ID=SPEC",
        help="Override a checkpoint with latest, best, epoch:N, or an explicit path; repeatable.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="ID.KEY=VALUE",
        help="Override one experiment parameter using a JSON scalar/list value; repeatable.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-tag", help="Add a subdirectory under output-root for this batch.")
    parser.add_argument(
        "--python-exe",
        type=Path,
        help="Override the per-experiment Python executable for every selected entry.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--eval-grid-size", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--max-test-cases", type=int)
    volume_group = parser.add_mutually_exclusive_group()
    volume_group.add_argument("--save-volumes", dest="save_volumes", action="store_true")
    volume_group.add_argument("--no-save-volumes", dest="save_volumes", action="store_false")
    parser.set_defaults(save_volumes=None)
    visualization_group = parser.add_mutually_exclusive_group()
    visualization_group.add_argument("--skip-visualization", dest="skip_visualization", action="store_true")
    visualization_group.add_argument("--visualize", dest="skip_visualization", action="store_false")
    parser.set_defaults(skip_visualization=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        config_path = _path(args.config)
        config = _prepare_config(_read_config(config_path), args)
        if args.list or args.list_checkpoints:
            return _list_registry(config, args)

        experiments = config["experiments"]
        selected_ids = _select_experiments(experiments, args, for_listing=False)
        plan: list[dict[str, Any]] = []
        errors: list[str] = []
        for experiment_id in selected_ids:
            try:
                resolved = _resolved_experiment(experiment_id, experiments[experiment_id], config, args)
                if not resolved["cache_root"].is_dir():
                    raise FileNotFoundError(f"cache root not found: {resolved['cache_root']}")
                if not resolved["python_exe"].is_file():
                    raise FileNotFoundError(f"python executable not found: {resolved['python_exe']}")
                output_nonempty = resolved["output_dir"].is_dir() and any(resolved["output_dir"].iterdir())
                command = build_command(resolved, resolved["python_exe"])
                plan.append(
                    {
                        "experiment": resolved,
                        "command": command,
                        "output_nonempty": output_nonempty,
                    }
                )
                if output_nonempty and not args.skip_existing:
                    errors.append(
                        f"{experiment_id}: output directory already exists and is non-empty: "
                        f"{resolved['output_dir']}"
                    )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{experiment_id}: {exc}")

        print(f"Selected {len(selected_ids)} experiment(s).")
        for item in plan:
            experiment = item["experiment"]
            status = "EXISTS" if item["output_nonempty"] else "READY"
            print(
                f"[{status}] {experiment['id']} model={experiment['model']} "
                f"dataset={experiment['dataset']} checkpoint={experiment['checkpoint_path']}"
            )
            print(f"  output: {experiment['output_dir']}")
            print(f"  command: {_display_command(item['command'])}")
        if errors:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))
        if args.dry_run:
            print("Dry run complete; no output files were written and no evaluation process was started.")
            return 0

        output_roots = {item["experiment"]["output_root"] for item in plan}
        if len(output_roots) != 1:
            raise ValueError(f"all selected experiments must share one output root, got: {output_roots}")
        batch_root = next(iter(output_roots))
        batch_root.mkdir(parents=True, exist_ok=True)
        serializable_plan = {
            "schema": "x2cbct.table4_a_to_b.batch_plan.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "experiments": [
                {
                    "id": item["experiment"]["id"],
                    "model": item["experiment"]["model"],
                    "dataset": item["experiment"]["dataset"],
                    "backend": item["experiment"]["backend"],
                    "cache_root": str(item["experiment"]["cache_root"]),
                    "checkpoint": str(item["experiment"]["checkpoint_path"]),
                    "python_exe": str(item["experiment"]["python_exe"]),
                    "output_dir": str(item["experiment"]["output_dir"]),
                    "params": item["experiment"]["params"],
                    "command": item["command"],
                }
                for item in plan
            ],
        }
        _write_json(batch_root / "batch_plan.json", serializable_plan)

        results: list[dict[str, Any]] = []
        for item in plan:
            experiment = item["experiment"]
            experiment_id = experiment["id"]
            if item["output_nonempty"] and args.skip_existing:
                metrics = _load_metrics(experiment["output_dir"])
                results.append(
                    {
                        "id": experiment_id,
                        "model": experiment["model"],
                        "dataset": experiment["dataset"],
                        "status": "skipped_existing",
                        "returncode": None,
                        "output_dir": str(experiment["output_dir"]),
                        "metrics": metrics,
                    }
                )
                continue

            print(f"\n=== Running {experiment_id} ===", flush=True)
            log_path = batch_root / "_logs" / f"{experiment_id}-{experiment['checkpoint_path'].stem}.log"
            returncode = _run_command(item["command"], log_path=log_path)
            metrics = _load_metrics(experiment["output_dir"])
            status = "completed" if returncode == 0 and metrics is not None else "failed"
            results.append(
                {
                    "id": experiment_id,
                    "model": experiment["model"],
                    "dataset": experiment["dataset"],
                    "status": status,
                    "returncode": returncode,
                    "output_dir": str(experiment["output_dir"]),
                    "log_path": str(log_path),
                    "metrics": metrics,
                }
            )
            _write_json(
                batch_root / "batch_summary.json",
                {
                    "schema": "x2cbct.table4_a_to_b.batch_summary.v1",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "results": results,
                },
            )
            if status == "failed" and not args.continue_on_error:
                print(f"{experiment_id} failed; stopping the batch.", file=sys.stderr)
                return returncode if returncode != 0 else 1

        failures = [result for result in results if result["status"] == "failed"]
        print(f"Batch complete: {len(results) - len(failures)} succeeded/skipped, {len(failures)} failed.")
        return 1 if failures else 0
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
