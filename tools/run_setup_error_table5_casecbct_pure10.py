from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_setup_error_table5_casecbct_pure10 import (  # noqa: E402
    DEFAULT_CONFIG,
    prepare,
)


BATCH_RUNNER = TOOLS_ROOT / "run_setup_error_table5_models.py"
FIXED_BATCH_OPTIONS = {
    "--output-dir",
    "--model-config",
    "--cache-root",
    "--formal-registration-results",
    "--roi-reviewed",
    "--si-pair",
    "--yaw-pair",
    "--view-count",
    "--eval-grid-size",
    "--require-inference-split",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def _blocked_forwarded_option(values: list[str]) -> str | None:
    for value in values:
        option = value.split("=", 1)[0]
        if option in FIXED_BATCH_OPTIONS:
            return option
    return None


def _parse_known_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "One-click five-model Table-5 experiment on two high-quality "
            "CaSE-CBCT test cases with pure +10 mm S-I / +10 degree yaw."
        )
    )
    parser.add_argument("--case-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-source-baseline", action="store_true")
    parser.add_argument("--registration-itk-threads", type=int, default=4)
    return parser.parse_known_args()


def main() -> int:
    args, forwarded = _parse_known_args()
    try:
        blocked = _blocked_forwarded_option(forwarded)
        if blocked is not None:
            raise ValueError(
                f"{blocked} is fixed by --case-config for this controlled experiment"
            )
        config_path = _project_path(args.case_config)
        config = _read_config(config_path)
        prepared = prepare(
            config_path,
            force=bool(args.rebuild_source_baseline),
            itk_threads=int(args.registration_itk_threads),
        )
        print(
            f"[casecbct:baseline] status={prepared['status']} "
            f"output={prepared['output_root']}",
            flush=True,
        )
        if args.prepare_only:
            print(json.dumps(prepared, ensure_ascii=False, indent=2))
            return 0

        experiments = {
            str(item["axis"]): item for item in config["experiments"]
        }
        si = experiments["si-max"]
        yaw = experiments["yaw-max"]
        command = [
            sys.executable,
            str(BATCH_RUNNER),
            "--model-config",
            str(_project_path(config["model_config"])),
            "--output-dir",
            str(_project_path(config["batch_output"])),
            "--cache-root",
            str(_project_path(config["identity_cache_root"])),
            "--cache-root",
            str(_project_path(config["augmented_cache_root"])),
            "--formal-registration-results",
            str(
                _project_path(config["source_registration_output"])
                / "registration_results.csv"
            ),
            "--roi-reviewed",
            str(_project_path(config["roi_reviewed"])),
            "--si-pair",
            (
                f"{si['patient_id']}:{si['case_id']}:{si['case_id']}:"
                f"{si['reference_augmentation_id']}:"
                f"{si['moving_augmentation_id']}"
            ),
            "--yaw-pair",
            (
                f"{yaw['patient_id']}:{yaw['case_id']}:{yaw['case_id']}:"
                f"{yaw['reference_augmentation_id']}:"
                f"{yaw['moving_augmentation_id']}"
            ),
            "--view-count",
            str(config["view_count"]),
            "--eval-grid-size",
            str(config["eval_grid_size"]),
            "--require-inference-split",
            "test",
            *forwarded,
        ]
        print(
            "[casecbct:batch] " + subprocess.list2cmdline(command),
            flush=True,
        )
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except Exception as exc:
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
