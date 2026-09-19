from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _case_limit_arg(value: int | None) -> list[str]:
    if value is None or value <= 0:
        return []
    return ["--max-cases", str(value)]


def _ensure_clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output dir already exists and is not empty: {path}")


def _tail_text(path: Path, max_lines: int = 24) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _launch_shards(args: argparse.Namespace, forwarded: list[str]) -> list[Path]:
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen[str], Any, Any, Path]] = []
    shard_dirs: list[Path] = []
    for index in range(args.parallel_workers):
        shard_dir = shard_root / f"shard_{index:02d}"
        _ensure_clean_output_dir(shard_dir)
        shard_dirs.append(shard_dir)
        stdout_handle = (log_dir / f"shard_{index:02d}.stdout.log").open("w", encoding="utf-8")
        stderr_handle = (log_dir / f"shard_{index:02d}.stderr.log").open("w", encoding="utf-8")
        cmd = [
            str(args.python_exe),
            str(PROJECT_ROOT / "tools" / "test_difnet_v0b.py"),
            *forwarded,
            "--output-dir",
            str(shard_dir),
            "--shard-count",
            str(args.parallel_workers),
            "--shard-index",
            str(index),
            "--skip-visualization",
        ]
        processes.append(
            (
                index,
                subprocess.Popen(cmd, cwd=PROJECT_ROOT, text=True, stdout=stdout_handle, stderr=stderr_handle),
                stdout_handle,
                stderr_handle,
                shard_dir,
            )
        )

    failed: list[str] = []
    try:
        for index, process, stdout_handle, stderr_handle, shard_dir in processes:
            code = process.wait()
            stdout_handle.close()
            stderr_handle.close()
            if code != 0:
                stderr_path = args.output_dir / "command_logs" / f"shard_{index:02d}.stderr.log"
                failed.append(
                    f"shard_{index:02d} exit_code={code} dir={shard_dir}\n"
                    f"stderr_tail:\n{_tail_text(stderr_path)}"
                )
    finally:
        for _, process, stdout_handle, stderr_handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            if not stdout_handle.closed:
                stdout_handle.close()
            if not stderr_handle.closed:
                stderr_handle.close()
    if failed:
        raise RuntimeError("parallel test shard failed: " + "; ".join(failed))
    return shard_dirs


def _read_metrics_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _mean_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("mae_norm", "mae_hu", "psnr", "ssim"):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def _merge_outputs(args: argparse.Namespace, shard_dirs: list[Path]) -> Path:
    manifests = [_read_json(path / "prediction_manifest.json") for path in shard_dirs]
    summaries = [_read_json(path / "metrics_summary.json") for path in shard_dirs]
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for shard_dir in shard_dirs:
        names, rows = _read_metrics_rows(shard_dir / "metrics_per_case.csv")
        fieldnames = fieldnames or names
        all_rows.extend(rows)

    predictions = []
    for manifest in manifests:
        predictions.extend(manifest.get("predictions", []))

    base_summary = dict(summaries[0])
    base_summary.update(_mean_metrics(all_rows))
    base_summary.update(
        {
            "case_count": len(all_rows),
            "parallel_workers": args.parallel_workers,
            "shard_dirs": [str(path) for path in shard_dirs],
        }
    )
    base_summary.pop("shard_count", None)
    base_summary.pop("shard_index", None)

    base_manifest = dict(manifests[0])
    base_manifest.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "parallel_workers": args.parallel_workers,
            "shard_dirs": [str(path) for path in shard_dirs],
            "predictions": predictions,
        }
    )
    base_manifest.pop("shard_count", None)
    base_manifest.pop("shard_index", None)

    _write_json(base_summary, args.output_dir / "metrics_summary.json")
    _write_json(base_manifest, args.output_dir / "prediction_manifest.json")
    with (args.output_dir / "metrics_per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(all_rows)
    return args.output_dir / "prediction_manifest.json"


def _run_artifact_command(cmd: list[str], label: str) -> None:
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} export failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr, flush=True)


def _export_visual_artifacts(args: argparse.Namespace, manifest_path: Path) -> dict[str, str]:
    comparisons_dir = args.comparisons_dir or args.output_dir / "comparisons"
    viewer_dir = args.viewer_dir or args.output_dir / "viewer"
    visualize_cmd = [
        str(args.python_exe),
        str(PROJECT_ROOT / "tools" / "visualize_difnet_v0b_predictions.py"),
        "--prediction-manifest",
        str(manifest_path),
        "--output-dir",
        str(comparisons_dir),
        "--slice-mode",
        args.visualization_slice_mode,
        "--num-slices",
        str(args.visualization_num_slices),
    ]
    visualize_cmd.extend(_case_limit_arg(args.visualization_max_cases))
    _run_artifact_command(visualize_cmd, "comparison")

    viewer_cmd = [
        str(args.python_exe),
        str(PROJECT_ROOT / "tools" / "view_difnet_v0b_prediction.py"),
        "--prediction-manifest",
        str(manifest_path),
        "--output-dir",
        str(viewer_dir),
        "--slice-stride",
        str(args.viewer_slice_stride),
    ]
    viewer_cmd.extend(_case_limit_arg(args.viewer_max_cases))
    _run_artifact_command(viewer_cmd, "viewer")
    return {
        "comparisons_dir": str(comparisons_dir),
        "viewer_dir": str(viewer_dir),
        "viewer_html": str(viewer_dir / "index.html"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DIF-Net V0b test inference in multiple shard processes.",
        allow_abbrev=False,
    )
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--comparisons-dir", type=Path)
    parser.add_argument("--viewer-dir", type=Path)
    parser.add_argument("--visualization-max-cases", type=int, default=50)
    parser.add_argument("--visualization-slice-mode", choices=("axial", "coronal", "sagittal", "all"), default="all")
    parser.add_argument("--visualization-num-slices", type=int, default=1)
    parser.add_argument("--viewer-max-cases", type=int, default=50)
    parser.add_argument("--viewer-slice-stride", type=int, default=4)
    args, forwarded = parser.parse_known_args()

    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be >= 1")
    _ensure_clean_output_dir(args.output_dir)
    shard_dirs = _launch_shards(args, forwarded)
    manifest_path = _merge_outputs(args, shard_dirs)
    visual_artifacts = {} if args.skip_visualization else _export_visual_artifacts(args, manifest_path)
    summary = _read_json(args.output_dir / "metrics_summary.json")
    print(
        json.dumps(
            {"ok": True, "output_dir": str(args.output_dir), "metrics_summary": summary, **visual_artifacts},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
