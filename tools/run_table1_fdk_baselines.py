from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW_COUNTS = (2, 5, 8, 10)


@dataclass(frozen=True)
class FdkTask:
    dataset: str
    cache_root: Path
    view_count: int
    output_name: str


def _parse_view_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not counts:
        raise argparse.ArgumentTypeError("expected at least one view count")
    return tuple(dict.fromkeys(counts))


def table1_fdk_tasks(view_counts: tuple[int, ...] = DEFAULT_VIEW_COUNTS) -> list[FdkTask]:
    cache_root = PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256"
    tasks: list[FdkTask] = []
    for view_count in view_counts:
        tasks.append(
            FdkTask(
                dataset="luna16",
                cache_root=cache_root / "luna16" / "test_noaug_256" / "luna16",
                view_count=view_count,
                output_name=f"luna16_view{view_count}",
            )
        )
        tasks.append(
            FdkTask(
                dataset="self_projection1208_mixed",
                cache_root=cache_root / "self_projection1208" / "test_mixed_noaug_256",
                view_count=view_count,
                output_name=f"self_projection1208_mixed_view{view_count}",
            )
        )
    return tasks


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _safe_rmtree(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    target = path.resolve()
    if not str(target).lower().startswith(str(root).lower()):
        raise RuntimeError(f"refusing to remove outside workspace: {target}")
    if target in {root, root / "runs", root / "runs" / "fdk"}:
        raise RuntimeError(f"refusing to remove broad workspace path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _validate_inputs(tasks: list[FdkTask]) -> None:
    missing = [str(task.cache_root) for task in tasks if not (task.cache_root / "cache_manifest.json").exists()]
    if missing:
        raise FileNotFoundError("missing cache_manifest.json for FDK tasks: " + "; ".join(missing))


def _prepare_outputs(tasks: list[FdkTask], output_root: Path, *, overwrite: bool) -> None:
    conflicts = []
    for task in tasks:
        out_dir = output_root / task.output_name
        if out_dir.exists() and any(out_dir.iterdir()):
            conflicts.append(str(out_dir))
    for summary_name in ("summary.csv", "summary.json"):
        summary_path = output_root / summary_name
        if summary_path.exists():
            conflicts.append(str(summary_path))
    if conflicts and not overwrite:
        raise FileExistsError("FDK output already exists; rerun with --overwrite: " + "; ".join(conflicts))
    if overwrite:
        for task in tasks:
            _safe_rmtree(output_root / task.output_name)
        for summary_name in ("summary.csv", "summary.json"):
            summary_path = output_root / summary_name
            if summary_path.exists():
                summary_path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _run_task(task: FdkTask, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    out_dir = output_root / task.output_name
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "test_fdk_baseline.py"),
        "--cache-root",
        str(task.cache_root),
        "--split",
        "test",
        "--view-count",
        str(task.view_count),
        "--output-dir",
        str(out_dir),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print(
        json.dumps(
            {
                "event": "fdk_batch_task_start",
                "dataset": task.dataset,
                "view_count": task.view_count,
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FDK task failed: {task.output_name}")
    summary = json.loads((out_dir / "metrics_summary.json").read_text(encoding="utf-8-sig"))
    row = {
        "dataset": task.dataset,
        "view_count": task.view_count,
        "cache_root": str(task.cache_root),
        "output_dir": str(out_dir),
        "case_count": summary.get("case_count"),
        "mae_norm": summary.get("mae_norm"),
        "mae_hu": summary.get("mae_hu"),
        "psnr": summary.get("psnr"),
        "ssim": summary.get("ssim"),
    }
    print(json.dumps({"event": "fdk_batch_task_done", **row}, ensure_ascii=False), flush=True)
    return row


def _write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    summary = {
        "schema": "x2cbct_table1_fdk_batch.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "tasks": rows,
    }
    _write_json(summary, output_root / "summary.json")
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "view_count",
                "cache_root",
                "output_dir",
                "case_count",
                "mae_norm",
                "mae_hu",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Table 1 FDK noaug baseline matrix.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "fdk")
    parser.add_argument("--view-counts", type=_parse_view_counts, default=DEFAULT_VIEW_COUNTS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = table1_fdk_tasks(args.view_counts)
    _validate_inputs(tasks)
    _prepare_outputs(tasks, args.output_root, overwrite=args.overwrite)
    rows = [_run_task(task, args.output_root, overwrite=args.overwrite) for task in tasks]
    _write_summary(rows, args.output_root)
    print(json.dumps({"ok": True, "output_root": str(args.output_root), "task_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
