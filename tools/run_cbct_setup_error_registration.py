from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_error_manifest import read_pair_manifest
from x2cbct_difnet.setup_error_registration import (
    load_roi_review,
    register_pair,
)


def patient_summary(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(str(row["patient_id"]), []).append(row)
    summary = []
    for patient_id, rows in sorted(grouped.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0])):
        ok = [r for r in rows if r["status"] == "ok"]
        si_values = [float(r["setup_error_SI_mm"]) for r in ok]
        abs_si_values = [float(r["setup_error_abs_SI_mm"]) for r in ok]
        yaw_values = [float(r["setup_error_yaw_deg"]) for r in ok]
        abs_yaw_values = [float(r["setup_error_abs_yaw_deg"]) for r in ok]
        summary.append(
            {
                "patient_id": patient_id,
                "pair_count": len(rows),
                "ok_count": len(ok),
                "setup_error_SI_mean_mm": mean(si_values),
                "setup_error_abs_SI_mean_mm": mean(abs_si_values),
                "setup_error_abs_SI_max_mm": max(abs_si_values) if abs_si_values else 0.0,
                "setup_error_yaw_mean_deg": mean(yaw_values),
                "setup_error_abs_yaw_mean_deg": mean(abs_yaw_values),
                "setup_error_abs_yaw_max_deg": max(abs_yaw_values) if abs_yaw_values else 0.0,
            }
        )
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_result_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add((str(row["patient_id"]), str(row["reference_case_id"]), str(row["moving_case_id"])))
    return keys


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "registration_results.csv",
        output_dir / "registration_initializations.csv",
        output_dir / "registration_results.json",
        output_dir / "patient_summary.csv",
        output_dir / "failed_cases.csv",
        output_dir / "dataset_summary.json",
        output_dir / "progress.json",
    ]


def clear_outputs_for_fresh_run(output_dir: Path, *, resume: bool) -> None:
    if resume:
        return
    for path in output_files(output_dir):
        if path.exists():
            path.unlink()


def failed_result_row(pair: dict, status: str, note: str) -> dict:
    return {
        "patient_id": str(pair["patient_id"]),
        "reference_case_id": str(pair["reference_case_id"]),
        "moving_case_id": str(pair["moving_case_id"]),
        "status": status,
        "LR_mm": 0.0,
        "AP_mm": 0.0,
        "SI_mm": 0.0,
        "translation_norm_mm": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "yaw_deg": 0.0,
        "rotation_norm_deg": 0.0,
        "setup_error_SI_mm": 0.0,
        "setup_error_abs_SI_mm": 0.0,
        "setup_error_yaw_deg": 0.0,
        "setup_error_abs_yaw_deg": 0.0,
        "similarity_score": float("nan"),
        "selected_initialization_id": "",
        "roi_status": "",
        "notes": note,
    }


def register_pair_worker(payload: tuple[int, int, dict, dict, int]) -> dict:
    index, total, pair, roi_by_case, itk_threads = payload
    try:
        if itk_threads > 0:
            import SimpleITK as sitk

            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(itk_threads))
        started = time.time()
        result, initializations = register_pair(pair, roi_by_case)
        result_row = asdict(result)
        init_rows = [
            {
                "patient_id": pair["patient_id"],
                "reference_case_id": pair["reference_case_id"],
                "moving_case_id": pair["moving_case_id"],
                **asdict(init),
            }
            for init in initializations
        ]
        return {
            "index": index,
            "total": total,
            "pair": pair,
            "result": result_row,
            "initializations": init_rows,
            "seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "index": index,
            "total": total,
            "pair": pair,
            "result": failed_result_row(pair, "failed", f"worker exception: {exc}"),
            "initializations": [],
            "seconds": 0.0,
        }


def write_pair_outputs(
    *,
    output_dir: Path,
    progress_path: Path,
    result_csv: Path,
    init_csv: Path,
    failed_csv: Path,
    payload: dict,
    completed_count: int,
    total: int,
    started_at: float,
) -> None:
    pair = payload["pair"]
    result_row = payload["result"]
    append_csv_row(result_csv, result_row)
    if result_row["status"] != "ok":
        append_csv_row(failed_csv, result_row)
    for init_row in payload["initializations"]:
        append_csv_row(init_csv, init_row)
    elapsed = time.time() - started_at
    write_json(
        progress_path,
        {
            "schema": "x2cbct_setup_error_registration_progress.v1",
            "total_pairs": total,
            "completed_pairs": completed_count,
            "last_completed_index": payload["index"],
            "last_completed_pair": {
                "patient_id": str(pair["patient_id"]),
                "reference_case_id": str(pair["reference_case_id"]),
                "moving_case_id": str(pair["moving_case_id"]),
            },
            "last_status": result_row["status"],
            "last_pair_seconds": payload["seconds"],
            "elapsed_seconds": elapsed,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CBCT cervical-ROI setup-error rigid registration.")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--roi-reviewed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration")
    parser.add_argument("--patient-id")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip pairs already present in registration_results.csv.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel pair registrations. Start with 2 on this dataset.")
    parser.add_argument("--itk-threads", type=int, default=0, help="SimpleITK threads per worker. 0 keeps the SimpleITK default.")
    args = parser.parse_args()

    if not args.roi_reviewed.exists():
        candidate = args.roi_reviewed.parent / "roi_candidates.csv"
        message = [
            f"ROI review file not found: {args.roi_reviewed}",
            "人工审查页面里的自动保存默认只保存在浏览器本地缓存。",
            "请重新启动 reviewer，在页面底部点击“保存到项目文件”，或点击“下载 roi_reviewed.csv”后放到上述路径。",
        ]
        if candidate.exists():
            message.append(f"当前目录存在候选文件，可先打开 reviewer 审查: {candidate}")
            message.append(
                "启动命令: .\\.conda\\x2cbct-v0\\python.exe tools\\serve_cervical_roi_reviewer.py "
                f"--roi-candidates {candidate}"
            )
        raise SystemExit("\n".join(message))

    manifest = read_pair_manifest(args.pair_manifest)
    roi_by_case = load_roi_review(args.roi_reviewed)
    pairs = manifest["pairs"]
    if args.patient_id is not None:
        pairs = [p for p in pairs if str(p["patient_id"]) == str(args.patient_id)]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.output_dir / "registration_results.csv"
    init_csv = args.output_dir / "registration_initializations.csv"
    result_json = args.output_dir / "registration_results.json"
    summary_csv = args.output_dir / "patient_summary.csv"
    failed_csv = args.output_dir / "failed_cases.csv"
    dataset_summary_path = args.output_dir / "dataset_summary.json"
    progress_path = args.output_dir / "progress.json"
    clear_outputs_for_fresh_run(args.output_dir, resume=args.resume)

    completed = load_existing_result_keys(result_csv) if args.resume else set()
    pending: list[tuple[int, dict]] = []
    for index, pair in enumerate(pairs, start=1):
        key = (str(pair["patient_id"]), str(pair["reference_case_id"]), str(pair["moving_case_id"]))
        if key in completed:
            print(f"[{index}/{len(pairs)}] skip existing patient={key[0]} {key[1]}->{key[2]}", flush=True)
        else:
            pending.append((index, pair))

    started_at = time.time()
    total = len(pairs)
    worker_count = max(1, int(args.workers))
    print(
        f"starting registration: total={total}, pending={len(pending)}, workers={worker_count}, itk_threads={args.itk_threads}",
        flush=True,
    )
    completed_count = len(completed)
    if worker_count == 1:
        for index, pair in pending:
            key = (str(pair["patient_id"]), str(pair["reference_case_id"]), str(pair["moving_case_id"]))
            print(f"[{index}/{total}] registering patient={key[0]} {key[1]}->{key[2]}", flush=True)
            payload = register_pair_worker((index, total, pair, roi_by_case, int(args.itk_threads)))
            completed_count += 1
            write_pair_outputs(
                output_dir=args.output_dir,
                progress_path=progress_path,
                result_csv=result_csv,
                init_csv=init_csv,
                failed_csv=failed_csv,
                payload=payload,
                completed_count=completed_count,
                total=total,
                started_at=started_at,
            )
            print(
                f"[{index}/{total}] done status={payload['result']['status']} seconds={payload['seconds']:.1f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(register_pair_worker, (index, total, pair, roi_by_case, int(args.itk_threads)))
                for index, pair in pending
            ]
            for future in as_completed(futures):
                payload = future.result()
                pair = payload["pair"]
                completed_count += 1
                write_pair_outputs(
                    output_dir=args.output_dir,
                    progress_path=progress_path,
                    result_csv=result_csv,
                    init_csv=init_csv,
                    failed_csv=failed_csv,
                    payload=payload,
                    completed_count=completed_count,
                    total=total,
                    started_at=started_at,
                )
                print(
                    f"[{payload['index']}/{total}] done patient={pair['patient_id']} "
                    f"{pair['reference_case_id']}->{pair['moving_case_id']} "
                    f"status={payload['result']['status']} seconds={payload['seconds']:.1f} "
                    f"completed={completed_count}/{total}",
                    flush=True,
                )

    result_rows = []
    if result_csv.exists() and result_csv.stat().st_size > 0:
        with result_csv.open("r", encoding="utf-8", newline="") as handle:
            result_rows = list(csv.DictReader(handle))
    summary_rows = patient_summary(result_rows)
    write_csv(summary_rows, summary_csv)
    if result_rows:
        write_json(
            result_json,
            {
                "schema": "x2cbct_setup_error_si_yaw_registration.v2",
                "primary_metrics": ["setup_error_abs_SI_mm", "setup_error_abs_yaw_deg"],
                "results": result_rows,
            },
        )
    dataset_summary = {
        "schema": "x2cbct_setup_error_si_yaw_dataset_summary.v1",
        "primary_metrics": ["setup_error_abs_SI_mm", "setup_error_abs_yaw_deg"],
        "pair_count": len(result_rows),
        "ok_count": sum(1 for r in result_rows if r["status"] == "ok"),
        "failed_count": sum(1 for r in result_rows if r["status"] == "failed"),
        "skipped_count": sum(1 for r in result_rows if r["status"] == "skipped"),
        "abs_SI_exceeds_2mm_count": sum(1 for r in result_rows if r["status"] == "ok" and float(r["setup_error_abs_SI_mm"]) > 2.0),
        "abs_SI_exceeds_3mm_count": sum(1 for r in result_rows if r["status"] == "ok" and float(r["setup_error_abs_SI_mm"]) > 3.0),
        "abs_yaw_exceeds_2deg_count": sum(1 for r in result_rows if r["status"] == "ok" and float(r["setup_error_abs_yaw_deg"]) > 2.0),
        "abs_yaw_exceeds_5deg_count": sum(1 for r in result_rows if r["status"] == "ok" and float(r["setup_error_abs_yaw_deg"]) > 5.0),
        "patient_summary_csv": str(summary_csv),
        "failed_cases_csv": str(failed_csv),
        "progress_json": str(progress_path),
    }
    write_json(dataset_summary_path, dataset_summary)
    print(
        json.dumps(
            {
                "ok": True,
                "results_csv": str(result_csv),
                "initializations_csv": str(init_csv),
                "results_json": str(result_json),
                "dataset_summary": str(dataset_summary_path),
                "progress": str(progress_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
