from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def enrich_result(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    si = float(updated.get("setup_error_SI_mm") or updated["SI_mm"])
    yaw = float(updated.get("setup_error_yaw_deg") or updated["yaw_deg"])
    updated["setup_error_SI_mm"] = str(si)
    updated["setup_error_abs_SI_mm"] = str(abs(si))
    updated["setup_error_yaw_deg"] = str(yaw)
    updated["setup_error_abs_yaw_deg"] = str(abs(yaw))
    return updated


def patient_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]), []).append(row)
    summary = []
    for patient_id, patient_rows in sorted(grouped.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0])):
        ok = [r for r in patient_rows if r["status"] == "ok"]
        si = [float(r["setup_error_SI_mm"]) for r in ok]
        abs_si = [float(r["setup_error_abs_SI_mm"]) for r in ok]
        yaw = [float(r["setup_error_yaw_deg"]) for r in ok]
        abs_yaw = [float(r["setup_error_abs_yaw_deg"]) for r in ok]
        summary.append(
            {
                "patient_id": patient_id,
                "pair_count": len(patient_rows),
                "ok_count": len(ok),
                "setup_error_SI_mean_mm": mean(si),
                "setup_error_abs_SI_mean_mm": mean(abs_si),
                "setup_error_abs_SI_max_mm": max(abs_si) if abs_si else 0.0,
                "setup_error_yaw_mean_deg": mean(yaw),
                "setup_error_abs_yaw_mean_deg": mean(abs_yaw),
                "setup_error_abs_yaw_max_deg": max(abs_yaw) if abs_yaw else 0.0,
            }
        )
    return summary


def dataset_summary(rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    ok = [r for r in rows if r["status"] == "ok"]
    return {
        "schema": "x2cbct_setup_error_si_yaw_dataset_summary.v1",
        "primary_metrics": ["setup_error_abs_SI_mm", "setup_error_abs_yaw_deg"],
        "pair_count": len(rows),
        "ok_count": len(ok),
        "failed_count": sum(1 for r in rows if r["status"] == "failed"),
        "skipped_count": sum(1 for r in rows if r["status"] == "skipped"),
        "abs_SI_exceeds_2mm_count": sum(1 for r in ok if float(r["setup_error_abs_SI_mm"]) > 2.0),
        "abs_SI_exceeds_3mm_count": sum(1 for r in ok if float(r["setup_error_abs_SI_mm"]) > 3.0),
        "abs_yaw_exceeds_2deg_count": sum(1 for r in ok if float(r["setup_error_abs_yaw_deg"]) > 2.0),
        "abs_yaw_exceeds_5deg_count": sum(1 for r in ok if float(r["setup_error_abs_yaw_deg"]) > 5.0),
        "patient_summary_csv": str(output_dir / "patient_summary.csv"),
        "results_csv": str(output_dir / "registration_results.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild setup-error reports with S-I displacement and yaw rotation as primary metrics.")
    parser.add_argument("--registration-results", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration_fixed" / "registration_results.csv")
    parser.add_argument("--initializations", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration_fixed" / "registration_initializations.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration_si_yaw_fixed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [enrich_result(row) for row in read_csv(args.registration_results)]
    write_csv(rows, args.output_dir / "registration_results.csv")
    write_json(
        args.output_dir / "registration_results.json",
        {
            "schema": "x2cbct_setup_error_si_yaw_registration.v2",
            "primary_metrics": ["setup_error_abs_SI_mm", "setup_error_abs_yaw_deg"],
            "results": rows,
        },
    )
    write_csv(patient_summary(rows), args.output_dir / "patient_summary.csv")
    write_json(args.output_dir / "dataset_summary.json", dataset_summary(rows, args.output_dir))
    if args.initializations.exists():
        init_rows = [enrich_result(row) if "SI_mm" in row and "yaw_deg" in row else row for row in read_csv(args.initializations)]
        write_csv(init_rows, args.output_dir / "registration_initializations.csv")
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
