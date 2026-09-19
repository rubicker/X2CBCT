from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lungmask" / "luna16_view5_reconstruction_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "segmentation_lungmask" / "reconstruction_metrics_view5"
METRICS = ("mae_norm", "mae_hu", "psnr", "ssim")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _canonical_case_id(row: dict[str, str], column: str | None) -> str:
    candidate_columns = [column] if column else ["case_id", "case_key", "patient_id"]
    value = ""
    for candidate in candidate_columns:
        if candidate and row.get(candidate):
            value = str(row[candidate]).strip()
            break
    if not value:
        raise ValueError(f"row has no usable case identifier; checked {candidate_columns}")
    return value[len("luna16_") :] if value.startswith("luna16_") else value


def _load_method(method: dict[str, Any], expected_cases: int) -> dict[str, dict[str, float]]:
    path = _resolve(method["metrics_csv"])
    mapping = {metric: metric for metric in METRICS}
    mapping.update(method.get("metric_columns") or {})
    records: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = _canonical_case_id(row, method.get("case_id_column"))
            if case_id in records:
                raise ValueError(f"duplicate case {case_id} in {path}")
            values = {metric: float(row[mapping[metric]]) for metric in METRICS}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite metric for case {case_id} in {path}")
            records[case_id] = values
    if len(records) != expected_cases:
        raise ValueError(f"{method['name']} has {len(records)} cases, expected {expected_cases}: {path}")
    return records


def _summary_row(method: str, records: dict[str, dict[str, float]]) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "case_count": len(records)}
    for metric in METRICS:
        values = np.asarray([item[metric] for item in records.values()], dtype=np.float64)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1))
        row[f"{metric}_median"] = float(np.median(values))
        row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
        row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
    return row


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, iterations: int) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(int(iterations), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_comparisons(
    methods: dict[str, dict[str, dict[str, float]]],
    *,
    primary: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if primary not in methods:
        raise ValueError(f"primary method is absent: {primary}")
    comparisons: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    direction = {"mae_norm": -1.0, "mae_hu": -1.0, "psnr": 1.0, "ssim": 1.0}
    for metric in METRICS:
        metric_rows: list[dict[str, Any]] = []
        for baseline, baseline_records in methods.items():
            if baseline == primary:
                continue
            shared = sorted(set(methods[primary]) & set(baseline_records))
            primary_values = np.asarray([methods[primary][case][metric] for case in shared], dtype=np.float64)
            baseline_values = np.asarray([baseline_records[case][metric] for case in shared], dtype=np.float64)
            improvement = direction[metric] * (primary_values - baseline_values)
            ci_low, ci_high = _bootstrap_ci(improvement, rng=rng, iterations=iterations)
            p_value = (
                1.0
                if np.allclose(improvement, 0.0)
                else float(scipy.stats.wilcoxon(improvement, alternative="two-sided").pvalue)
            )
            metric_rows.append(
                {
                    "primary_method": primary,
                    "baseline_method": baseline,
                    "metric": metric,
                    "positive_improvement_definition": (
                        "primary_minus_baseline" if direction[metric] > 0 else "baseline_minus_primary"
                    ),
                    "paired_case_count": len(shared),
                    "primary_mean": float(primary_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "mean_improvement": float(improvement.mean()),
                    "improvement_ci95_low": ci_low,
                    "improvement_ci95_high": ci_high,
                    "wilcoxon_p": p_value,
                    "holm_adjusted_p": None,
                }
            )
        order = sorted(range(len(metric_rows)), key=lambda index: metric_rows[index]["wilcoxon_p"])
        running_max = 0.0
        total = len(metric_rows)
        for rank, index in enumerate(order, start=1):
            adjusted = min(1.0, metric_rows[index]["wilcoxon_p"] * (total - rank + 1))
            running_max = max(running_max, adjusted)
            metric_rows[index]["holm_adjusted_p"] = running_max
        comparisons.extend(metric_rows)
    return comparisons


def summarize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    expected_cases = int(config["expected_cases"])
    methods = {
        str(method["name"]): _load_method(method, expected_cases)
        for method in config["methods"]
    }
    common_cases = set.intersection(*(set(records) for records in methods.values()))
    if len(common_cases) != expected_cases:
        counts = {name: len(set(records) - common_cases) for name, records in methods.items()}
        raise ValueError(f"strict paired cohort has {len(common_cases)} cases, expected {expected_cases}; extras={counts}")

    summary_rows = [_summary_row(name, records) for name, records in methods.items()]
    long_rows = [
        {"case_id": case_id, "method": name, **records[case_id]}
        for case_id in sorted(common_cases)
        for name, records in methods.items()
    ]
    comparisons = _paired_comparisons(
        methods,
        primary=str(config["primary_method"]),
        iterations=int(config.get("bootstrap_iterations", 10000)),
        seed=int(config.get("bootstrap_seed", 20260724)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_rows, output_dir / "metrics_summary.csv")
    _write_csv(long_rows, output_dir / "metrics_per_case_long.csv")
    _write_csv(comparisons, output_dir / "paired_vs_case_cbct.csv")
    result = {
        "schema": "x2cbct_reconstruction_metrics_result.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path.resolve()),
        "case_count": len(common_cases),
        "method_count": len(methods),
        "primary_method": config["primary_method"],
        "metrics": list(METRICS),
        "summary": {row["method"]: row for row in summary_rows},
        "statistical_analysis": {
            "paired_test": "two-sided Wilcoxon signed-rank",
            "multiple_comparison_correction": "Holm within each metric",
            "confidence_interval": (
                f"paired bootstrap mean improvement, 95%, "
                f"{int(config.get('bootstrap_iterations', 10000))} resamples"
            ),
        },
    }
    _write_json(result, output_dir / "metrics_summary.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strict paired view-5 reconstruction metric table.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = summarize(_resolve(args.config), _resolve(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
