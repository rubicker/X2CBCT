from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


METRIC_COLUMNS = ("mae_norm", "mae_hu", "psnr", "ssim")
FULL_KINDS = {"full", "full_final"}
HASH_MODEL_VERSIONS = {
    "v0b",
    "v2-hash-a1",
    "v2-hash-a1a",
    "v2-hash-a1b",
    "v2-hash-a2",
    "v2-hash-a2a",
    "v2-hash-a2b",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sort_key(path: Path) -> tuple[int, str]:
    order = {
        "luna16_v5": 0,
        "luna16_v5_hashA1": 1,
        "luna16_v5_hashA1a": 2,
        "luna16_v5_hashA1b": 3,
        "luna16_v5_hashA2": 4,
        "luna16_v5_hashA2a": 5,
        "luna16_v5_hashA2b": 6,
    }
    return order.get(path.name, 99), path.name


def label_for(run_name: str, model_version: str) -> str:
    if model_version == "v0b":
        return "baseline"
    suffix = run_name.replace("luna16_v5_", "")
    return f"{suffix} ({model_version})"


def validation_records_from_final(run_dir: Path) -> list[dict[str, Any]]:
    final_path = run_dir / "metrics_final.json"
    if not final_path.exists():
        return []
    final = read_json(final_path)
    rows = []
    for record in final.get("validation_records", []):
        if not record.get("ok"):
            continue
        metrics = record.get("metrics_summary")
        if not isinstance(metrics, dict):
            continue
        row = {
            "epoch": int(record["epoch"]),
            "kind": str(record["kind"]),
            "output_dir": str(record.get("output_dir", "")),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def validation_records_from_dirs(run_dir: Path) -> list[dict[str, Any]]:
    validation_dir = run_dir / "validation"
    if not validation_dir.exists():
        return []
    rows = []
    for child in sorted(validation_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = child / "test" / "metrics_summary.json"
        if not summary.exists():
            continue
        match = re.match(r"epoch_(\d+)_(.+)", child.name)
        if not match:
            continue
        row = {
            "epoch": int(match.group(1)),
            "kind": match.group(2),
            "output_dir": str(child),
        }
        row.update(read_json(summary))
        rows.append(row)
    return rows


def load_train_loss(run_dir: Path) -> dict[str, float | int | None]:
    path = run_dir / "train_log.csv"
    if not path.exists():
        return {"train_epochs": None, "train_loss_first": None, "train_loss_min": None, "train_loss_final": None}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    losses = [float(row["train_loss"]) for row in rows if row.get("train_loss")]
    return {
        "train_epochs": len(rows),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_min": min(losses) if losses else None,
        "train_loss_final": losses[-1] if losses else None,
    }


def collect_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = read_json(run_dir / "config_resolved.json")
    model_config = cfg.get("model_config", {})
    hashgrid = model_config.get("hashgrid", {}) if isinstance(model_config, dict) else {}
    injection = model_config.get("position_injection", {}) if isinstance(model_config, dict) else {}
    model_version = str(cfg.get("model_version", model_config.get("model_version", "unknown")))
    config_row = {
        "run": run_dir.name,
        "label": label_for(run_dir.name, model_version),
        "model_version": model_version,
        "view_count": cfg.get("view_count"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "num_points": cfg.get("num_points"),
        "lr": cfg.get("lr"),
        "combine": cfg.get("combine"),
        "mid_channels": cfg.get("mid_channels"),
        "loss_valid_policy": cfg.get("loss_valid_policy"),
        "hashgrid_backend_requested": cfg.get("hashgrid_backend"),
        "hashgrid_backend_actual": hashgrid.get("backend_actual"),
        "hash_num_levels": hashgrid.get("num_levels"),
        "hash_level_dim": hashgrid.get("level_dim"),
        "hash_base_resolution": hashgrid.get("base_resolution"),
        "hash_log2_hashmap_size": hashgrid.get("log2_hashmap_size"),
        "hash_output_dim": hashgrid.get("output_dim"),
        "position_injection_method": injection.get("method"),
        "position_gate_type": injection.get("gate_type"),
        "classifier_input_channels": injection.get("classifier_input_channels"),
    }
    config_row.update(load_train_loss(run_dir))

    records = validation_records_from_final(run_dir) or validation_records_from_dirs(run_dir)
    metric_rows = []
    for record in records:
        metric_rows.append(
            {
                "run": run_dir.name,
                "label": config_row["label"],
                "model_version": model_version,
                "epoch": int(record["epoch"]),
                "kind": str(record["kind"]),
                "case_count": int(record.get("case_count", 0)),
                "split": record.get("split"),
                "view_count": record.get("view_count"),
                "mae_norm": float(record["mae_norm"]),
                "mae_hu": float(record["mae_hu"]),
                "psnr": float(record["psnr"]),
                "ssim": float(record["ssim"]),
                "output_dir": record.get("output_dir", ""),
            }
        )
    return config_row, metric_rows


def plot_full_metrics(rows: list[dict[str, Any]], out_path: Path) -> None:
    full_rows = [row for row in rows if row["kind"] in FULL_KINDS]
    labels = []
    for row in full_rows:
        if row["label"] not in labels:
            labels.append(row["label"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.reshape(-1)
    titles = {
        "mae_norm": "MAE norm (lower is better)",
        "mae_hu": "MAE HU (lower is better)",
        "psnr": "PSNR (higher is better)",
        "ssim": "SSIM (higher is better)",
    }
    for ax, metric in zip(axes, METRIC_COLUMNS):
        for label in labels:
            series = sorted([row for row in full_rows if row["label"] == label], key=lambda item: item["epoch"])
            ax.plot(
                [row["epoch"] for row in series],
                [row[metric] for row in series],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
        ax.set_title(titles[metric])
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("LUNA16 v5 full validation metrics by hash configuration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def best_and_final_summary(configs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_run = {cfg["run"]: cfg for cfg in configs}
    for run_name, cfg in by_run.items():
        full = sorted([row for row in rows if row["run"] == run_name and row["kind"] in FULL_KINDS], key=lambda item: item["epoch"])
        if not full:
            continue
        best_psnr = max(full, key=lambda item: item["psnr"])
        best_ssim = max(full, key=lambda item: item["ssim"])
        final = full[-1]
        out.append(
            {
                "run": run_name,
                "label": cfg["label"],
                "model_version": cfg["model_version"],
                "best_psnr_epoch": best_psnr["epoch"],
                "best_full_psnr": best_psnr["psnr"],
                "best_ssim_epoch": best_ssim["epoch"],
                "best_full_ssim": best_ssim["ssim"],
                "final_full_epoch": final["epoch"],
                "final_full_psnr": final["psnr"],
                "final_full_ssim": final["ssim"],
                "train_loss_min": cfg.get("train_loss_min"),
                "train_loss_final": cfg.get("train_loss_final"),
            }
        )
    baseline = next((row for row in out if row["model_version"] == "v0b"), None)
    if baseline is not None:
        for row in out:
            row["delta_best_psnr_vs_baseline"] = row["best_full_psnr"] - baseline["best_full_psnr"]
            row["delta_final_psnr_vs_baseline"] = row["final_full_psnr"] - baseline["final_full_psnr"]
            row["delta_best_ssim_vs_baseline"] = row["best_full_ssim"] - baseline["best_full_ssim"]
            row["delta_final_ssim_vs_baseline"] = row["final_full_ssim"] - baseline["final_full_ssim"]
    return out


def read_case_metrics(run_dir: Path, epoch_dir: str = "epoch_0400_full_final") -> dict[str, dict[str, float]]:
    path = run_dir / "validation" / epoch_dir / "test" / "metrics_per_case.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["case_key"]: {
            "mae_norm": float(row["mae_norm"]),
            "mae_hu": float(row["mae_hu"]),
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
        }
        for row in rows
    }


def per_case_delta(output_root: Path, configs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_cfg = next((cfg for cfg in configs if cfg["model_version"] == "v0b"), None)
    if baseline_cfg is None:
        return [], []
    baseline = read_case_metrics(output_root / baseline_cfg["run"])
    rows = []
    summaries = []
    for cfg in configs:
        if cfg["model_version"] == "v0b":
            continue
        current = read_case_metrics(output_root / cfg["run"])
        mae_deltas = []
        psnr_deltas = []
        ssim_deltas = []
        for case_key, base_metrics in baseline.items():
            if case_key not in current:
                continue
            mae_delta = current[case_key]["mae_norm"] - base_metrics["mae_norm"]
            psnr_delta = current[case_key]["psnr"] - base_metrics["psnr"]
            ssim_delta = current[case_key]["ssim"] - base_metrics["ssim"]
            mae_deltas.append(mae_delta)
            psnr_deltas.append(psnr_delta)
            ssim_deltas.append(ssim_delta)
            rows.append(
                {
                    "run": cfg["run"],
                    "label": cfg["label"],
                    "case_key": case_key,
                    "delta_mae_norm_vs_baseline": mae_delta,
                    "delta_mae_hu_vs_baseline": current[case_key]["mae_hu"] - base_metrics["mae_hu"],
                    "delta_psnr_vs_baseline": psnr_delta,
                    "delta_ssim_vs_baseline": ssim_delta,
                    "hash_mae_norm": current[case_key]["mae_norm"],
                    "baseline_mae_norm": base_metrics["mae_norm"],
                }
            )
        if mae_deltas:
            summaries.append(
                {
                    "run": cfg["run"],
                    "label": cfg["label"],
                    "case_count": len(mae_deltas),
                    "improved_cases_by_psnr": sum(1 for item in psnr_deltas if item > 0),
                    "worse_cases_by_psnr": sum(1 for item in psnr_deltas if item < 0),
                    "mean_delta_psnr": sum(psnr_deltas) / len(psnr_deltas),
                    "improved_cases_by_ssim": sum(1 for item in ssim_deltas if item > 0),
                    "worse_cases_by_ssim": sum(1 for item in ssim_deltas if item < 0),
                    "mean_delta_ssim": sum(ssim_deltas) / len(ssim_deltas),
                }
            )
    return rows, summaries


def checkpoint_parameter_stats(output_root: Path, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    interesting = (
        "position_encoder.backend.encoder.embeddings",
        "position_injection.hash_gate",
        "position_injection.point_gate",
        "position_injection.position_map",
        "point_classifier.conv0",
    )
    for cfg in configs:
        checkpoint = output_root / cfg["run"] / "checkpoints" / "epoch_0400.pt"
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model_state", payload)
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not tensor.dtype.is_floating_point:
                continue
            if not any(token in name for token in interesting):
                continue
            flat = tensor.float()
            rows.append(
                {
                    "run": cfg["run"],
                    "label": cfg["label"],
                    "model_version": cfg["model_version"],
                    "parameter": name,
                    "shape": "x".join(str(dim) for dim in tensor.shape),
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std(unbiased=False).item()),
                    "abs_max": float(flat.abs().max().item()),
                    "norm": float(flat.norm().item()),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize LUNA16 v5 hash experiment metrics.")
    parser.add_argument("--output-root", type=Path, default=Path("runs/output"))
    parser.add_argument("--run-prefix", default="luna16_v5")
    parser.add_argument("--analysis-dir", type=Path, default=Path("runs/output/luna16_v5_hash_analysis"))
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    run_dirs = []
    for path in output_root.iterdir():
        config_path = path / "config_resolved.json"
        if not path.is_dir() or not path.name.startswith(args.run_prefix) or not config_path.exists():
            continue
        config = read_json(config_path)
        model_config = config.get("model_config", {})
        model_version = str(config.get("model_version", model_config.get("model_version", "unknown")))
        combine = str(config.get("combine", model_config.get("combine", "")))
        if model_version in HASH_MODEL_VERSIONS and combine in {"mlp", "max"}:
            run_dirs.append(path)
    run_dirs = sorted(run_dirs, key=run_sort_key)
    if not run_dirs:
        raise FileNotFoundError(f"no runs found under {output_root} with prefix {args.run_prefix!r}")

    configs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config_row, rows = collect_run(run_dir)
        configs.append(config_row)
        metric_rows.extend(rows)

    analysis_dir = args.analysis_dir.resolve()
    write_csv(configs, analysis_dir / "run_configs.csv")
    write_csv(metric_rows, analysis_dir / "validation_metrics.csv")
    plot_full_metrics(metric_rows, analysis_dir / "luna16_v5_full_validation_metrics.png")

    summary_rows = best_and_final_summary(configs, metric_rows)
    write_csv(summary_rows, analysis_dir / "best_and_final_summary.csv")
    case_rows, case_summary = per_case_delta(output_root, configs)
    write_csv(case_rows, analysis_dir / "per_case_delta_vs_baseline_final_full.csv")
    write_csv(case_summary, analysis_dir / "per_case_delta_summary_final_full.csv")
    parameter_rows = checkpoint_parameter_stats(output_root, configs)
    write_csv(parameter_rows, analysis_dir / "checkpoint_parameter_stats_epoch0400.csv")

    write_json(
        {
            "ok": True,
            "run_count": len(run_dirs),
            "runs": [path.name for path in run_dirs],
            "artifacts": {
                "run_configs": str(analysis_dir / "run_configs.csv"),
                "validation_metrics": str(analysis_dir / "validation_metrics.csv"),
                "full_validation_plot": str(analysis_dir / "luna16_v5_full_validation_metrics.png"),
                "best_and_final_summary": str(analysis_dir / "best_and_final_summary.csv"),
                "per_case_delta": str(analysis_dir / "per_case_delta_vs_baseline_final_full.csv"),
                "per_case_delta_summary": str(analysis_dir / "per_case_delta_summary_final_full.csv"),
                "checkpoint_parameter_stats": str(analysis_dir / "checkpoint_parameter_stats_epoch0400.csv"),
            },
        },
        analysis_dir / "analysis_summary.json",
    )
    print(f"wrote analysis artifacts to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
