from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_augmentation import transform_volume_setup, volume_background_value


METRIC_KEYS = ("mae_norm", "mae_hu", "psnr", "ssim")
DEFAULT_EQUIVARIANCE_IDS = (
    "si_m6_rot_0",
    "si_p6_rot_0",
    "si_0_rot_m4",
    "si_0_rot_p4",
    "si_0_rot_m9",
    "si_0_rot_p9",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_id(case_key: str) -> str:
    text = str(case_key)
    if text.startswith("luna16_"):
        text = text[len("luna16_") :]
    return text.split("_si_", 1)[0]


def _read_metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "metrics_per_case.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_id"] = _source_id(str(row["case_key"]))
        for key in METRIC_KEYS:
            row[key] = float(row[key])
        row["translation_mm"] = int(row.get("translation_mm") or 0)
        row["rotation_deg"] = int(row.get("rotation_deg") or 0)
    return rows


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _contrast_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) > 1:
        interval = stats.t.interval(
            0.95,
            len(array) - 1,
            loc=float(array.mean()),
            scale=float(stats.sem(array)),
        )
        p_value = float(stats.ttest_1samp(array, 0.0).pvalue)
    else:
        interval = (float("nan"), float("nan"))
        p_value = float("nan")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "mean_95ci": [float(interval[0]), float(interval[1])],
        "paired_source_ttest_p": p_value,
        "positive_case_count": int(np.sum(array > 0.0)),
        "negative_case_count": int(np.sum(array < 0.0)),
        "zero_case_count": int(np.sum(array == 0.0)),
    }


def _paired_summary(
    base_by_source: dict[str, dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented_rows:
        by_source[row["source_id"]].append(row)

    output: dict[str, Any] = {}
    for key in METRIC_KEYS:
        sample_deltas = np.asarray(
            [float(row[key]) - float(base_by_source[row["source_id"]][key]) for row in augmented_rows],
            dtype=np.float64,
        )
        source_deltas = np.asarray(
            [
                _mean([float(row[key]) - float(base_by_source[source_id][key]) for row in rows])
                for source_id, rows in sorted(by_source.items())
            ],
            dtype=np.float64,
        )
        if len(source_deltas) > 1:
            interval = stats.t.interval(
                0.95,
                len(source_deltas) - 1,
                loc=float(source_deltas.mean()),
                scale=float(stats.sem(source_deltas)),
            )
            p_value = float(stats.ttest_1samp(source_deltas, 0.0).pvalue)
        else:
            interval = (float("nan"), float("nan"))
            p_value = float("nan")
        output[key] = {
            "sample_mean_delta": float(sample_deltas.mean()),
            "source_mean_delta": float(source_deltas.mean()),
            "source_delta_sd": float(source_deltas.std(ddof=1)) if len(source_deltas) > 1 else 0.0,
            "source_mean_delta_95ci": [float(interval[0]), float(interval[1])],
            "paired_source_ttest_p": p_value,
        }
    return output


def _factor_contrasts(
    base_by_source: dict[str, dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in augmented_rows:
        by_source[row["source_id"]][(int(row["translation_mm"]), int(row["rotation_deg"]))] = row

    pure_translation = ((-6, 0), (6, 0))
    pure_rotation_4 = ((0, -4), (0, 4))
    pure_rotation_9 = ((0, -9), (0, 9))
    all_rotation_4 = tuple((translation, rotation) for translation in (-6, 0, 6) for rotation in (-4, 4))
    all_rotation_9 = tuple((translation, rotation) for translation in (-6, 0, 6) for rotation in (-9, 9))
    source_ids = sorted(base_by_source)

    def delta(source_id: str, setup: tuple[int, int], metric: str) -> float:
        return float(by_source[source_id][setup][metric]) - float(base_by_source[source_id][metric])

    def average(source_id: str, setups: tuple[tuple[int, int], ...], metric: str) -> float:
        return _mean([delta(source_id, setup, metric) for setup in setups])

    definitions = (
        ("pure_translation_vs_base", lambda source_id, metric: average(source_id, pure_translation, metric)),
        ("pure_rotation_4_vs_base", lambda source_id, metric: average(source_id, pure_rotation_4, metric)),
        ("pure_rotation_9_vs_base", lambda source_id, metric: average(source_id, pure_rotation_9, metric)),
        ("all_rotation_4_vs_base", lambda source_id, metric: average(source_id, all_rotation_4, metric)),
        ("all_rotation_9_vs_base", lambda source_id, metric: average(source_id, all_rotation_9, metric)),
        (
            "pure_rotation_9_minus_pure_translation",
            lambda source_id, metric: average(source_id, pure_rotation_9, metric)
            - average(source_id, pure_translation, metric),
        ),
        (
            "pure_rotation_9_minus_pure_rotation_4",
            lambda source_id, metric: average(source_id, pure_rotation_9, metric)
            - average(source_id, pure_rotation_4, metric),
        ),
        (
            "all_rotation_9_minus_all_rotation_4",
            lambda source_id, metric: average(source_id, all_rotation_9, metric)
            - average(source_id, all_rotation_4, metric),
        ),
        (
            "plus6_at_rotation9_minus_zero_translation",
            lambda source_id, metric: _mean(
                [
                    delta(source_id, (6, -9), metric) - delta(source_id, (0, -9), metric),
                    delta(source_id, (6, 9), metric) - delta(source_id, (0, 9), metric),
                ]
            ),
        ),
    )
    output = []
    for metric in METRIC_KEYS:
        for name, calculator in definitions:
            output.append(
                {
                    "metric": metric,
                    "contrast": name,
                    **_contrast_summary([calculator(source_id, metric) for source_id in source_ids]),
                }
            )
    return output


def _augmentation_rows(
    base_by_source: dict[str, dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented_rows:
        groups[str(row["augmentation_id"])].append(row)
    output = []
    for augmentation_id, rows in groups.items():
        item: dict[str, Any] = {
            "augmentation_id": augmentation_id,
            "translation_mm": int(rows[0]["translation_mm"]),
            "rotation_deg": int(rows[0]["rotation_deg"]),
            "case_count": len(rows),
        }
        for key in METRIC_KEYS:
            item[key] = _mean([float(row[key]) for row in rows])
            item[f"{key}_delta"] = _mean(
                [float(row[key]) - float(base_by_source[row["source_id"]][key]) for row in rows]
            )
        output.append(item)
    return sorted(output, key=lambda row: (row["translation_mm"], row["rotation_deg"]))


def _case_difficulty(
    base_by_source: dict[str, dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented_rows:
        groups[row["source_id"]].append(row)
    output: dict[str, Any] = {}
    for key in ("psnr", "ssim"):
        source_ids = sorted(base_by_source)
        base_values = np.asarray([base_by_source[source_id][key] for source_id in source_ids], dtype=np.float64)
        augmented_means = np.asarray(
            [_mean([row[key] for row in groups[source_id]]) for source_id in source_ids],
            dtype=np.float64,
        )
        within_source_sd = _mean(
            [
                float(np.std([row[key] for row in groups[source_id]], ddof=1))
                for source_id in source_ids
            ]
        )
        output[key] = {
            "base_vs_augmented_source_mean_pearson_r": float(stats.pearsonr(base_values, augmented_means).statistic),
            "mean_within_source_transform_sd": within_source_sd,
            "between_source_base_sd": float(base_values.std(ddof=1)),
        }
    return output


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _foreground_metrics(row: dict[str, Any], threshold: float) -> dict[str, float]:
    prediction = np.load(row["generated_cbct_path"], mmap_mode="r")
    target = np.load(row["target_cbct_path"], mmap_mode="r")
    mask = np.asarray(target) > float(threshold)
    difference = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    foreground_mse = float(np.mean(np.square(difference[mask])))
    background_mse = float(np.mean(np.square(difference[~mask])))
    return {
        "foreground_fraction": float(mask.mean()),
        "foreground_mae": float(np.mean(np.abs(difference[mask]))),
        "foreground_psnr": float(-10.0 * math.log10(max(foreground_mse, 1e-12))),
        "background_mae": float(np.mean(np.abs(difference[~mask]))),
        "background_psnr": float(-10.0 * math.log10(max(background_mse, 1e-12))),
    }


def _gradient_energy(path: str | Path) -> float:
    array = np.load(path, mmap_mode="r")
    return _mean([float(np.mean(np.abs(np.diff(array, axis=axis)))) for axis in range(3)])


def _volume_analysis(
    base_predictions: list[dict[str, Any]],
    augmented_predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    started = time.time()
    base_by_source: dict[str, dict[str, Any]] = {}
    for row in base_predictions:
        source_id = _source_id(str(row.get("source_case_id") or row["case_key"]))
        base_by_source[source_id] = {
            **_foreground_metrics(row, threshold),
            "target_gradient_energy": _gradient_energy(row["target_cbct_path"]),
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(augmented_predictions, start=1):
        source_id = _source_id(str(row.get("source_case_id") or row["case_key"]))
        values = {
            **_foreground_metrics(row, threshold),
            "target_gradient_energy": _gradient_energy(row["target_cbct_path"]),
        }
        base = base_by_source[source_id]
        groups[str(row["augmentation_id"])].append(
            {
                **values,
                "gradient_ratio_to_base": values["target_gradient_energy"] / base["target_gradient_energy"],
                "source_id": source_id,
                "translation_mm": int(row["translation_mm"]),
                "rotation_deg": int(row["rotation_deg"]),
            }
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "volume_analysis_progress",
                        "done": index,
                        "total": len(augmented_predictions),
                        "elapsed_sec": time.time() - started,
                    }
                ),
                flush=True,
            )

    keys = (
        "foreground_fraction",
        "foreground_mae",
        "foreground_psnr",
        "background_mae",
        "background_psnr",
    )
    base_means = {key: _mean([row[key] for row in base_by_source.values()]) for key in keys}
    by_augmentation = []
    for augmentation_id, rows in groups.items():
        item: dict[str, Any] = {
            "augmentation_id": augmentation_id,
            "translation_mm": rows[0]["translation_mm"],
            "rotation_deg": rows[0]["rotation_deg"],
            "case_count": len(rows),
            "target_gradient_ratio_to_base": _mean([row["gradient_ratio_to_base"] for row in rows]),
        }
        for key in keys:
            item[key] = _mean([row[key] for row in rows])
            item[f"{key}_delta"] = _mean(
                [row[key] - base_by_source[row["source_id"]][key] for row in rows]
            )
        by_augmentation.append(item)
    by_augmentation.sort(key=lambda row: (row["translation_mm"], row["rotation_deg"]))
    all_augmented = [row for rows in groups.values() for row in rows]
    augmented_means = {key: _mean([row[key] for row in all_augmented]) for key in keys}
    return {
        "foreground_threshold": float(threshold),
        "base_means": base_means,
        "augmented_means": augmented_means,
        "augmented_paired_deltas": {
            key: _mean([row[key] - base_by_source[row["source_id"]][key] for row in all_augmented])
            for key in keys
        },
        "by_augmentation": by_augmentation,
        "elapsed_sec": time.time() - started,
    }


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))))


def _equivariance_analysis(
    base_predictions: list[dict[str, Any]],
    augmented_predictions: list[dict[str, Any]],
    *,
    augmentation_ids: set[str],
    output_spacing_z_mm: float,
) -> dict[str, Any]:
    base_by_source = {
        _source_id(str(row.get("source_case_id") or row["case_key"])): row for row in base_predictions
    }
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in augmented_predictions:
        augmentation_id = str(row["augmentation_id"])
        if augmentation_id not in augmentation_ids:
            continue
        source_id = _source_id(str(row.get("source_case_id") or row["case_key"]))
        base = base_by_source[source_id]
        base_target = np.load(base["target_cbct_path"]).astype(np.float32)
        base_prediction = np.load(base["generated_cbct_path"]).astype(np.float32)
        augmented_target = np.load(row["target_cbct_path"]).astype(np.float32)
        augmented_prediction = np.load(row["generated_cbct_path"]).astype(np.float32)
        transform_kwargs = {
            "translation_mm": int(row["translation_mm"]),
            "rotation_deg": int(row["rotation_deg"]),
            "spacing_z_mm": float(output_spacing_z_mm),
        }
        expected_target = transform_volume_setup(
            base_target,
            fill_value=volume_background_value(base_target),
            **transform_kwargs,
        )
        expected_prediction = transform_volume_setup(
            base_prediction,
            fill_value=volume_background_value(base_prediction),
            **transform_kwargs,
        )
        groups[augmentation_id].append(
            {
                "source_id": source_id,
                "target_direct_difference": _mae(augmented_target, base_target),
                "target_expected_transform_residual": _mae(augmented_target, expected_target),
                "prediction_direct_difference": _mae(augmented_prediction, base_prediction),
                "prediction_expected_transform_residual": _mae(augmented_prediction, expected_prediction),
                "prediction_target_mae": _mae(augmented_prediction, augmented_target),
            }
        )

    by_augmentation = []
    for augmentation_id, rows in groups.items():
        direct_target = _mean([row["target_direct_difference"] for row in rows])
        expected_target = _mean([row["target_expected_transform_residual"] for row in rows])
        direct_prediction = _mean([row["prediction_direct_difference"] for row in rows])
        expected_prediction = _mean([row["prediction_expected_transform_residual"] for row in rows])
        by_augmentation.append(
            {
                "augmentation_id": augmentation_id,
                "case_count": len(rows),
                "target_direct_difference": direct_target,
                "target_expected_transform_residual": expected_target,
                "target_residual_over_direct": expected_target / direct_target,
                "prediction_direct_difference": direct_prediction,
                "prediction_expected_transform_residual": expected_prediction,
                "prediction_residual_over_direct": expected_prediction / direct_prediction,
                "prediction_target_mae": _mean([row["prediction_target_mae"] for row in rows]),
            }
        )
    by_augmentation.sort(key=lambda row: row["augmentation_id"])
    source_values: dict[str, dict[str, float]] = defaultdict(dict)
    for augmentation_id, rows in groups.items():
        for row in rows:
            source_values[str(row["source_id"])][augmentation_id] = float(
                row["prediction_expected_transform_residual"]
            )
    source_ids = sorted(source_values)

    def source_average(source_id: str, augmentation_ids: tuple[str, ...]) -> float:
        return _mean([source_values[source_id][augmentation_id] for augmentation_id in augmentation_ids])

    translation_ids = ("si_m6_rot_0", "si_p6_rot_0")
    rotation_4_ids = ("si_0_rot_m4", "si_0_rot_p4")
    rotation_9_ids = ("si_0_rot_m9", "si_0_rot_p9")
    residual_groups = {
        "pure_translation": translation_ids,
        "pure_rotation_4": rotation_4_ids,
        "pure_rotation_9": rotation_9_ids,
    }
    residual_summary = {
        name: _contrast_summary([source_average(source_id, ids) for source_id in source_ids])
        for name, ids in residual_groups.items()
    }
    residual_contrasts = {
        "pure_rotation_4_minus_pure_translation": _contrast_summary(
            [
                source_average(source_id, rotation_4_ids)
                - source_average(source_id, translation_ids)
                for source_id in source_ids
            ]
        ),
        "pure_rotation_9_minus_pure_translation": _contrast_summary(
            [
                source_average(source_id, rotation_9_ids)
                - source_average(source_id, translation_ids)
                for source_id in source_ids
            ]
        ),
        "pure_rotation_9_minus_pure_rotation_4": _contrast_summary(
            [
                source_average(source_id, rotation_9_ids)
                - source_average(source_id, rotation_4_ids)
                for source_id in source_ids
            ]
        ),
    }
    return {
        "by_augmentation": by_augmentation,
        "prediction_transform_residual_groups": residual_summary,
        "prediction_transform_residual_contrasts": residual_contrasts,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    paired = summary["paired_metric_deltas"]
    case_difficulty = summary["case_difficulty"]
    by_augmentation = summary["by_augmentation"]
    best_psnr = max(by_augmentation, key=lambda row: row["psnr_delta"])
    worst_psnr = min(by_augmentation, key=lambda row: row["psnr_delta"])
    factor_lookup = {
        (row["metric"], row["contrast"]): row for row in summary.get("factor_contrasts", [])
    }
    lines = [
        "# LUNA16 摆位增强鲁棒性配对分析",
        "",
        "## 样本关系",
        "",
        f"- 原始 source case：{summary['counts']['source_cases']}。",
        f"- 原始测试样本：{summary['counts']['base_samples']}。",
        f"- 增强测试样本：{summary['counts']['augmented_samples']}。",
        f"- 每个 source case 的增强样本：{summary['counts']['augmentations_per_source']}。",
        "",
        "## 配对结论",
        "",
        f"- PSNR 平均变化：{paired['psnr']['source_mean_delta']:+.6f} dB，"
        f"95% CI [{paired['psnr']['source_mean_delta_95ci'][0]:+.6f}, "
        f"{paired['psnr']['source_mean_delta_95ci'][1]:+.6f}]。",
        f"- SSIM 平均变化：{paired['ssim']['source_mean_delta']:+.6f}，"
        f"95% CI [{paired['ssim']['source_mean_delta_95ci'][0]:+.6f}, "
        f"{paired['ssim']['source_mean_delta_95ci'][1]:+.6f}]。",
        f"- 同一病例变换前后 PSNR 相关系数："
        f"{case_difficulty['psnr']['base_vs_augmented_source_mean_pearson_r']:.6f}。",
        f"- 同一病例变换前后 SSIM 相关系数："
        f"{case_difficulty['ssim']['base_vs_augmented_source_mean_pearson_r']:.6f}。",
        f"- PSNR 上升最多的档位：`{best_psnr['augmentation_id']}`，"
        f"配对变化 {best_psnr['psnr_delta']:+.6f} dB。",
        f"- PSNR 下降最多的档位：`{worst_psnr['augmentation_id']}`，"
        f"配对变化 {worst_psnr['psnr_delta']:+.6f} dB。",
        "",
        "## 平移与旋转对比",
        "",
    ]
    pure_translation = factor_lookup.get(("psnr", "pure_translation_vs_base"))
    pure_rotation_9 = factor_lookup.get(("psnr", "pure_rotation_9_vs_base"))
    rotation_9_minus_translation = factor_lookup.get(
        ("psnr", "pure_rotation_9_minus_pure_translation")
    )
    rotation_9_minus_rotation_4 = factor_lookup.get(
        ("psnr", "pure_rotation_9_minus_pure_rotation_4")
    )
    if pure_translation and pure_rotation_9 and rotation_9_minus_translation and rotation_9_minus_rotation_4:
        lines.extend(
            [
                f"- 纯 `±6 mm` 平移 PSNR 变化：{pure_translation['mean']:+.6f} dB，"
                f"95% CI [{pure_translation['mean_95ci'][0]:+.6f}, "
                f"{pure_translation['mean_95ci'][1]:+.6f}]。",
                f"- 纯 `±9°` 旋转 PSNR 变化：{pure_rotation_9['mean']:+.6f} dB，"
                f"95% CI [{pure_rotation_9['mean_95ci'][0]:+.6f}, "
                f"{pure_rotation_9['mean_95ci'][1]:+.6f}]。",
                f"- 纯 `±9°` 相对纯平移额外降低："
                f"{rotation_9_minus_translation['mean']:+.6f} dB。",
                f"- 纯 `±9°` 相对纯 `±4°` 额外降低："
                f"{rotation_9_minus_rotation_4['mean']:+.6f} dB。",
                "",
            ]
        )
    lines.extend(["## 体数据检查", ""])
    volume = summary.get("volume_analysis")
    if volume:
        foreground = volume["augmented_paired_deltas"]
        gradient_ratios = [
            float(row["target_gradient_ratio_to_base"]) for row in volume["by_augmentation"]
        ]
        lines.extend(
            [
                f"- 前景区域 MAE 平均变化：{foreground['foreground_mae']:+.6f}。",
                f"- 前景区域 PSNR 平均变化：{foreground['foreground_psnr']:+.6f} dB。",
                f"- 增强目标梯度相对原目标的范围："
                f"{min(gradient_ratios):.3f}～{max(gradient_ratios):.3f}。",
            ]
        )
    equivariance = summary.get("equivariance_analysis")
    if equivariance:
        by_equivariance = equivariance["by_augmentation"]
        target_ratios = [float(row["target_residual_over_direct"]) for row in by_equivariance]
        prediction_ratios = [float(row["prediction_residual_over_direct"]) for row in by_equivariance]
        residual_groups = equivariance["prediction_transform_residual_groups"]
        lines.extend(
            [
                f"- 目标按记录参数变换后的残差占直接差异："
                f"{min(target_ratios):.3f}～{max(target_ratios):.3f}。",
                f"- 预测按同样参数变换后的残差占直接差异："
                f"{min(prediction_ratios):.3f}～{max(prediction_ratios):.3f}。",
                f"- 纯平移预测跟随残差：{residual_groups['pure_translation']['mean']:.6f}。",
                f"- 纯 `±4°` 预测跟随残差：{residual_groups['pure_rotation_4']['mean']:.6f}。",
                f"- 纯 `±9°` 预测跟随残差：{residual_groups['pure_rotation_9']['mean']:.6f}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "总体均值接近不代表数据没有发生变化。配对结果显示各档位存在稳定差异，"
            "病例本身的难度差异明显大于同一病例内部的摆位变换差异。",
            "旋转和平移使用线性插值，会降低目标体数据的高频细节，使部分档位更容易获得较高指标。",
            "预测随目标一起移动和旋转，表明当前 DIF-Net 的投影特征查询机制已经具备较强的同步刚体变换适应性。",
            "",
            "详细数值见 `analysis_summary.json`、`metrics_by_augmentation.csv`、"
            "`factor_contrasts.csv`、`foreground_by_augmentation.csv` 和 "
            "`equivariance_by_augmentation.csv`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired analysis for LUNA16 base and setup-augmented test runs.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=PROJECT_ROOT / "runs" / "test" / "luna16_v5_oldeval",
    )
    parser.add_argument(
        "--augmented-run",
        type=Path,
        default=PROJECT_ROOT / "runs" / "test" / "luna16_v5_oldeval_aug",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "audits" / "luna16_v5_oldeval_aug_diagnosis",
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--output-spacing-z-mm", type=float, default=3.2)
    parser.add_argument("--skip-volume-analysis", action="store_true")
    parser.add_argument(
        "--equivariance-augmentation-ids",
        default=",".join(DEFAULT_EQUIVARIANCE_IDS),
    )
    args = parser.parse_args()

    base_run = args.base_run.resolve()
    augmented_run = args.augmented_run.resolve()
    output_dir = args.output_dir.resolve()
    base_rows = _read_metric_rows(base_run)
    augmented_rows = _read_metric_rows(augmented_run)
    base_by_source = {row["source_id"]: row for row in base_rows}
    missing_sources = sorted({row["source_id"] for row in augmented_rows} - set(base_by_source))
    if missing_sources:
        raise RuntimeError(f"{len(missing_sources)} augmented source cases are missing from base run")
    per_source_counts = defaultdict(int)
    for row in augmented_rows:
        per_source_counts[row["source_id"]] += 1
    unique_counts = sorted(set(per_source_counts.values()))
    if len(unique_counts) != 1:
        raise RuntimeError(f"augmentation count differs between source cases: {unique_counts}")

    base_manifest = _read_json(base_run / "prediction_manifest.json")
    augmented_manifest = _read_json(augmented_run / "prediction_manifest.json")
    by_augmentation = _augmentation_rows(base_by_source, augmented_rows)
    summary: dict[str, Any] = {
        "schema": "x2cbct_luna16_augmentation_robustness_analysis.v1",
        "base_run": str(base_run),
        "augmented_run": str(augmented_run),
        "counts": {
            "source_cases": len(base_by_source),
            "base_samples": len(base_rows),
            "augmented_samples": len(augmented_rows),
            "augmentations_per_source": unique_counts[0],
        },
        "run_summaries": {
            "base": _read_json(base_run / "metrics_summary.json"),
            "augmented": _read_json(augmented_run / "metrics_summary.json"),
        },
        "paired_metric_deltas": _paired_summary(base_by_source, augmented_rows),
        "factor_contrasts": _factor_contrasts(base_by_source, augmented_rows),
        "case_difficulty": _case_difficulty(base_by_source, augmented_rows),
        "by_augmentation": by_augmentation,
    }

    if not args.skip_volume_analysis:
        summary["volume_analysis"] = _volume_analysis(
            base_manifest["predictions"],
            augmented_manifest["predictions"],
            threshold=args.foreground_threshold,
        )
        requested_ids = {
            item.strip() for item in str(args.equivariance_augmentation_ids).split(",") if item.strip()
        }
        summary["equivariance_analysis"] = _equivariance_analysis(
            base_manifest["predictions"],
            augmented_manifest["predictions"],
            augmentation_ids=requested_ids,
            output_spacing_z_mm=args.output_spacing_z_mm,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary, output_dir / "analysis_summary.json")
    _write_csv(by_augmentation, output_dir / "metrics_by_augmentation.csv")
    _write_csv(summary["factor_contrasts"], output_dir / "factor_contrasts.csv")
    if "volume_analysis" in summary:
        _write_csv(
            summary["volume_analysis"]["by_augmentation"],
            output_dir / "foreground_by_augmentation.csv",
        )
        _write_csv(
            summary["equivariance_analysis"]["by_augmentation"],
            output_dir / "equivariance_by_augmentation.csv",
        )
    _write_markdown(summary, output_dir / "analysis_report.md")
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(output_dir),
                "source_cases": len(base_by_source),
                "augmented_samples": len(augmented_rows),
                "volume_analysis": not args.skip_volume_analysis,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
