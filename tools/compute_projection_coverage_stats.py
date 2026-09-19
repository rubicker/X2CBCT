"""Projection-coverage statistics for the self_projection1208 real cache.

For each test sample and each of view_2 / view_5 / view_10, materialize the
official point block used for training/eval and run the ClinicalGeometryAdapter
to compute, per 3D query point, the per-view valid_mask (whether the projected
2D detector coordinate falls inside the projection image). Then aggregate:

  - per-view invalid ratio: fraction of query points invalid at a given view
  - all-views-invalid ratio: fraction of query points whose projected feature
    falls OUTSIDE the detector for *every* view (i.e. the query point has no
    in-field view at all). This is the paper's "projection feature falls
    outside the projection image" statistic under the sparse-view protocol.
  - per-view-pairwise coverage breakdown for context.

This is a pure geometry statistic; no model weights or training metrics are
touched. Results are written to JSON under runs/audits/projection_coverage/.

Usage (PowerShell):
  & D:\code\X2CBCT\.conda\x2cbct-v0\python.exe `
    D:\code\X2CBCT\tools\compute_projection_coverage_stats.py `
    --cache-root `
    D:\code\X2CBCT\data_cache\difnet_v1_si10_rot10_3x3_256\self_projection1208\test_mixed_noaug_256 `
    --split test `
    --views view_2 view_5 view_10 `
    --eval-grid-size 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cached_dataset import CachedDifNetV0bDataset  # noqa: E402


def _summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def run_one(cache_root: str, split: str, view_key: str, eval_grid_size: int, max_cases: int | None) -> dict:
    ds = CachedDifNetV0bDataset(
        cache_root=cache_root,
        split=split,
        view_key=view_key,
        num_points=10000,  # ignored when eval_grid_size is set
        eval_grid_size=eval_grid_size,
        max_cases=max_cases,
    )
    # n_views is only known after loading a sample's npz, so peek at item 0.
    if len(ds.samples) == 0:
        return {"cache_root": str(Path(cache_root).resolve()), "split": split, "view_key": view_key, "n_cases": 0}
    first = ds[0]
    n_views = int(first["valid_mask"].shape[0])
    per_view_invalid_ratios = []
    all_invalid_ratios = []
    all_invalid_counts = []
    total_points = 0
    total_all_invalid = 0
    per_view_total_invalid = np.zeros(n_views, dtype=np.int64)
    case_details = []
    for i in range(len(ds)):
        item = ds[i]
        vm = item["valid_mask"]  # shape [V, N], bool
        n_v, n_p = vm.shape
        if n_v != n_views:
            raise RuntimeError(f"view count mismatch at case {i}: {n_v} vs {n_views}")
        per_view_invalid = (~vm).mean(axis=1)  # [V]
        all_invalid = (~vm).all(axis=0)  # [N]
        all_invalid_count = int(all_invalid.sum())
        all_invalid_ratio = float(all_invalid.mean())
        per_view_invalid_ratios.append(per_view_invalid)
        all_invalid_ratios.append(all_invalid_ratio)
        all_invalid_counts.append(all_invalid_count)
        total_points += n_p
        total_all_invalid += all_invalid_count
        per_view_total_invalid += (~vm).sum(axis=1)
        case_details.append(
            {
                "sample_id": item["name"],
                "view_key": view_key,
                "n_views": n_v,
                "n_points": n_p,
                "all_invalid_count": all_invalid_count,
                "all_invalid_ratio": all_invalid_ratio,
                "per_view_invalid_ratio": [float(x) for x in per_view_invalid.tolist()],
            }
        )

    per_view_invalid_arr = np.stack(per_view_invalid_ratios, axis=0) if per_view_invalid_ratios else np.zeros((0, n_views))
    overall_all_invalid_ratio = (total_all_invalid / total_points) if total_points else 0.0
    overall_per_view_invalid_ratio = (per_view_total_invalid / total_points).tolist() if total_points else []
    return {
        "cache_root": str(Path(cache_root).resolve()),
        "split": split,
        "view_key": view_key,
        "n_views": int(n_views),
        "n_cases": len(ds.samples),
        "eval_grid_size": eval_grid_size,
        "total_query_points": int(total_points),
        "total_all_invalid_points": int(total_all_invalid),
        "overall_all_invalid_ratio": overall_all_invalid_ratio,
        "overall_per_view_invalid_ratio": [float(x) for x in overall_per_view_invalid_ratio],
        "all_invalid_ratio_per_case": _summarize(all_invalid_ratios),
        "overall_all_invalid_ratio_pct": overall_all_invalid_ratio * 100,
        "per_view_invalid_ratio_summary": [
            {
                "view_index": v,
                "overall_ratio": float(overall_per_view_invalid_ratio[v]) if total_points else 0.0,
                "overall_ratio_pct": (per_view_total_invalid[v] / total_points * 100) if total_points else 0.0,
                "per_case_summary": _summarize(per_view_invalid_arr[:, v]),
            }
            for v in range(n_views)
        ],
        "per_view_all_invalid_intersection_pct": _pairwise_all_invalid(per_view_invalid_arr),
        "case_details": case_details,
    }


def _pairwise_all_invalid(arr: np.ndarray) -> dict | None:
    if arr.size == 0:
        return None
    return {"shape": list(arr.shape)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--views", nargs="+", default=["view_2", "view_5", "view_10"])
    ap.add_argument("--eval-grid-size", type=int, default=128)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument(
        "--output",
        default=None,
        help="output json path; default runs/audits/projection_coverage/<cache_name>_<split>.json",
    )
    args = ap.parse_args()

    results = {}
    for vk in args.views:
        print(f"[{vk}] computing coverage on {args.cache_root} split={args.split} grid={args.eval_grid_size}", flush=True)
        results[vk] = run_one(args.cache_root, args.split, vk, args.eval_grid_size, args.max_cases)
        r = results[vk]
        print(
            f"  cases={r['n_cases']} points/case={r['total_query_points']//max(r['n_cases'],1)} "
            f"overall_all_invalid={r['overall_all_invalid_ratio_pct']:.3f}% "
            f"per_view_invalid%={[round(x*100,3) for x in r['overall_per_view_invalid_ratio']]}",
            flush=True,
        )

    out_dir = Path("D:/code/X2CBCT/runs/audits/projection_coverage")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_name = Path(args.cache_root).name
    out_path = Path(args.output) if args.output else out_dir / f"{cache_name}_{args.split}_grid{args.eval_grid_size}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())