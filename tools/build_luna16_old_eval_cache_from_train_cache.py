from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.v1_training_cache import VIEW_SUBSET_INDICES


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seriesuid(row: dict[str, Any]) -> str:
    for key in ("seriesuid", "source_case_id", "patient_id", "case_id", "case_key"):
        value = row.get(key)
        if value:
            text = str(value)
            if text.startswith("subset") and "_" in text:
                maybe_subset, rest = text.split("_", 1)
                if _remove_prefix(maybe_subset, "subset").isdigit():
                    return rest
            return _remove_prefix(text, "luna16_")
    raise ValueError(f"cannot derive seriesuid from row: {row}")


def _sample_source_case_id(sample: dict[str, Any]) -> str:
    for key in ("source_case_id", "case_id", "case_key"):
        value = sample.get(key)
        if value:
            text = _remove_prefix(str(value), "luna16_")
            marker = "_si_"
            if marker in text:
                text = text.split(marker, 1)[0]
            return text
    raise ValueError(f"cannot derive source case id from sample: {sample.get('sample_id')}")


def _view_sample_counts(samples: list[dict[str, Any]], views: tuple[str, ...], split: str) -> dict[str, dict[str, int]]:
    return {
        view: {
            "train": sum(1 for item in samples if item["view_key"] == view and item["split"] == "train"),
            "eval": sum(1 for item in samples if item["view_key"] == view and item["split"] == "eval"),
            "test": sum(1 for item in samples if item["view_key"] == view and item["split"] == "test"),
            split: sum(1 for item in samples if item["view_key"] == view and item["split"] == split),
        }
        for view in views
    }


def build_old_eval_cache(
    *,
    source_cache_root: Path,
    registry_path: Path,
    output_root: Path,
    views: tuple[str, ...],
    output_split: str,
    overwrite: bool,
    command: str,
) -> dict[str, Any]:
    source_cache_root = source_cache_root.resolve()
    registry_path = registry_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest = _read_json(source_cache_root / "cache_manifest.json")
    registry = _read_json(registry_path)
    eval_cases = {
        _seriesuid(row)
        for row in registry.get("cases", [])
        if str(row.get("split", "")).strip().lower() == "eval"
    }
    if not eval_cases:
        raise ValueError(f"no eval cases found in registry: {registry_path}")

    selected: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for sample in source_manifest.get("samples", []):
        if sample.get("view_key") not in views:
            continue
        if str(sample.get("split")) != "train":
            continue
        source_case_id = _sample_source_case_id(sample)
        if source_case_id not in eval_cases:
            continue
        copied = dict(sample)
        copied["split"] = output_split
        copied["old_luna16_split"] = "eval"
        copied["source_cache_root"] = str(source_cache_root)
        selected.append(copied)
        seen_cases.add(source_case_id)

    missing = sorted(eval_cases - seen_cases)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{len(missing)} old eval cases are absent from source cache: {preview}")

    case_ids = sorted({_sample_source_case_id(item) for item in selected})
    manifest = dict(source_manifest)
    manifest.update(
        {
            "cache_version": output_root.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_command": command,
            "views": list(views),
            "view_counts": {view: len(VIEW_SUBSET_INDICES[view]) for view in views},
            "test_is_validation": output_split in {"eval", "test"},
            "split_policy": "temporary_old_eval_replay",
            "counts": {
                "cases": len(case_ids),
                "samples": len(selected),
                "split": {output_split: len(case_ids)},
            },
            "split": {
                "policy": "temporary_old_eval_replay",
                "split_unit": "case",
                "source_registry": str(registry_path),
                "source_old_split": "eval",
                "output_split": output_split,
                "case_count": len(case_ids),
                "case_overlap_count": 0,
                "has_validation_split": output_split == "eval",
            },
            "derived_eval_cache": {
                "source_cache_root": str(source_cache_root),
                "source_cache_manifest": str((source_cache_root / "cache_manifest.json").resolve()),
                "source_registry": str(registry_path),
                "source_old_split": "eval",
                "asset_reference_mode": "absolute_paths",
                "writes_body_or_projection_assets": False,
                "view_subset_indices": {key: VIEW_SUBSET_INDICES[key] for key in views},
            },
            "view_sample_counts": _view_sample_counts(selected, views, output_split),
            "samples": selected,
        }
    )
    _write_json(manifest, output_root / "cache_manifest.json")
    _write_json(
        {
            "ok": True,
            "cache_root": str(output_root),
            "source_cache_root": str(source_cache_root),
            "registry": str(registry_path),
            "old_eval_case_count": len(case_ids),
            "sample_count": len(selected),
            "output_split": output_split,
            "views": list(views),
            "writes_body_or_projection_assets": False,
        },
        output_root / "reports" / "cache_summary.json",
    )
    return manifest


def _parse_views(raw: str) -> tuple[str, ...]:
    views: list[str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token in {"2", "5", "10"}:
            token = f"view_{token}"
        if token not in VIEW_SUBSET_INDICES:
            raise ValueError(f"unsupported view: {token}")
        views.append(token)
    return tuple(dict.fromkeys(views))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a lightweight temporary LUNA16 old-eval replay cache from an existing noaug training cache."
    )
    parser.add_argument(
        "--source-cache-root",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "luna16" / "train_noaug_256",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "luna16" / "temp_old_eval49_noaug_256",
    )
    parser.add_argument("--views", default="2,5,10")
    parser.add_argument("--output-split", choices=("eval", "test"), default="eval")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_old_eval_cache(
        source_cache_root=args.source_cache_root,
        registry_path=args.registry,
        output_root=args.output_root,
        views=_parse_views(args.views),
        output_split=args.output_split,
        overwrite=args.overwrite,
        command=" ".join(sys.argv),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "cache_root": str(args.output_root.resolve()),
                "counts": manifest.get("counts"),
                "view_sample_counts": manifest.get("view_sample_counts"),
                "output_split": args.output_split,
                "writes_body_or_projection_assets": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
