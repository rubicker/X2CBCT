from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _augmentation_id(sample: dict[str, Any]) -> str | None:
    setup = sample.get("setup_augmentation")
    if isinstance(setup, dict) and setup.get("augmentation_id") is not None:
        return str(setup["augmentation_id"])
    if sample.get("augmentation_id") is not None:
        return str(sample["augmentation_id"])
    return None


def _iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _resolve_cache_path(cache_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if not any(token in value for token in (".npz", ".npy", ".json", ".nii", ".gz")):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = cache_root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(cache_root.resolve())
    except ValueError:
        return None
    return resolved


def _referenced_paths(cache_root: Path, samples: list[dict[str, Any]]) -> set[Path]:
    paths: set[Path] = set()
    for sample in samples:
        for value in _iter_values(sample):
            path = _resolve_cache_path(cache_root, value)
            if path is not None:
                paths.add(path)
    return paths


def _asset_dir(cache_root: Path, path: Path) -> Path | None:
    rel = path.relative_to(cache_root.resolve()).parts
    if len(rel) >= 3 and rel[0] in {"cases", "projections"}:
        return cache_root.resolve() / rel[0] / rel[1]
    return None


def _move_path(path: Path, cache_root: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    rel = path.relative_to(cache_root.resolve())
    destination = backup_root / "assets" / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(path), str(destination))


def _refresh_manifest_counts(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    views = sorted({str(sample.get("view_key")) for sample in samples if sample.get("view_key")})
    splits = sorted({str(sample.get("split")) for sample in samples if sample.get("split")})
    if "view_sample_counts" in manifest:
        manifest["view_sample_counts"] = {
            view: {split: sum(1 for sample in samples if sample.get("view_key") == view and sample.get("split") == split) for split in splits}
            for view in views
        }
    if isinstance(manifest.get("counts"), dict):
        case_rows = {
            (str(sample.get("split")), str(sample.get("case_id") or sample.get("case_key") or sample.get("sample_id")))
            for sample in samples
        }
        manifest["counts"]["cases"] = len(case_rows)
        manifest["counts"]["split"] = {
            split: sum(1 for row_split, _ in case_rows if row_split == split)
            for split in splits
        }
        manifest["counts"]["samples"] = len(samples)


def remove_augmentation_id(
    cache_root: Path,
    augmentation_id: str,
    *,
    dry_run: bool = False,
    backup_parent: Path | None = None,
) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    manifest_path = cache_root / "cache_manifest.json"
    manifest = _read_json(manifest_path)
    samples = list(manifest.get("samples", []))
    removed = [sample for sample in samples if _augmentation_id(sample) == augmentation_id]
    kept = [sample for sample in samples if _augmentation_id(sample) != augmentation_id]
    removed_refs = _referenced_paths(cache_root, removed)
    kept_refs = _referenced_paths(cache_root, kept)
    removable_files = sorted(path for path in removed_refs - kept_refs if path.exists())
    removable_dirs = sorted(
        {
            asset
            for path in removable_files
            for asset in [_asset_dir(cache_root, path)]
            if asset is not None and not any(ref != path and str(ref).startswith(str(asset)) for ref in kept_refs)
        },
        key=lambda item: len(item.parts),
        reverse=True,
    )
    if dry_run:
        return {
            "cache_root": str(cache_root),
            "augmentation_id": augmentation_id,
            "dry_run": True,
            "removed_sample_count": len(removed),
            "kept_sample_count": len(kept),
            "movable_file_count": len(removable_files),
            "movable_asset_dir_count": len(removable_dirs),
        }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_base = Path(backup_parent).resolve() if backup_parent is not None else cache_root / "filter_backups"
    backup_root = backup_base / f"remove_{augmentation_id}_{stamp}"
    (backup_root / "manifest").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, backup_root / "manifest" / "cache_manifest.json")
    for path in removable_files:
        if not any(str(path).startswith(str(directory)) for directory in removable_dirs):
            _move_path(path, cache_root, backup_root)
    for directory in removable_dirs:
        _move_path(directory, cache_root, backup_root)
    manifest["samples"] = kept
    _refresh_manifest_counts(manifest, kept)
    setup = manifest.get("setup_augmentation")
    if isinstance(setup, dict):
        excluded = list(setup.get("excluded_augmentation_ids", []))
        if augmentation_id not in excluded:
            excluded.append(augmentation_id)
        setup["excluded_augmentation_ids"] = excluded
        setup["active_combination_count_per_source_case"] = max(
            0,
            int(setup.get("combination_count_per_source_case", 0)) - len(excluded),
        )
    manifest.setdefault("filter_history", []).append(
        {
            "timestamp": stamp,
            "operation": "remove_augmentation_id",
            "augmentation_id": augmentation_id,
            "removed_sample_count": len(removed),
            "backup_root": str(backup_root),
        }
    )
    _write_json(manifest_path, manifest)
    return {
        "cache_root": str(cache_root),
        "augmentation_id": augmentation_id,
        "dry_run": False,
        "removed_sample_count": len(removed),
        "kept_sample_count": len(kept),
        "moved_file_count": len(removable_files),
        "moved_asset_dir_count": len(removable_dirs),
        "backup_root": str(backup_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove samples with one setup augmentation id from a DIF-Net cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--augmentation-id", default="si_0_rot_0")
    parser.add_argument("--backup-parent", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = remove_augmentation_id(
        args.cache_root,
        args.augmentation_id,
        dry_run=args.dry_run,
        backup_parent=args.backup_parent,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
