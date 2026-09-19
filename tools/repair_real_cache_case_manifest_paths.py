from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x2cbct_difnet.cache import resolve_cache_path


VIEW_SUFFIXES = ("view_2", "view_5", "view_10")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def case_key_from_sample_id(sample_id: str) -> str:
    for view in VIEW_SUFFIXES:
        suffix = f"_{view}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    raise ValueError(f"cannot infer case key from sample_id: {sample_id}")


def resolve_path(cache_root: Path, value: str) -> Path:
    return resolve_cache_path(cache_root, value)


def target_manifest_path(cache_root: Path, sample: dict[str, Any], source_root: Path | None) -> str:
    sample_id = str(sample["sample_id"])
    case_key = case_key_from_sample_id(sample_id)
    root = source_root or cache_root
    candidate = (root / "cases" / case_key / "case_manifest.json").resolve()
    if source_root is None:
        return candidate.relative_to(cache_root.resolve()).as_posix()
    return str(candidate)


def backup_file(path: Path, cache_root: Path, backup_root: Path) -> None:
    relative = path.resolve().relative_to(cache_root.resolve())
    backup_path = backup_root / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)


def repair_npz(path: Path, new_manifest_path: str, *, apply: bool) -> bool:
    with np.load(path, allow_pickle=False) as raw:
        old_manifest_path = str(raw["case_manifest_path"].item())
        if old_manifest_path == new_manifest_path:
            return False
        payload = {key: raw[key] for key in raw.files}
    payload["case_manifest_path"] = np.array(new_manifest_path)
    if apply:
        np.savez_compressed(path, **payload)
    return True


def repair_cache(cache_root: Path, *, apply: bool) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    manifest_path = cache_root / "cache_manifest.json"
    manifest = read_json(manifest_path)
    source_root = None
    derived = manifest.get("derived_training_cache")
    if isinstance(derived, dict) and derived.get("source_cache_root"):
        source_root = Path(str(derived["source_cache_root"])).resolve()

    backup_root = cache_root / "repair_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    changed_manifest_samples = 0
    changed_npz = 0
    missing_targets: list[dict[str, str]] = []
    checked_samples = 0

    samples = manifest.get("samples", [])
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("geometry_branch", "real_projection") != "real_projection":
            continue
        if "sample_id" not in sample or "sample_path" not in sample:
            continue
        checked_samples += 1
        new_manifest_path = target_manifest_path(cache_root, sample, source_root)
        target_path = resolve_path(source_root or cache_root, new_manifest_path)
        if not target_path.exists():
            missing_targets.append({"sample_id": str(sample["sample_id"]), "expected": str(target_path)})
            continue
        if str(sample.get("case_manifest_path")) != new_manifest_path:
            sample["case_manifest_path"] = new_manifest_path
            changed_manifest_samples += 1
        npz_path = resolve_path(cache_root, str(sample["sample_path"]))
        if not npz_path.exists():
            missing_targets.append({"sample_id": str(sample["sample_id"]), "expected": str(npz_path)})
            continue
        if apply:
            backup_file(npz_path, cache_root, backup_root)
        if repair_npz(npz_path, new_manifest_path, apply=apply):
            changed_npz += 1

    if missing_targets:
        return {
            "ok": False,
            "cache_root": str(cache_root),
            "checked_samples": checked_samples,
            "changed_manifest_samples": changed_manifest_samples,
            "changed_npz": changed_npz,
            "missing_targets": missing_targets[:20],
            "missing_target_count": len(missing_targets),
        }

    if apply and changed_manifest_samples:
        backup_file(manifest_path, cache_root, backup_root)
        write_json(manifest_path, manifest)

    return {
        "ok": True,
        "cache_root": str(cache_root),
        "checked_samples": checked_samples,
        "changed_manifest_samples": changed_manifest_samples,
        "changed_npz": changed_npz,
        "backup_root": str(backup_root) if apply and (changed_manifest_samples or changed_npz) else None,
        "applied": apply,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair real base_reference cache case_manifest_path indexes in place.")
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    reports = [repair_cache(path, apply=args.apply) for path in args.cache_root]
    print(json.dumps({"ok": all(item["ok"] for item in reports), "reports": reports}, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
