from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.normalization import VIEW_TARGETS
from x2cbct_difnet.v1_training_cache import VIEW_SUBSET_INDICES


SOURCE_VIEW = "view_10"
TARGET_VIEW = "view_8"
TARGET_COUNT = 8


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_cache_roots() -> list[Path]:
    root = PROJECT_ROOT / "data_cache" / "difnet_v1_si10_rot10_3x3_256"
    return [
        root / "self_projection1208" / "train_mixed_noaug_256",
        root / "self_projection1208" / "train_mixed_aug3x3_256",
        root / "self_projection1208" / "test_mixed_noaug_256",
        root / "self_projection1208" / "test_mixed_aug3x3_256",
        root / "luna16" / "train_noaug_256",
        root / "luna16" / "train_aug3x3_256",
        root / "luna16" / "test_noaug_256" / "luna16",
        root / "luna16" / "test_aug3x3_256" / "luna16",
    ]


def _backup_manifest(manifest_path: Path, backup_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    relative = manifest_path.resolve().relative_to(PROJECT_ROOT.resolve())
    backup_path = backup_root / stamp / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def _replace_view_suffix(value: str) -> str:
    suffix = f"_{SOURCE_VIEW}"
    if not str(value).endswith(suffix):
        raise ValueError(f"sample id does not end with {suffix}: {value}")
    return str(value)[: -len(suffix)] + f"_{TARGET_VIEW}"


def _subset_list(values: list[Any]) -> list[Any]:
    return [values[index] for index in VIEW_SUBSET_INDICES[TARGET_VIEW]]


def _subset_npz_value(value: np.ndarray) -> np.ndarray:
    if value.ndim == 0:
        return value
    if len(value) == len(VIEW_SUBSET_INDICES[SOURCE_VIEW]):
        return value[VIEW_SUBSET_INDICES[TARGET_VIEW]]
    return value


def _copy_real_npz(source_path: Path, output_path: Path, *, sample_id: str, split: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=False) as raw:
        payload = {key: raw[key] for key in raw.files}
        payload["sample_id"] = np.array(sample_id)
        payload["split"] = np.array(split)
        payload["view_key"] = np.array(TARGET_VIEW)
        for key in ("projection_payload_paths", "projection_files", "target_angles_deg", "gantry_angles_deg", "angular_residuals_deg"):
            if key in payload:
                payload[key] = _subset_npz_value(payload[key])
        np.savez_compressed(output_path, **payload)


def _view_values_for_manifest(manifest: dict[str, Any]) -> Any:
    if str(manifest.get("schema")) == "x2cbct_difnet_v0b_cache.v2":
        return [float(item) for item in VIEW_TARGETS[TARGET_VIEW]]
    return TARGET_VIEW


def _set_manifest_views(manifest: dict[str, Any]) -> None:
    views = manifest.get("views")
    value = _view_values_for_manifest(manifest)
    if isinstance(views, dict):
        views[TARGET_VIEW] = value
    elif isinstance(views, list):
        if TARGET_VIEW not in views:
            views.append(TARGET_VIEW)
    else:
        raise ValueError(f"unsupported manifest views field: {type(views).__name__}")
    if isinstance(manifest.get("view_counts"), dict):
        manifest["view_counts"][TARGET_VIEW] = TARGET_COUNT


def _sample_split(sample: dict[str, Any]) -> str:
    split = str(sample.get("split") or "")
    if split:
        return split
    sample_path = str(sample.get("sample_path") or "")
    if "/train/" in sample_path.replace("\\", "/"):
        return "train"
    if "/test/" in sample_path.replace("\\", "/"):
        return "test"
    raise ValueError(f"cannot determine sample split: {sample.get('sample_id')}")


def _real_view8_sample(cache_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    sample = copy.deepcopy(source)
    split = _sample_split(source)
    sample_id = _replace_view_suffix(str(source["sample_id"]))
    sample_path = Path(TARGET_VIEW) / split / f"{sample_id}.npz"
    source_npz = cache_root / str(source["sample_path"])
    output_npz = cache_root / sample_path
    _copy_real_npz(source_npz, output_npz, sample_id=sample_id, split=split)

    sample["sample_id"] = sample_id
    sample["view_key"] = TARGET_VIEW
    sample["sample_path"] = sample_path.as_posix()
    sample["projection_payload_paths"] = _subset_list(list(source.get("projection_payload_paths", [])))
    if "selected_projections" in sample:
        sample["selected_projections"] = _subset_list(list(source.get("selected_projections", [])))
    selected = sample.get("selected_projections") or []
    if selected:
        sample["projection_geometry"] = {
            **dict(sample.get("projection_geometry", {})),
            "gantry_angles_deg": [float(item["gantry_angle_deg"]) for item in selected],
            "selected_projection_files": [str(item["file"]) for item in selected],
        }
    return sample


def _luna_view8_sample(source: dict[str, Any]) -> dict[str, Any]:
    sample = copy.deepcopy(source)
    sample["sample_id"] = _replace_view_suffix(str(source["sample_id"]))
    sample["view_key"] = TARGET_VIEW
    sample["view_count"] = TARGET_COUNT
    sample["view_indices"] = list(VIEW_SUBSET_INDICES[TARGET_VIEW])
    sample["projection_paths"] = _subset_list(list(source.get("projection_paths", [])))
    sample["angles_deg"] = _subset_list(list(source.get("angles_deg", [])))
    return sample


def _refresh_counts(manifest: dict[str, Any]) -> None:
    samples = list(manifest.get("samples", []))
    if isinstance(manifest.get("view_sample_counts"), dict):
        splits = sorted({str(sample.get("split")) for sample in samples if sample.get("split")})
        manifest["view_sample_counts"][TARGET_VIEW] = {
            split: sum(1 for sample in samples if sample.get("view_key") == TARGET_VIEW and sample.get("split") == split)
            for split in splits
        }
    if isinstance(manifest.get("counts"), dict):
        manifest["counts"]["samples"] = len(samples)
        if isinstance(manifest["counts"].get("split"), dict):
            for split in list(manifest["counts"]["split"]):
                manifest["counts"]["split"][split] = sum(
                    1
                    for sample in samples
                    if sample.get("view_key") == SOURCE_VIEW and sample.get("split") == split
                )
    if isinstance(manifest.get("derived_training_cache"), dict):
        indices = manifest["derived_training_cache"].setdefault("view_subset_indices", {})
        indices[TARGET_VIEW] = list(VIEW_SUBSET_INDICES[TARGET_VIEW])


def append_view8(cache_root: Path, *, backup_root: Path, overwrite: bool) -> dict[str, Any]:
    root = cache_root.resolve()
    manifest_path = root / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    samples = list(manifest.get("samples", []))
    existing = [sample for sample in samples if sample.get("view_key") == TARGET_VIEW]
    if existing and not overwrite:
        return {
            "cache_root": str(root),
            "status": "skipped_existing",
            "existing_view8_samples": len(existing),
        }
    if overwrite:
        samples = [sample for sample in samples if sample.get("view_key") != TARGET_VIEW]
        view_dir = root / TARGET_VIEW
        if view_dir.exists():
            shutil.rmtree(view_dir)

    source_samples = [sample for sample in samples if sample.get("view_key") == SOURCE_VIEW]
    if not source_samples:
        raise ValueError(f"{root} has no {SOURCE_VIEW} samples")
    schema = str(manifest.get("schema"))
    if schema == "x2cbct_difnet_v0b_cache.v2":
        generated = [_real_view8_sample(root, sample) for sample in source_samples]
    elif schema == "x2cbct_synthrad2025_task2_view_cache.v1":
        generated = [_luna_view8_sample(sample) for sample in source_samples]
    else:
        raise ValueError(f"unsupported cache schema: {schema}")

    backup_path = _backup_manifest(manifest_path, backup_root)
    manifest["samples"] = [*samples, *generated]
    _set_manifest_views(manifest)
    manifest["generated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["view8_append"] = {
        "schema": "x2cbct_append_view8_from_view10.v1",
        "generated_at": manifest["generated_at"],
        "source_view": SOURCE_VIEW,
        "target_view": TARGET_VIEW,
        "target_angles_deg": list(VIEW_TARGETS[TARGET_VIEW]),
        "source_indices": list(VIEW_SUBSET_INDICES[TARGET_VIEW]),
        "backup_manifest": str(backup_path),
        "writes_body_or_projection_assets": False,
    }
    _refresh_counts(manifest)
    _write_json(manifest, manifest_path)
    return {
        "cache_root": str(root),
        "status": "appended",
        "source_view10_samples": len(source_samples),
        "new_view8_samples": len(generated),
        "backup_manifest": str(backup_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a formal view_8 branch by subsetting existing view_10 cache samples.")
    parser.add_argument("--cache-root", type=Path, action="append", help="Cache root containing cache_manifest.json. Repeatable.")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=PROJECT_ROOT / "data_cache" / "_view8_append_backups",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = args.cache_root or _default_cache_roots()
    results = [
        append_view8(root, backup_root=args.backup_root.resolve(), overwrite=args.overwrite)
        for root in roots
    ]
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
