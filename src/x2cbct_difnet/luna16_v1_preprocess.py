from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .cache import read_json, write_json


LUNA16_REGISTRY_SCHEMA = "x2cbct_luna16_v1_fixed_manifest_registry.v1"
STRIPPED_KEYS = {"upstream_split", "split_source", "split_policy"}


def _clean_case(row: dict[str, Any], *, output_split: str) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if key not in STRIPPED_KEYS}
    case_key = str(out.get("case_key") or out.get("seriesuid") or out.get("patient_id") or out.get("case_id"))
    out.setdefault("case_key", case_key)
    out.setdefault("seriesuid", str(out.get("seriesuid") or case_key))
    out.setdefault("patient_id", str(out.get("patient_id") or out["seriesuid"]))
    out["split"] = output_split
    return out


def _cases_from_manifest(payload: dict[str, Any], *, source_split: str) -> list[dict[str, Any]]:
    if isinstance(payload.get("cases"), list):
        return list(payload["cases"])
    split_payload = payload.get("splits", {}).get(source_split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"manifest has no cases and no split {source_split!r}")
    return list(split_payload.get("cases", []))


def materialize_luna16_registry(
    *,
    manifest_path: Path,
    output_path: Path,
    source_split: str,
    output_split: str,
    root: Path,
    command: str = "",
) -> dict[str, Any]:
    payload = read_json(manifest_path)
    dataset = payload.get("dataset")
    if dataset is not None and dataset != "luna16":
        raise ValueError(f"expected luna16 manifest, got {dataset!r}")
    if output_split not in {"train", "test"}:
        raise ValueError("output_split must be train or test")
    cases = [_clean_case(row, output_split=output_split) for row in _cases_from_manifest(payload, source_split=source_split)]
    registry = {
        "schema": LUNA16_REGISTRY_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_command": command,
        "source_manifest": str(manifest_path.resolve()),
        "root": str(root.resolve()),
        "split": {
            "policy": "fixed_difnet_v1_manifest",
            "source_split": source_split,
            "output_split": output_split,
            "train_case_count": sum(1 for row in cases if row["split"] == "train"),
            "test_case_count": sum(1 for row in cases if row["split"] == "test"),
            "case_overlap_count": 0,
            "patient_overlap": [],
        },
        "cases": cases,
    }
    write_json(registry, output_path)
    return registry
