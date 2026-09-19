from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.luna16_v1_preprocess import materialize_luna16_registry


def test_materialize_luna16_registry_strips_upstream_split_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "luna16_noaug.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "x2cbct_difnet_v1_derivative_manifest.v1",
                "name": "luna16_noaug",
                "dataset": "luna16",
                "splits": {
                    "train": {"cases": []},
                    "test": {
                        "cases": [
                            {
                                "case_key": "case_a",
                                "seriesuid": "case_a",
                                "patient_id": "case_a",
                                "ct_path": "subset/case_a.mhd",
                                "upstream_split": "eval",
                                "split_source": "splits.json",
                                "split_policy": "old_policy",
                            }
                        ]
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    registry = materialize_luna16_registry(
        manifest_path=manifest_path,
        output_path=tmp_path / "registry.json",
        source_split="test",
        output_split="test",
        root=tmp_path / "LUNA16",
        command="pytest",
    )

    assert registry["split"]["test_case_count"] == 1
    assert registry["cases"][0]["split"] == "test"
    serialized = json.dumps(registry, ensure_ascii=False)
    assert "eval" not in serialized
    assert "upstream_split" not in serialized
    assert "split_source" not in serialized
    assert "old_policy" not in serialized
