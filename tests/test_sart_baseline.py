from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.run_table1_sart_baselines import table1_sart_tasks
from x2cbct_difnet.sart_baseline import SART_NORMALIZATION_METHOD, normalize_sart_output


def test_sart_output_normalization_uses_only_sart_values() -> None:
    raw = np.linspace(-20.0, 40.0, 1000, dtype=np.float32).reshape(10, 10, 10)

    normalized, metadata = normalize_sart_output(raw)

    assert normalized.shape == raw.shape
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == pytest.approx(0.0)
    assert float(normalized.max()) == pytest.approx(1.0)
    assert metadata["method"] == SART_NORMALIZATION_METHOD
    assert metadata["uses_target_volume"] is False
    assert metadata["low"] == pytest.approx(float(np.percentile(raw, 0.5)))
    assert metadata["high"] == pytest.approx(float(np.percentile(raw, 99.5)))


def test_table1_sart_tasks_are_the_confirmed_noaug_matrix() -> None:
    tasks = table1_sart_tasks()

    assert [(task.dataset, task.view_count, task.output_name) for task in tasks] == [
        ("luna16", 2, "luna16_view2"),
        ("luna16", 5, "luna16_view5"),
        ("luna16", 10, "luna16_view10"),
        ("self_projection1208_mixed", 2, "self_projection1208_mixed_view2"),
        ("self_projection1208_mixed", 5, "self_projection1208_mixed_view5"),
        ("self_projection1208_mixed", 10, "self_projection1208_mixed_view10"),
    ]
    assert all("test_noaug_256" in str(task.cache_root) for task in tasks[:3])
    assert all("test_mixed_noaug_256" in str(task.cache_root) for task in tasks[3:])
    assert all("test_patient" not in str(task.cache_root) for task in tasks)
