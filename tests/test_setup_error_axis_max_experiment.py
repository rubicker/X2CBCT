from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from run_setup_error_axis_max_experiment import (
    FocusPair,
    _attach_grid_control,
    _configured_backend,
    _configured_view_count,
    _list_checkpoints,
    _order_cache_roots_for_split,
    _parse_pair_spec,
    _select_axis_max_pair,
    _selected_axes,
)
from run_setup_error_focus_experiment import _sample_augmentation_id


def _formal_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        ("10", "0", "4"): {
            "status": "ok",
            "setup_error_SI_mm": "6.8",
            "setup_error_abs_SI_mm": "6.8",
            "setup_error_yaw_deg": "-1.6",
            "setup_error_abs_yaw_deg": "1.6",
        },
        ("24", "2", "8"): {
            "status": "ok",
            "setup_error_SI_mm": "-0.1",
            "setup_error_abs_SI_mm": "0.1",
            "setup_error_yaw_deg": "7.8",
            "setup_error_abs_yaw_deg": "7.8",
        },
        ("99", "0", "1"): {
            "status": "failed",
            "setup_error_SI_mm": "100",
            "setup_error_abs_SI_mm": "100",
            "setup_error_yaw_deg": "100",
            "setup_error_abs_yaw_deg": "100",
        },
    }


def test_selects_independent_axis_max_pairs() -> None:
    rows = _formal_rows()

    assert _select_axis_max_pair(rows, "si-max") == FocusPair("10", "0", "4")
    assert _select_axis_max_pair(rows, "yaw-max") == FocusPair("24", "2", "8")


def test_parse_pair_spec_uses_patient_reference_moving_order() -> None:
    assert _parse_pair_spec("24:2:8") == FocusPair("24", "2", "8")
    assert _parse_pair_spec(
        "4:13:13:identity:si_p10_rot_0"
    ) == FocusPair("4", "13", "13", "identity", "si_p10_rot_0")
    assert (
        _parse_pair_spec("4:13:13:identity:si_p10_rot_0").pair_key
        == "patient4_case13_si_p10_rot_0_vs_case13"
    )

    with pytest.raises(Exception, match="PATIENT:REFERENCE:MOVING"):
        _parse_pair_spec("24:8")


def test_sample_augmentation_id_distinguishes_identity_and_exact_variant() -> None:
    assert _sample_augmentation_id({}) == "identity"
    assert (
        _sample_augmentation_id(
            {"setup_augmentation": {"augmentation_id": "si_0_rot_p10"}}
        )
        == "si_0_rot_p10"
    )


def test_selected_axes_keeps_experiments_separate() -> None:
    assert _selected_axes("si-max") == ["si-max"]
    assert _selected_axes("yaw-max") == ["yaw-max"]
    assert _selected_axes("both") == ["si-max", "yaw-max"]


def test_list_checkpoints_reads_only_run_and_checkpoints_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "epoch_001.pt").write_bytes(b"root")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_002.pt").write_bytes(b"child")
    nested = checkpoint_dir / "nested"
    nested.mkdir()
    (nested / "epoch_003.pt").write_bytes(b"ignored")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    rows = _list_checkpoints(tmp_path)

    assert [Path(row["path"]).name for row in rows] == [
        "epoch_002.pt",
        "epoch_001.pt",
    ]


def test_configured_view_count_supports_both_checkpoint_layouts() -> None:
    assert _configured_view_count({"view_count": 5}) == 5
    assert _configured_view_count({"dataset": {"view_count": 10}}) == 10
    assert _configured_view_count({}) is None


def test_configured_backend_distinguishes_checkpoint_families() -> None:
    assert _configured_backend({"model_version": "v0b", "combine": "mlp"}) == "difnet"
    assert (
        _configured_backend({"model": {"official_config": "finetune_s1.yaml"}})
        == "deepsparse"
    )
    assert _configured_backend({}) is None


def test_cache_roots_are_prioritized_for_required_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    mixed_root = tmp_path / "mixed"

    splits = {
        train_root: ["train"],
        test_root: ["test"],
        mixed_root: ["train", "test"],
    }
    monkeypatch.setattr(
        "run_setup_error_axis_max_experiment._candidate_splits",
        lambda root: splits[root],
    )

    assert _order_cache_roots_for_split(
        [train_root, test_root, mixed_root],
        "test",
    ) == [test_root, mixed_root, train_root]


def test_grid_control_is_attached_without_replacing_eval_grid_truth() -> None:
    comparison = [
        {
            "pair_key": "patient6_case2_vs_case0",
            "true_setup_error_SI_mm": 8.8,
            "true_setup_error_yaw_deg": 1.2,
        }
    ]
    controls = [
        {
            "pair_key": "patient6_case2_vs_case0",
            "evaluation_grid_size": 128,
            "source_grid_size": 256,
            "source_grid_setup_error_SI_mm": 8.85,
            "source_grid_setup_error_yaw_deg": 1.25,
            "evaluation_minus_source_SI_mm": -0.05,
            "abs_evaluation_minus_source_SI_mm": 0.05,
            "evaluation_minus_source_yaw_deg": -0.05,
            "abs_evaluation_minus_source_yaw_deg": 0.05,
            "overall_pass": True,
        }
    ]

    rows = _attach_grid_control(comparison, controls)

    assert rows[0]["true_setup_error_SI_mm"] == 8.8
    assert rows[0]["true_registration_grid_size"] == 128
    assert rows[0]["source_formal_grid_size"] == 256
    assert rows[0]["grid_control_pass"] is True
