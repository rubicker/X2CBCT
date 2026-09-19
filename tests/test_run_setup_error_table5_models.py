from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from run_setup_error_table5_models import (
    DEFAULT_MODEL_CONFIG,
    _apply_overrides,
    _archive_model_output,
    _auto_resume_if_needed,
    _axis_command,
    _load_model_config,
    _reusable_model_output,
    _selected_models,
    _verify_common_truth,
    _wide_rows,
)


def test_default_config_contains_all_five_table5_models() -> None:
    config = _load_model_config(DEFAULT_MODEL_CONFIG)

    assert [model["id"] for model in config["models"]] == [
        "difnet",
        "dufal",
        "c2rv",
        "deepsparse",
        "ours",
    ]
    assert [model["backend"] for model in config["models"]] == [
        "difnet",
        "difnet",
        "difnet",
        "deepsparse",
        "difnet",
    ]
    assert all(Path(model["checkpoint"]).is_file() for model in config["models"])


def test_model_selection_and_checkpoint_override_are_explicit() -> None:
    config = _load_model_config(DEFAULT_MODEL_CONFIG)
    selected = _selected_models(config, ["c2rv,ours"])
    override = PROJECT_ROOT / "custom.pt"

    models = _apply_overrides(
        selected,
        checkpoints={"ours": override},
        chunk_sizes={"c2rv": 12345},
    )

    assert [model["id"] for model in models] == ["c2rv", "ours"]
    assert models[0]["chunk_size"] == 12345
    assert models[1]["checkpoint"] == override


def test_axis_command_keeps_model_specific_backend_and_checkpoint(
    tmp_path: Path,
) -> None:
    config = _load_model_config(DEFAULT_MODEL_CONFIG)
    model = _selected_models(config, ["deepsparse"])[0]
    args = argparse.Namespace(
        python=Path(sys.executable),
        formal_registration_results=tmp_path / "formal.csv",
        roi_reviewed=tmp_path / "roi.csv",
        si_pair="6:0:2",
        yaw_pair="0:0:6",
        require_inference_split="test",
        view_count=5,
        eval_grid_size=128,
        grid_control_si_tolerance_mm=0.5,
        grid_control_yaw_tolerance_deg=0.5,
        device="cuda",
        cache_root=[tmp_path / "cache"],
        skip_viewer=True,
        enforce_grid_control=False,
        dry_run=True,
    )

    command = _axis_command(args, model, output_dir=tmp_path / "out")

    assert Path(command[0]) == PROJECT_ROOT / ".conda" / "deepsparse" / "Scripts" / "python.exe"
    assert command[command.index("--backend") + 1] == "deepsparse"
    assert command[command.index("--checkpoint") + 1] == str(model["checkpoint"])
    assert command[command.index("--eval-grid-size") + 1] == "128"
    assert "--dry-run" in command
    assert "--skip-viewer" in command


def test_resume_reuses_only_matching_checkpoint_and_archives_stale_output(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model_runs" / "c2rv"
    model_dir.mkdir(parents=True)
    (model_dir / "table_ready_results.csv").write_text("axis_experiment\n", encoding="utf-8")
    (model_dir / "run_summary.json").write_text(
        '{"ok": true, "checkpoint": "C:/checkpoints/epoch_0100.pt"}\n',
        encoding="utf-8",
    )

    assert _reusable_model_output(model_dir, Path("C:/checkpoints/epoch_0100.pt"))
    assert not _reusable_model_output(model_dir, Path("C:/checkpoints/epoch_0150.pt"))

    archive = _archive_model_output(model_dir)

    assert not model_dir.exists()
    assert archive.parent == tmp_path / "model_runs" / "_resume_archive"
    assert (archive / "run_summary.json").is_file()


def test_existing_batch_output_enables_safe_resume_automatically(tmp_path: Path) -> None:
    args = argparse.Namespace(output_dir=tmp_path / "batch", resume=False)
    args.output_dir.mkdir()
    (args.output_dir / "resolved_batch.json").write_text("{}\n", encoding="utf-8")

    assert _auto_resume_if_needed(args)
    assert args.resume is True


def _long_row(model_id: str, axis: str, true_si: str, true_yaw: str) -> dict[str, str]:
    return {
        "model_id": model_id,
        "axis_experiment": axis,
        "true_SI_mm": true_si,
        "reconstructed_SI_mm": "1.0",
        "abs_error_SI_mm": "0.1",
        "true_yaw_deg": true_yaw,
        "reconstructed_yaw_deg": "2.0",
        "abs_error_yaw_deg": "0.2",
        "grid_control_pass": "True",
    }


def test_cross_model_truth_must_be_identical_and_wide_table_has_one_row_per_model() -> None:
    config = _load_model_config(DEFAULT_MODEL_CONFIG)
    models = _selected_models(config, ["difnet", "dufal"])
    rows = [
        _long_row("difnet", "si-max", "8.8", "1.2"),
        _long_row("difnet", "yaw-max", "-0.6", "9.3"),
        _long_row("dufal", "si-max", "8.8", "1.2"),
        _long_row("dufal", "yaw-max", "-0.6", "9.3"),
    ]

    _verify_common_truth(rows)
    wide = _wide_rows(rows, models)

    assert [row["model_id"] for row in wide] == ["difnet", "dufal"]
    assert wide[0]["si_pair_true_C128_SI_mm"] == "8.8"
    assert wide[1]["yaw_pair_true_C128_yaw_deg"] == "9.3"
    assert len({header.casefold() for header in wide[0]}) == len(wide[0])

    rows[-1]["true_yaw_deg"] = "9.4"
    with pytest.raises(ValueError, match="inconsistent C128"):
        _verify_common_truth(rows)


def test_cross_model_truth_allows_sub_four_decimal_registration_jitter() -> None:
    rows = [
        _long_row("difnet", "si-max", "8.6378925279", "1.2894834577"),
        _long_row("difnet", "yaw-max", "-0.6741743567", "9.1970555952"),
        _long_row("dufal", "si-max", "8.6378925230", "1.2894834440"),
        _long_row("dufal", "yaw-max", "-0.6741743506", "9.1970569076"),
    ]

    _verify_common_truth(rows)
