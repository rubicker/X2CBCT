from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_setup_error_focus_experiment as focus_experiment
from run_setup_error_focus_experiment import (
    FocusPair,
    _comparison_rows,
    _denormalize_cbct,
    _load_pairs,
    _write_nifti,
)


def test_default_focus_pairs_are_fixed() -> None:
    pairs = _load_pairs(None)

    assert len(pairs) == 10
    assert pairs[0] == FocusPair("0", "0", "5")
    assert pairs[-1] == FocusPair("27", "0", "12")
    assert pairs[4].pair_key == "patient24_case7_vs_case2"


def test_find_dataset_for_case_forwards_c2rv_grid_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, cache_root: Path, **kwargs: object) -> None:
            captured["cache_root"] = cache_root
            captured.update(kwargs)

    monkeypatch.setattr(focus_experiment, "_candidate_splits", lambda _root: ["test"])
    monkeypatch.setattr(focus_experiment, "_find_sample_index", lambda *_args: 3)
    monkeypatch.setattr(focus_experiment, "CachedDifNetV0bDataset", FakeDataset)

    dataset, index = focus_experiment._find_dataset_for_case(
        [Path("cache")],
        "view5",
        "6",
        "2",
        128,
        c2rv_grids=True,
        c2rv_grid_scales=(16, 8, 4),
    )

    assert isinstance(dataset, FakeDataset)
    assert index == 3
    assert captured["c2rv_grids"] is True
    assert captured["c2rv_grid_scales"] == (16, 8, 4)


def test_denormalize_cbct_uses_fixed_hu_profile() -> None:
    array = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    norm = {"mode": "fixed", "hu_min": -290.0, "hu_span": 2350.0}

    out = _denormalize_cbct(array, norm)

    assert np.allclose(out, [-290.0, 885.0, 2060.0])


def test_comparison_rows_compute_signed_errors() -> None:
    pair = FocusPair("24", "2", "8")
    predictions = [
        {
            "pair_key": pair.pair_key,
            "inference_cache_version": "train_mixed_noaug_256",
            "inference_split": "train",
            "generated_cbct_path": "generated.npy",
            "target_cbct_path": "target.npy",
        }
    ]
    pred_results = [
        {
            "patient_id": "24",
            "reference_case_id": "2",
            "moving_case_id": "8_bhat",
            "status": "ok",
            "setup_error_SI_mm": "1.5",
            "setup_error_yaw_deg": "6.5",
            "similarity_score": "0.9",
        }
    ]
    formal = {
        ("24", "2", "8"): {
            "status": "ok",
            "setup_error_SI_mm": "-0.5",
            "setup_error_yaw_deg": "7.0",
        }
    }

    rows = _comparison_rows([pair], predictions, pred_results, formal)

    assert rows[0]["delta_SI_mm"] == 2.0
    assert rows[0]["delta_yaw_deg"] == -0.5
    assert rows[0]["SI_direction_match"] is False
    assert rows[0]["yaw_direction_match"] is True


def test_write_nifti_preserves_processed_geometry(tmp_path: Path) -> None:
    array = np.zeros((3, 4, 5), dtype=np.float32)
    manifest = {
        "processed_spacing_xyz": [1.1, 1.2, 1.3],
        "processed_origin_xyz": [-2.0, -3.0, -4.0],
        "processed_direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }

    path = _write_nifti(array, manifest, tmp_path / "cbct.nii.gz")
    image = sitk.ReadImage(str(path))

    assert image.GetSize() == (5, 4, 3)
    assert np.allclose(image.GetSpacing(), (1.1, 1.2, 1.3))
    assert np.allclose(image.GetOrigin(), (-2.0, -3.0, -4.0))
