from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from x2cbct_difnet.setup_error_manifest import build_pair_manifest
from x2cbct_difnet.setup_error_registration import initialization_specs, moving_to_reference_components, primary_setup_error_components
from x2cbct_difnet.setup_error_roi import generate_roi_candidate, read_roi_csv, write_roi_candidates


def write_cbct(path: Path, *, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((32, 34, 36), dtype=np.float32)
    array[8:26, 16:28, 14 + offset : 20 + offset] = 600.0
    array[10:23, 20:26, 16 + offset : 18 + offset] = 1000.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((-18.0, -17.0, -16.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))


def test_pair_manifest_uses_first_case_as_reference(tmp_path: Path) -> None:
    write_cbct(tmp_path / "0" / "0" / "CBCT.nii.gz")
    write_cbct(tmp_path / "0" / "1" / "CBCT.nii.gz", offset=1)
    write_cbct(tmp_path / "1" / "0" / "CBCT.nii.gz")

    manifest = build_pair_manifest(tmp_path)

    assert manifest["patient_count"] == 2
    assert manifest["case_count"] == 3
    assert manifest["pair_count"] == 1
    assert manifest["pairs"][0]["patient_id"] == "0"
    assert manifest["pairs"][0]["reference_case_id"] == "0"
    assert manifest["pairs"][0]["moving_case_id"] == "1"
    assert manifest["header_consistency"]["all_consistent"] is True


def test_roi_candidate_round_trip_csv(tmp_path: Path) -> None:
    cbct = tmp_path / "0" / "0" / "CBCT.nii.gz"
    write_cbct(cbct)

    candidate = generate_roi_candidate("0", "0", cbct, threshold_low=300, threshold_high=1700)
    csv_path = tmp_path / "roi_candidates.csv"
    write_roi_candidates([candidate], csv_path)
    rows = read_roi_csv(csv_path)
    row = rows[("0", "0")]

    assert row["status"] in {"accepted", "needs_fix"}
    assert int(row["x_min"]) <= int(row["x_max"])
    assert int(row["y_min"]) <= int(row["y_max"])
    assert int(row["z_min"]) <= int(row["z_max"])
    assert float(row["threshold_low"]) == 300.0


def test_registration_initialization_plan_matches_protocol() -> None:
    specs = initialization_specs()

    assert len(specs) == 13
    assert specs[0][0] == "base"
    names = {item[0] for item in specs}
    assert "LR_-3mm" in names
    assert "LR_+3mm" in names
    assert "pitch_-2deg" in names
    assert "yaw_+2deg" in names


def test_moving_to_reference_components_handles_composite_inverse() -> None:
    transform = sitk.Euler3DTransform()
    transform.SetRotation(0.0, 0.0, 0.0)
    transform.SetTranslation((2.0, -3.0, 4.0))
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(transform)

    tx, ty, tz, pitch, roll, yaw = moving_to_reference_components(composite)

    assert np.isclose(tx, -2.0)
    assert np.isclose(ty, 3.0)
    assert np.isclose(tz, -4.0)
    assert np.isclose(pitch, 0.0)
    assert np.isclose(roll, 0.0)
    assert np.isclose(yaw, 0.0)


def test_primary_setup_error_uses_si_and_yaw_only() -> None:
    metrics = primary_setup_error_components(-3.5, 6.25)

    assert metrics["setup_error_SI_mm"] == -3.5
    assert metrics["setup_error_abs_SI_mm"] == 3.5
    assert metrics["setup_error_yaw_deg"] == 6.25
    assert metrics["setup_error_abs_yaw_deg"] == 6.25
