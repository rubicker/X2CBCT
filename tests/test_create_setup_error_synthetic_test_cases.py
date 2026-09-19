from __future__ import annotations

import random
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from create_setup_error_synthetic_test_cases import (
    Pair,
    _find_source_sample,
    _read_nifti_unicode,
    draw_four_decimal_noninteger,
    require_four_decimal_noninteger,
    signed_target_from_baseline,
)


def test_random_magnitudes_are_deterministic_noninteger_four_dp() -> None:
    first = draw_four_decimal_noninteger(
        random.Random(20260724),
        low="7.2501",
        high="9.7500",
        name="S-I",
        minimum=Decimal("7.0"),
    )
    second = draw_four_decimal_noninteger(
        random.Random(20260724),
        low="7.2501",
        high="9.7500",
        name="S-I",
        minimum=Decimal("7.0"),
    )

    assert first == second
    assert first >= Decimal("7.0")
    assert first != first.to_integral_value()
    assert f"{first:.4f}"[-1] != "0"


def test_explicit_magnitude_rejects_integer_and_more_than_four_dp() -> None:
    assert require_four_decimal_noninteger(
        "8.1234", name="yaw", minimum=Decimal("8.0")
    ) == Decimal("8.1234")

    with pytest.raises(ValueError, match="non-integer"):
        require_four_decimal_noninteger(
            "9.0000", name="yaw", minimum=Decimal("8.0")
        )
    with pytest.raises(ValueError, match="at most four"):
        require_four_decimal_noninteger(
            "8.12345", name="yaw", minimum=Decimal("8.0")
        )


def test_target_direction_follows_baseline_sign() -> None:
    rng = random.Random(1)

    assert signed_target_from_baseline(
        Decimal("8.1234"), 5.0, mode="away-from-zero", rng=rng
    ) == Decimal("8.1234")
    assert signed_target_from_baseline(
        Decimal("8.1234"), -2.0, mode="away-from-zero", rng=rng
    ) == Decimal("-8.1234")


def test_source_case_must_be_view5_test() -> None:
    pair = Pair("6", "0", "2")
    test_manifest = {
        "samples": [
            {
                "sample_id": "patient6_case2_view_5",
                "patient_id": "6",
                "case_id": "2",
                "view_key": "view_5",
                "split": "test",
            }
        ]
    }

    assert _find_source_sample(test_manifest, pair)["split"] == "test"
    test_manifest["samples"][0]["split"] = "train"
    with pytest.raises(ValueError, match="required split='test'"):
        _find_source_sample(test_manifest, pair)


def test_unicode_safe_nifti_reader_preserves_shape_spacing_and_values(
    tmp_path: Path,
) -> None:
    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.9, 1.1))
    ascii_path = tmp_path / "source.nii.gz"
    sitk.WriteImage(image, str(ascii_path))
    unicode_dir = tmp_path / "新建文件夹"
    unicode_dir.mkdir()
    unicode_path = unicode_dir / "source.nii.gz"
    ascii_path.rename(unicode_path)

    metadata, loaded = _read_nifti_unicode(unicode_path)

    assert loaded.shape == array.shape
    assert np.asarray(loaded) == pytest.approx(array)
    assert metadata.GetSpacing() == pytest.approx((0.8, 0.9, 1.1))
