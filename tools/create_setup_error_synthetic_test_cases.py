from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import (
    _normalized_projection_clip_high_ratio,
    _normalized_projection_mean,
    _projection_spacing_yx,
    _resize_projection,
    _resize_volume_to_cube,
    read_json,
    relative_to_cache,
    resolve_cache_path,
    write_json,
)
from x2cbct_difnet.geometry import read_rtk_scan_geometry
from x2cbct_difnet.normalization import (
    denormalize_cbct_hu,
    normalize_cbct_hu,
    normalize_projection,
)
from x2cbct_difnet.setup_augmentation import (
    image_background_value,
    shift_image_vertical,
    transform_volume_setup,
    volume_background_value,
)
from x2cbct_difnet.setup_error_registration import (
    load_roi_review,
    register_pair,
    write_registration_outputs,
)


DEFAULT_SOURCE_CACHE = (
    PROJECT_ROOT
    / "data_cache"
    / "difnet_v1_si10_rot10_3x3_256"
    / "self_projection1208"
    / "test_mixed_noaug_256"
)
DEFAULT_RAW_DATASET_ROOT = Path(r"E:\新建文件夹\self_projection1208")
DEFAULT_REFERENCE_CACHE = (
    PROJECT_ROOT
    / "data_cache"
    / "difnet_v1_si10_rot10_3x3_256"
    / "self_projection1208"
    / "train_mixed_noaug_256"
)
DEFAULT_SHARED_BASELINE = (
    PROJECT_ROOT / "runs" / "setup_error" / "model_experiments" / "_shared_baseline"
)
DEFAULT_FORMAL_RESULTS = (
    DEFAULT_SHARED_BASELINE / "registration_si_yaw_fixed" / "registration_results.csv"
)
DEFAULT_ROI_REVIEWED = DEFAULT_SHARED_BASELINE / "cervical_roi_review" / "roi_reviewed.csv"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "setup_error"
    / "synthetic_test_stress_cases"
    / "seed_20260724"
)
FOUR_DP = Decimal("0.0001")
VIEW_KEY = "view_5"


@dataclass(frozen=True)
class NiftiMetadata:
    spacing_xyz: tuple[float, ...]

    def GetSpacing(self) -> tuple[float, ...]:
        return self.spacing_xyz


def _read_nifti_unicode(path: Path) -> tuple[NiftiMetadata, np.ndarray]:
    """Read the NIfTI subset used by this dataset when SimpleITK rejects Unicode paths."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        header = handle.read(348)
        if len(header) != 348:
            raise ValueError(f"truncated NIfTI header: {path}")
        little_size = struct.unpack("<i", header[0:4])[0]
        big_size = struct.unpack(">i", header[0:4])[0]
        if little_size == 348:
            endian = "<"
        elif big_size == 348:
            endian = ">"
        else:
            raise ValueError(f"not a NIfTI-1 header: {path}")
        dims = struct.unpack(f"{endian}8h", header[40:56])
        ndim = int(dims[0])
        if ndim < 2 or ndim > 7:
            raise ValueError(f"unsupported NIfTI dimension count {ndim}: {path}")
        shape_xyz = tuple(int(value) for value in dims[1 : ndim + 1])
        datatype = int(struct.unpack(f"{endian}h", header[70:72])[0])
        bitpix = int(struct.unpack(f"{endian}h", header[72:74])[0])
        pixdim = struct.unpack(f"{endian}8f", header[76:108])
        vox_offset = float(struct.unpack(f"{endian}f", header[108:112])[0])
        slope = float(struct.unpack(f"{endian}f", header[112:116])[0])
        intercept = float(struct.unpack(f"{endian}f", header[116:120])[0])
        dtype_codes = {
            2: "u1",
            4: "i2",
            8: "i4",
            16: "f4",
            64: "f8",
            256: "i1",
            512: "u2",
            768: "u4",
            1024: "i8",
            1280: "u8",
        }
        dtype_code = dtype_codes.get(datatype)
        if dtype_code is None:
            raise ValueError(f"unsupported NIfTI datatype {datatype}: {path}")
        dtype = np.dtype(dtype_code).newbyteorder(endian)
        if dtype.itemsize * 8 != bitpix:
            raise ValueError(
                f"NIfTI datatype/bitpix mismatch {datatype}/{bitpix}: {path}"
            )
        current = 348
        target = max(int(round(vox_offset)), 348)
        if target > current:
            handle.read(target - current)
        count = int(np.prod(shape_xyz, dtype=np.int64))
        raw = handle.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise ValueError(f"truncated NIfTI payload: {path}")
    array = np.frombuffer(raw, dtype=dtype, count=count).reshape(
        tuple(reversed(shape_xyz)), order="C"
    )
    if math.isfinite(slope) and slope != 0.0:
        array = array.astype(np.float32) * slope + (
            intercept if math.isfinite(intercept) else 0.0
        )
    spacing = tuple(abs(float(value)) for value in pixdim[1 : ndim + 1])
    return NiftiMetadata(spacing_xyz=spacing), np.asarray(array)


def _read_image_robust(path: Path) -> tuple[Any, np.ndarray]:
    try:
        image = sitk.ReadImage(str(path))
        return image, np.asarray(sitk.GetArrayFromImage(image))
    except RuntimeError as exc:
        if all(ord(char) < 128 for char in str(path)):
            raise
        try:
            return _read_nifti_unicode(path)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"SimpleITK and Unicode-safe NIfTI readers both failed for {path}"
            ) from fallback_exc


@dataclass(frozen=True)
class Pair:
    patient_id: str
    reference_case_id: str
    moving_case_id: str

    @property
    def source_case_key(self) -> str:
        return f"patient{self.patient_id}_case{self.moving_case_id}"

    @property
    def pair_key(self) -> str:
        return (
            f"patient{self.patient_id}_case{self.moving_case_id}"
            f"_vs_case{self.reference_case_id}"
        )


@dataclass(frozen=True)
class StressSpec:
    experiment: str
    pair: Pair
    target_si_mm: Decimal
    target_yaw_deg: Decimal
    translation_mm: Decimal
    yaw_deg: Decimal
    baseline_si_mm: float
    baseline_yaw_deg: float

    @property
    def augmentation_id(self) -> str:
        return (
            f"synthetic_si_{decimal_token(self.translation_mm)}"
            f"_yaw_{decimal_token(self.yaw_deg)}"
        )

    @property
    def case_key(self) -> str:
        target = (
            f"si_target_{decimal_token(self.target_si_mm)}"
            if self.experiment == "si-max"
            else f"yaw_target_{decimal_token(self.target_yaw_deg)}"
        )
        return f"{self.pair.source_case_key}_synthetic_{target}"


def parse_pair(text: str) -> Pair:
    fields = [part.strip() for part in str(text).split(":")]
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError(
            f"pair must be PATIENT:REFERENCE:MOVING, got {text!r}"
        )
    return Pair(*fields)


def decimal_token(value: Decimal) -> str:
    text = f"{abs(value):.4f}".replace(".", "p")
    if value < 0:
        return f"m{text}"
    if value > 0:
        return f"p{text}"
    return "0p0000"


def require_four_decimal_noninteger(value: str, *, name: str, minimum: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} is not a valid decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    quantized = parsed.quantize(FOUR_DP)
    if parsed != quantized:
        raise ValueError(f"{name} accepts at most four decimal places, got {value!r}")
    if abs(quantized) < minimum:
        raise ValueError(f"|{name}| must be >= {minimum}, got {quantized}")
    if quantized == quantized.to_integral_value():
        raise ValueError(f"{name} must be non-integer, got {quantized:.4f}")
    return abs(quantized)


def draw_four_decimal_noninteger(
    rng: random.Random,
    *,
    low: str,
    high: str,
    name: str,
    minimum: Decimal,
) -> Decimal:
    low_d = Decimal(str(low))
    high_d = Decimal(str(high))
    if not low_d.is_finite() or not high_d.is_finite() or high_d < low_d:
        raise ValueError(f"invalid {name} range: [{low}, {high}]")
    if low_d < minimum:
        raise ValueError(f"{name} lower bound must be >= {minimum}, got {low_d}")
    low_units = int((low_d / FOUR_DP).to_integral_value(rounding=ROUND_CEILING))
    high_units = int((high_d / FOUR_DP).to_integral_value(rounding=ROUND_FLOOR))
    candidates = [
        unit
        for unit in range(low_units, high_units + 1)
        if unit % 10000 != 0 and unit % 10 != 0
    ]
    if not candidates:
        raise ValueError(
            f"{name} range contains no non-integer value with a non-zero fourth decimal"
        )
    return (Decimal(rng.choice(candidates)) * FOUR_DP).quantize(FOUR_DP)


def signed_target_from_baseline(
    magnitude: Decimal,
    baseline: float,
    *,
    mode: str,
    rng: random.Random,
) -> Decimal:
    if mode == "positive":
        return abs(magnitude)
    if mode == "negative":
        return -abs(magnitude)
    if mode != "away-from-zero":
        raise ValueError(f"unsupported sign mode: {mode}")
    if baseline > 0:
        return abs(magnitude)
    if baseline < 0:
        return -abs(magnitude)
    return abs(magnitude) if rng.randrange(2) else -abs(magnitude)


def _read_formal_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (
                str(row["patient_id"]),
                str(row["reference_case_id"]),
                str(row["moving_case_id"]),
            ): row
            for row in csv.DictReader(handle)
        }


def _formal_row(
    rows: dict[tuple[str, str, str], dict[str, str]],
    pair: Pair,
) -> dict[str, str]:
    key = (pair.patient_id, pair.reference_case_id, pair.moving_case_id)
    row = rows.get(key)
    if row is None:
        raise ValueError(f"formal registration row is missing for {key}")
    if str(row.get("status", "")).lower() != "ok":
        raise ValueError(f"formal registration is not ok for {key}: {row.get('status')}")
    return row


def _find_source_sample(
    cache_manifest: dict[str, Any],
    pair: Pair,
    *,
    required_split: str = "test",
) -> dict[str, Any]:
    rows = [
        row
        for row in cache_manifest.get("samples", [])
        if str(row.get("patient_id")) == pair.patient_id
        and str(row.get("case_id")) == pair.moving_case_id
        and str(row.get("view_key")) == VIEW_KEY
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one {VIEW_KEY} sample for {pair.source_case_key}, "
            f"found {len(rows)}"
        )
    row = rows[0]
    if str(row.get("split")) != required_split:
        raise ValueError(
            f"{pair.source_case_key} is split={row.get('split')!r}, "
            f"required split={required_split!r}"
        )
    setup = row.get("setup_augmentation")
    if isinstance(setup, dict) and (
        float(setup.get("translation_mm", 0) or 0) != 0.0
        or float(setup.get("rotation_deg", 0) or 0) != 0.0
    ):
        raise ValueError(
            f"source sample must be unaugmented, got {setup.get('augmentation_id')}"
        )
    return row


def _all_gantry_angles(geometry_path: Path) -> list[float]:
    root = ET.parse(geometry_path).getroot()
    return [
        float(node.findtext("GantryAngle", default="nan"))
        for node in root.findall("Projection")
    ]


def _projection_ordinal(path: Path) -> int | None:
    try:
        return int(path.name.split("_", 1)[0])
    except ValueError:
        return None


def _angular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _select_projection(
    raw_case_path: Path,
    *,
    target_angle_deg: float,
) -> dict[str, Any]:
    angles = _all_gantry_angles(raw_case_path / "Geometry")
    candidates: list[dict[str, Any]] = []
    for path in raw_case_path.glob("*.nii.gz"):
        if path.name == "CBCT.nii.gz":
            continue
        ordinal = _projection_ordinal(path)
        if ordinal is None or ordinal >= len(angles):
            continue
        gantry = float(angles[ordinal])
        if not math.isfinite(gantry):
            continue
        candidates.append(
            {
                "path": path,
                "file": path.name,
                "ordinal": int(ordinal),
                "target_angle_deg": float(target_angle_deg),
                "gantry_angle_deg": gantry,
                "angular_residual_deg": _angular_distance_deg(
                    gantry, target_angle_deg
                ),
            }
        )
    if not candidates:
        raise ValueError(f"no projection candidates under {raw_case_path}")
    return min(
        candidates,
        key=lambda item: (
            float(item["angular_residual_deg"]),
            int(item["ordinal"]),
            str(item["file"]),
        ),
    )


def _projection_shift(
    translation_mm: float,
    *,
    sid_mm: float,
    sdd_mm: float,
    detector_spacing_y_mm: float,
    detector_height: int,
    projection_size: int,
) -> dict[str, float]:
    detector_shift_mm = -translation_mm * sdd_mm / sid_mm
    raw_pixel_shift_y = detector_shift_mm / detector_spacing_y_mm
    resized_pixel_shift_y = raw_pixel_shift_y * (
        (float(projection_size) - 1.0) / max(float(detector_height) - 1.0, 1.0)
    )
    return {
        "si_shift_mm": float(translation_mm),
        "sid_mm": float(sid_mm),
        "sdd_mm": float(sdd_mm),
        "magnification": float(sdd_mm / sid_mm),
        "detector_shift_mm": float(detector_shift_mm),
        "raw_pixel_shift_y": float(raw_pixel_shift_y),
        "resized_pixel_shift_y": float(resized_pixel_shift_y),
    }


def _make_projection(
    *,
    selected: dict[str, Any],
    raw_case_path: Path,
    output_root: Path,
    case_key: str,
    augmentation: dict[str, Any],
    projection_size: int,
    delta_raw: float,
) -> dict[str, Any]:
    image, raw = _read_image_robust(Path(selected["path"]))
    squeezed = np.squeeze(raw)
    detector_shape = tuple(int(value) for value in squeezed.shape[-2:])
    detector_spacing = _projection_spacing_yx(image)
    resized = _resize_projection(squeezed, projection_size)
    corrected = resized + float(delta_raw)
    shift_metadata = None
    translation_mm = float(augmentation["translation_mm"])
    if translation_mm != 0.0:
        geometry = read_rtk_scan_geometry(raw_case_path / "Geometry")
        shift_metadata = _projection_shift(
            translation_mm,
            sid_mm=float(geometry.source_to_isocenter_distance),
            sdd_mm=float(geometry.source_to_detector_distance),
            detector_spacing_y_mm=float(detector_spacing[0]),
            detector_height=int(detector_shape[0]),
            projection_size=int(projection_size),
        )
        fill = image_background_value(corrected)
        corrected = shift_image_vertical(
            corrected,
            shift_metadata["resized_pixel_shift_y"],
            fill_value=fill,
        )
        shift_metadata["projection_fill_value"] = float(fill)

    stem = Path(str(selected["file"])).stem
    payload_path = output_root / "projections" / case_key / f"{stem}.npy"
    metadata_path = output_root / "projections" / case_key / f"{stem}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(payload_path, normalize_projection(corrected).astype(np.float16))
    metadata = {
        "raw_projection_path": str(Path(selected["path"]).resolve()),
        "payload_path": relative_to_cache(payload_path, output_root),
        "metadata_path": relative_to_cache(metadata_path, output_root),
        "target_angle_deg": float(selected["target_angle_deg"]),
        "ordinal": int(selected["ordinal"]),
        "file": str(selected["file"]),
        "gantry_angle_deg": float(selected["gantry_angle_deg"]),
        "angular_residual_deg": float(selected["angular_residual_deg"]),
        "detector_shape": list(detector_shape),
        "detector_spacing_yx": list(detector_spacing),
        "projection_size": int(projection_size),
        "normalization": {"rule": "clip_0_8.6_div_8.6", "min": 0.0, "max": 8.6},
        "projection_correction": {
            "mode": "source_case_additive_offset",
            "method": "additive_offset",
            "applied": bool(delta_raw != 0.0),
            "delta_raw": float(delta_raw),
            "pre_norm_mean": _normalized_projection_mean(resized),
            "post_norm_mean": _normalized_projection_mean(corrected),
            "post_clip_high_ratio": _normalized_projection_clip_high_ratio(corrected),
        },
        "setup_augmentation": {
            **augmentation,
            "selected_source_file": str(selected["file"]),
            "selected_source_ordinal": int(selected["ordinal"]),
            "selected_target_angle_deg": float(selected["target_angle_deg"]),
            "selected_gantry_angle_deg": float(selected["gantry_angle_deg"]),
            "selected_angular_residual_deg": float(
                selected["angular_residual_deg"]
            ),
        },
    }
    if shift_metadata is not None:
        metadata["setup_augmentation"][
            "translation_projection_shift"
        ] = shift_metadata
    write_json(metadata, metadata_path)
    return metadata


def _case_offset(cache_manifest: dict[str, Any], source_case_key: str) -> float:
    table = (
        cache_manifest.get("projection_correction", {}).get("case_offset_table", {})
    )
    row = table.get(source_case_key) if isinstance(table, dict) else None
    return float(row.get("delta_raw", 0.0)) if isinstance(row, dict) else 0.0


def _build_stress_sample(
    *,
    spec: StressSpec,
    source_cache_root: Path,
    source_manifest: dict[str, Any],
    source_sample: dict[str, Any],
    raw_dataset_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_case_manifest = read_json(
        resolve_cache_path(source_cache_root, str(source_sample["case_manifest_path"]))
    )
    raw_case_path = (
        raw_dataset_root / spec.pair.patient_id / spec.pair.moving_case_id
    ).resolve()
    if not raw_case_path.exists():
        raise FileNotFoundError(f"raw source case is missing: {raw_case_path}")
    for required_name in ("CBCT.nii.gz", "Geometry"):
        required_path = raw_case_path / required_name
        if not required_path.exists():
            raise FileNotFoundError(f"raw source asset is missing: {required_path}")

    translation_mm = float(spec.translation_mm)
    yaw_deg = float(spec.yaw_deg)
    setup = {
        "augmentation_id": spec.augmentation_id,
        "translation_mm": translation_mm,
        "rotation_deg": yaw_deg,
        "rotation_axis": "yaw",
        "source_patient_id": int(spec.pair.patient_id),
        "source_case_id": int(spec.pair.moving_case_id),
        "source_case_key": spec.pair.source_case_key,
        "source_split": "test",
        "synthetic": True,
        "value_precision": 4,
        "volume_transform_order": "rotate_yaw_then_translate_si",
        "target_setup_error_SI_mm": float(spec.target_si_mm),
        "target_setup_error_yaw_deg": float(spec.target_yaw_deg),
        "parameter_semantics": "increment applied to the original moving case",
    }

    cbct_image, cbct_raw = _read_image_robust(raw_case_path / "CBCT.nii.gz")
    raw_fill = volume_background_value(cbct_raw)
    transformed_raw = transform_volume_setup(
        cbct_raw,
        translation_mm=translation_mm,
        rotation_deg=yaw_deg,
        spacing_z_mm=float(cbct_image.GetSpacing()[2]),
        fill_value=raw_fill,
    )
    target_size = int(source_case_manifest["target_size"])
    transformed_resized = _resize_volume_to_cube(transformed_raw, target_size)
    transformed_norm = normalize_cbct_hu(
        transformed_resized,
        normalization=source_manifest["cbct_normalization"],
    )
    normalized_fill = float(
        normalize_cbct_hu(
            np.asarray([raw_fill], dtype=np.float32),
            normalization=source_manifest["cbct_normalization"],
        )[0]
    )
    setup["volume_fill_value"] = float(raw_fill)
    setup["target_sampling_fill_value"] = normalized_fill

    case_dir = output_root / "cases" / spec.case_key
    cbct_path = case_dir / "cbct_norm.npy"
    case_manifest_path = case_dir / "case_manifest.json"
    geometry_summary_path = case_dir / "geometry_summary.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(cbct_path, transformed_norm.astype(np.float16))

    case_manifest = dict(source_case_manifest)
    case_manifest.update(
        {
            "case_key": spec.case_key,
            "patient_id": spec.pair.patient_id,
            "case_id": spec.pair.moving_case_id,
            "raw_case_path": str(raw_case_path),
            "case_manifest_path": relative_to_cache(case_manifest_path, output_root),
            "cbct_payload_path": relative_to_cache(cbct_path, output_root),
            "geometry_summary_path": relative_to_cache(
                geometry_summary_path, output_root
            ),
            "setup_augmentation": setup,
            "synthetic_stress_case": {
                "experiment": spec.experiment,
                "pair_key": spec.pair.pair_key,
                "source_cache_root": str(source_cache_root),
                "source_sample_path": str(
                    resolve_cache_path(
                        source_cache_root, str(source_sample["sample_path"])
                    )
                ),
                "baseline_setup_error_SI_mm": spec.baseline_si_mm,
                "baseline_setup_error_yaw_deg": spec.baseline_yaw_deg,
                "target_setup_error_SI_mm": float(spec.target_si_mm),
                "target_setup_error_yaw_deg": float(spec.target_yaw_deg),
                "not_an_observed_treatment_fraction": True,
            },
        }
    )
    write_json(case_manifest, case_manifest_path)
    write_json(
        {
            "geometry_path": str(raw_case_path / "Geometry"),
            **case_manifest["projection_geometry"],
        },
        geometry_summary_path,
    )

    projection_size = int(
        source_sample.get("projection_geometry", {}).get(
            "projection_size",
            source_manifest.get("projection_resize", {}).get("size", 256),
        )
    )
    target_angles = [
        float(item["target_angle_deg"])
        for item in source_sample["selected_projections"]
    ]
    selected = [
        _select_projection(
            raw_case_path,
            target_angle_deg=float(target_angle) + yaw_deg,
        )
        for target_angle in target_angles
    ]
    delta_raw = _case_offset(source_manifest, spec.pair.source_case_key)
    projections = [
        _make_projection(
            selected=item,
            raw_case_path=raw_case_path,
            output_root=output_root,
            case_key=spec.case_key,
            augmentation=setup,
            projection_size=projection_size,
            delta_raw=delta_raw,
        )
        for item in selected
    ]

    sample_id = f"{spec.case_key}_{VIEW_KEY}"
    sample_path = output_root / VIEW_KEY / "test" / f"{sample_id}.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        sample_id=np.array(sample_id),
        split=np.array("test"),
        view_key=np.array(VIEW_KEY),
        case_key=np.array(spec.case_key),
        patient_id=np.array(spec.pair.patient_id),
        case_id=np.array(spec.pair.moving_case_id),
        case_manifest_path=np.array(case_manifest["case_manifest_path"]),
        cbct_payload_path=np.array(case_manifest["cbct_payload_path"]),
        projection_payload_paths=np.array(
            [item["payload_path"] for item in projections]
        ),
        projection_files=np.array([item["file"] for item in projections]),
        target_angles_deg=np.array(
            [item["target_angle_deg"] for item in projections], dtype=np.float32
        ),
        gantry_angles_deg=np.array(
            [item["gantry_angle_deg"] for item in projections], dtype=np.float32
        ),
        angular_residuals_deg=np.array(
            [item["angular_residual_deg"] for item in projections], dtype=np.float32
        ),
        detector_shape=np.array(projections[0]["detector_shape"], dtype=np.int32),
        detector_spacing_yx=np.array(
            projections[0]["detector_spacing_yx"], dtype=np.float32
        ),
        projection_size=np.array(projection_size, dtype=np.int32),
        target_volume_mode=np.array("full"),
        augmentation_id=np.array(spec.augmentation_id),
        translation_mm=np.array(translation_mm, dtype=np.float32),
        rotation_deg=np.array(yaw_deg, dtype=np.float32),
        source_patient_id=np.array(spec.pair.patient_id),
        source_case_id=np.array(spec.pair.moving_case_id),
        source_case_key=np.array(spec.pair.source_case_key),
        volume_fill_value=np.array(raw_fill, dtype=np.float32),
        target_sampling_fill_value=np.array(normalized_fill, dtype=np.float32),
    )

    sample = {
        "sample_id": sample_id,
        "split": "test",
        "view_key": VIEW_KEY,
        "geometry_branch": "real_projection",
        "patient_id": spec.pair.patient_id,
        "case_id": spec.pair.moving_case_id,
        "case_key": spec.case_key,
        "sample_path": relative_to_cache(sample_path, output_root),
        "case_manifest_path": case_manifest["case_manifest_path"],
        "cbct_payload_path": case_manifest["cbct_payload_path"],
        "target_volume_mode": "full",
        "projection_payload_paths": [
            item["payload_path"] for item in projections
        ],
        "selected_projections": projections,
        "projection_geometry": {
            "type": "real_projection",
            "detector_shape": projections[0]["detector_shape"],
            "detector_spacing_yx": projections[0]["detector_spacing_yx"],
            "gantry_angles_deg": [
                item["gantry_angle_deg"] for item in projections
            ],
            "selected_projection_files": [item["file"] for item in projections],
        },
        "setup_augmentation": setup,
        "synthetic_stress_case": case_manifest["synthetic_stress_case"],
    }
    summary = {
        "experiment": spec.experiment,
        "pair_key": spec.pair.pair_key,
        "patient_id": spec.pair.patient_id,
        "reference_case_id": spec.pair.reference_case_id,
        "moving_case_id": spec.pair.moving_case_id,
        "source_split": "test",
        "source_case_key": spec.pair.source_case_key,
        "synthetic_case_key": spec.case_key,
        "target_setup_error_SI_mm": float(spec.target_si_mm),
        "target_setup_error_SI_mm_4dp": f"{spec.target_si_mm:.4f}",
        "target_setup_error_yaw_deg": float(spec.target_yaw_deg),
        "target_setup_error_yaw_deg_4dp": f"{spec.target_yaw_deg:.4f}",
        "parameter_semantics": (
            "target is the intended total setup error; applied values are only "
            "increments relative to the original moving case"
        ),
        "applied_translation_mm": translation_mm,
        "applied_translation_mm_4dp": f"{translation_mm:.4f}",
        "applied_yaw_deg": yaw_deg,
        "applied_yaw_deg_4dp": f"{yaw_deg:.4f}",
        "baseline_setup_error_SI_mm": spec.baseline_si_mm,
        "baseline_setup_error_yaw_deg": spec.baseline_yaw_deg,
        "sample_path": str(sample_path.resolve()),
        "cbct_payload_path": str(cbct_path.resolve()),
        "selected_projection_files": [item["file"] for item in projections],
        "target_angles_deg": [
            float(item["target_angle_deg"]) for item in projections
        ],
        "selected_gantry_angles_deg": [
            float(item["gantry_angle_deg"]) for item in projections
        ],
        "angular_residuals_deg": [
            float(item["angular_residual_deg"]) for item in projections
        ],
        "max_angular_residual_deg": max(
            float(item["angular_residual_deg"]) for item in projections
        ),
    }
    return sample, summary


def _write_cache_manifest(
    *,
    source_manifest: dict[str, Any],
    source_cache_root: Path,
    raw_dataset_root: Path,
    output_root: Path,
    samples: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    seed: int,
    command: str,
) -> Path:
    manifest = dict(source_manifest)
    source_cases = sorted(
        {str(item["source_case_key"]) for item in summaries}
    )
    manifest.update(
        {
            "schema": "x2cbct_difnet_v0b_cache.v2",
            "cache_version": output_root.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_command": command,
            "raw_dataset_root": str(raw_dataset_root.resolve()),
            "views": {VIEW_KEY: source_manifest["views"][VIEW_KEY]},
            "samples": samples,
            "failed_cases": [],
            "target_volume_mode": "full",
            "view_sample_counts": {VIEW_KEY: {"train": 0, "test": len(samples)}},
            "split": {
                "policy": "synthetic_test_only_from_held_out_test_cases",
                "split_unit": "precomputed_case",
                "seed": int(seed),
                "train_ratio": 0.0,
                "train_patient_count": 0,
                "test_patient_count": len(
                    {str(item["patient_id"]) for item in samples}
                ),
                "train_case_count": 0,
                "test_case_count": len(source_cases),
                "patient_overlap": [],
                "case_overlap_count": 0,
                "case_overlap": [],
                "allow_patient_overlap": False,
                "records_manifest_cases": source_cases,
                "train_cases": 0,
                "test_cases": len(source_cases),
                "indexed_train_cases": 0,
                "indexed_test_cases": len(source_cases),
                "has_validation_split": False,
                "test_split_role": "held-out synthetic stress test only",
            },
            "setup_augmentation": {
                "enabled": True,
                "parameter_semantics": (
                    "increments applied to source cases; not total setup errors"
                ),
                "translation_mm": [
                    float(item["applied_translation_mm"]) for item in summaries
                ],
                "rotation_deg": [
                    float(item["applied_yaw_deg"]) for item in summaries
                ],
                "combination_count_per_source_case": 1,
                "requires_zero_bin": False,
                "integer_bins_only": False,
                "decimal_precision": 4,
                "order": "rotation first, then translation",
                "synthetic_stress_test_only": True,
            },
            "source_case_counts": {
                "source_test_cases": len(source_cases),
                "synthetic_test_cases": len(samples),
                "setup_augmentation_combinations_per_source_case": 1,
            },
            "synthetic_setup_stress": {
                "schema": "x2cbct_synthetic_setup_stress.v1",
                "source_cache_root": str(source_cache_root.resolve()),
                "source_cache_manifest": str(
                    (source_cache_root / "cache_manifest.json").resolve()
                ),
                "raw_dataset_root": str(raw_dataset_root.resolve()),
                "source_split_required": "test",
                "view_key": VIEW_KEY,
                "seed": int(seed),
                "target_semantics": (
                    "requested values are total setup errors after perturbation"
                ),
                "scientific_role": "controlled synthetic perturbation stress test",
                "not_real_inter_fraction_setup_error": True,
                "cases": summaries,
            },
        }
    )
    path = output_root / "cache_manifest.json"
    write_json(manifest, path)
    write_json(
        {
            "ok": True,
            "cache_root": str(output_root.resolve()),
            "source_split_required": "test",
            "view_key": VIEW_KEY,
            "sample_count": len(samples),
            "cases": summaries,
        },
        output_root / "reports" / "cache_summary.json",
    )
    return path


def _find_reference(
    cache_root: Path,
    pair: Pair,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest = read_json(cache_root / "cache_manifest.json")
    rows = [
        item
        for item in manifest.get("samples", [])
        if str(item.get("patient_id")) == pair.patient_id
        and str(item.get("case_id")) == pair.reference_case_id
        and str(item.get("view_key")) == VIEW_KEY
    ]
    if not rows:
        raise ValueError(
            f"reference {pair.patient_id}:{pair.reference_case_id} is missing "
            f"from {cache_root}"
        )
    row = rows[0]
    case_manifest = read_json(
        resolve_cache_path(cache_root, str(row["case_manifest_path"]))
    )
    cbct_path = resolve_cache_path(cache_root, str(row["cbct_payload_path"]))
    return cbct_path, case_manifest, manifest


def _write_nifti(
    array: np.ndarray,
    case_manifest: dict[str, Any],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(
        tuple(float(value) for value in case_manifest["processed_spacing_xyz"])
    )
    image.SetOrigin(
        tuple(float(value) for value in case_manifest["processed_origin_xyz"])
    )
    image.SetDirection(
        tuple(float(value) for value in case_manifest["processed_direction"])
    )
    sitk.WriteImage(image, str(path))
    return path


def _validate_registration(
    *,
    specs: list[StressSpec],
    summaries: list[dict[str, Any]],
    output_root: Path,
    reference_cache_root: Path,
    cache_manifest: dict[str, Any],
    roi_reviewed: Path,
    min_verified_si_mm: float,
    min_verified_yaw_deg: float,
    target_tolerance_si_mm: float,
    target_tolerance_yaw_deg: float,
    itk_threads: int,
) -> dict[str, Any]:
    if itk_threads > 0:
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(itk_threads))
    roi_by_case = load_roi_review(roi_reviewed)
    results = []
    initialization_rows: list[dict[str, Any]] = []
    input_root = output_root / "registration_inputs"
    summary_by_experiment = {
        str(item["experiment"]): item for item in summaries
    }

    for spec in specs:
        reference_path, reference_case, reference_manifest = _find_reference(
            reference_cache_root, spec.pair
        )
        synthetic_sample = next(
            item
            for item in cache_manifest["samples"]
            if str(item["synthetic_stress_case"]["experiment"]) == spec.experiment
        )
        moving_path = resolve_cache_path(
            output_root, str(synthetic_sample["cbct_payload_path"])
        )
        moving_case = read_json(
            resolve_cache_path(
                output_root, str(synthetic_sample["case_manifest_path"])
            )
        )
        reference_hu = denormalize_cbct_hu(
            np.load(reference_path).astype(np.float32),
            normalization=reference_manifest["cbct_normalization"],
        )
        moving_hu = denormalize_cbct_hu(
            np.load(moving_path).astype(np.float32),
            normalization=cache_manifest["cbct_normalization"],
        )
        pair_dir = input_root / spec.experiment
        reference_nii = _write_nifti(
            reference_hu,
            reference_case,
            pair_dir / "reference_cbct_hu.nii.gz",
        )
        moving_nii = _write_nifti(
            moving_hu,
            moving_case,
            pair_dir / "synthetic_moving_cbct_hu.nii.gz",
        )
        registration_pair = {
            "patient_id": spec.pair.patient_id,
            "reference_case_id": spec.pair.reference_case_id,
            "moving_case_id": spec.pair.moving_case_id,
            "reference_cbct_path": str(reference_nii),
            "moving_cbct_path": str(moving_nii),
        }
        result, initializations = register_pair(registration_pair, roi_by_case)
        results.append(result)
        for item in initializations:
            initialization_rows.append(
                {
                    "experiment": spec.experiment,
                    "patient_id": spec.pair.patient_id,
                    "reference_case_id": spec.pair.reference_case_id,
                    "moving_case_id": spec.pair.moving_case_id,
                    **asdict(item),
                }
            )
        summary = summary_by_experiment[spec.experiment]
        summary.update(
            {
                "registration_status": result.status,
                "measured_setup_error_SI_mm": result.setup_error_SI_mm,
                "measured_setup_error_SI_mm_4dp": (
                    f"{result.setup_error_SI_mm:.4f}"
                ),
                "measured_setup_error_abs_SI_mm": result.setup_error_abs_SI_mm,
                "measured_setup_error_yaw_deg": result.setup_error_yaw_deg,
                "measured_setup_error_yaw_deg_4dp": (
                    f"{result.setup_error_yaw_deg:.4f}"
                ),
                "measured_setup_error_abs_yaw_deg": (
                    result.setup_error_abs_yaw_deg
                ),
                "target_error_SI_mm": abs(
                    result.setup_error_SI_mm - float(spec.target_si_mm)
                ),
                "target_error_yaw_deg": abs(
                    result.setup_error_yaw_deg - float(spec.target_yaw_deg)
                ),
                "selected_initialization_id": result.selected_initialization_id,
                "registration_notes": result.notes,
            }
        )

    paths = write_registration_outputs(
        output_root / "registration_validation",
        results,
        initialization_rows,
    )
    checks = {
        "si-max": next(
            (
                result.status == "ok"
                and result.setup_error_abs_SI_mm >= float(min_verified_si_mm)
                and abs(result.setup_error_SI_mm - float(spec.target_si_mm))
                <= float(target_tolerance_si_mm)
                for spec, result in zip(specs, results)
                if spec.experiment == "si-max"
            ),
            False,
        ),
        "yaw-max": next(
            (
                result.status == "ok"
                and result.setup_error_abs_yaw_deg >= float(min_verified_yaw_deg)
                and abs(result.setup_error_yaw_deg - float(spec.target_yaw_deg))
                <= float(target_tolerance_yaw_deg)
                for spec, result in zip(specs, results)
                if spec.experiment == "yaw-max"
            ),
            False,
        ),
    }
    report = {
        "ok": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "si_max_abs_SI_mm": float(min_verified_si_mm),
            "yaw_max_abs_yaw_deg": float(min_verified_yaw_deg),
            "si_target_tolerance_mm": float(target_tolerance_si_mm),
            "yaw_target_tolerance_deg": float(target_tolerance_yaw_deg),
        },
        "registration_outputs": paths,
        "cases": summaries,
    }
    write_json(
        report,
        output_root / "registration_validation" / "validation_report.json",
    )
    formatted_path = (
        output_root / "registration_validation" / "registration_results_4dp.csv"
    )
    with formatted_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment",
            "patient_id",
            "reference_case_id",
            "moving_case_id",
            "status",
            "target_setup_error_SI_mm",
            "target_setup_error_yaw_deg",
            "applied_translation_mm",
            "applied_yaw_deg",
            "measured_setup_error_SI_mm",
            "measured_setup_error_yaw_deg",
            "primary_target_abs_error",
            "passes_axis_target",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec, result in zip(specs, results):
            writer.writerow(
                {
                    "experiment": spec.experiment,
                    "patient_id": spec.pair.patient_id,
                    "reference_case_id": spec.pair.reference_case_id,
                    "moving_case_id": spec.pair.moving_case_id,
                    "status": result.status,
                    "target_setup_error_SI_mm": f"{spec.target_si_mm:.4f}",
                    "target_setup_error_yaw_deg": f"{spec.target_yaw_deg:.4f}",
                    "applied_translation_mm": f"{spec.translation_mm:.4f}",
                    "applied_yaw_deg": f"{spec.yaw_deg:.4f}",
                    "measured_setup_error_SI_mm": (
                        f"{result.setup_error_SI_mm:.4f}"
                    ),
                    "measured_setup_error_yaw_deg": (
                        f"{result.setup_error_yaw_deg:.4f}"
                    ),
                    "primary_target_abs_error": (
                        f"{abs(result.setup_error_SI_mm - float(spec.target_si_mm)):.4f}"
                        if spec.experiment == "si-max"
                        else f"{abs(result.setup_error_yaw_deg - float(spec.target_yaw_deg)):.4f}"
                    ),
                    "passes_axis_target": checks[spec.experiment],
                }
            )
    return report


def _resolve_specs(args: argparse.Namespace) -> list[StressSpec]:
    formal_rows = _read_formal_rows(args.formal_registration_results)
    si_baseline = _formal_row(formal_rows, args.si_pair)
    yaw_baseline = _formal_row(formal_rows, args.yaw_pair)
    rng = random.Random(int(args.seed))
    si_target_magnitude = (
        require_four_decimal_noninteger(
            args.si_target_mm,
            name="si_target_mm",
            minimum=Decimal(str(args.min_target_si_mm)),
        )
        if args.si_target_mm is not None
        else draw_four_decimal_noninteger(
            rng,
            low=args.si_random_min_mm,
            high=args.si_random_max_mm,
            name="S-I target",
            minimum=Decimal(str(args.min_target_si_mm)),
        )
    )
    yaw_target_magnitude = (
        require_four_decimal_noninteger(
            args.yaw_target_deg,
            name="yaw_target_deg",
            minimum=Decimal(str(args.min_target_yaw_deg)),
        )
        if args.yaw_target_deg is not None
        else draw_four_decimal_noninteger(
            rng,
            low=args.yaw_random_min_deg,
            high=args.yaw_random_max_deg,
            name="yaw target",
            minimum=Decimal(str(args.min_target_yaw_deg)),
        )
    )
    si_base_value = float(si_baseline["setup_error_SI_mm"])
    yaw_base_value = float(yaw_baseline["setup_error_yaw_deg"])
    si_target = signed_target_from_baseline(
        si_target_magnitude,
        si_base_value,
        mode=args.si_sign,
        rng=rng,
    )
    yaw_target = signed_target_from_baseline(
        yaw_target_magnitude,
        yaw_base_value,
        mode=args.yaw_sign,
        rng=rng,
    )
    si_increment = (
        Decimal(si_baseline["setup_error_SI_mm"]) - si_target
    ).quantize(FOUR_DP)
    yaw_increment = (
        yaw_target - Decimal(yaw_baseline["setup_error_yaw_deg"])
    ).quantize(FOUR_DP)
    return [
        StressSpec(
            experiment="si-max",
            pair=args.si_pair,
            target_si_mm=si_target,
            target_yaw_deg=Decimal(
                si_baseline["setup_error_yaw_deg"]
            ).quantize(FOUR_DP),
            translation_mm=si_increment,
            yaw_deg=Decimal("0.0000"),
            baseline_si_mm=si_base_value,
            baseline_yaw_deg=float(si_baseline["setup_error_yaw_deg"]),
        ),
        StressSpec(
            experiment="yaw-max",
            pair=args.yaw_pair,
            target_si_mm=Decimal(
                yaw_baseline["setup_error_SI_mm"]
            ).quantize(FOUR_DP),
            target_yaw_deg=yaw_target,
            translation_mm=Decimal("0.0000"),
            yaw_deg=yaw_increment,
            baseline_si_mm=float(yaw_baseline["setup_error_SI_mm"]),
            baseline_yaw_deg=yaw_base_value,
        ),
    ]


def _resolved_preview(
    *,
    args: argparse.Namespace,
    specs: list[StressSpec],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    cases = []
    for spec in specs:
        source = _find_source_sample(source_manifest, spec.pair)
        cases.append(
            {
                "experiment": spec.experiment,
                "pair": (
                    f"{spec.pair.patient_id}:{spec.pair.reference_case_id}:"
                    f"{spec.pair.moving_case_id}"
                ),
                "source_sample_id": source["sample_id"],
                "source_split": source["split"],
                "view_key": source["view_key"],
                "target_SI_mm_4dp": f"{spec.target_si_mm:.4f}",
                "target_yaw_deg_4dp": f"{spec.target_yaw_deg:.4f}",
                "applied_translation_increment_mm_4dp": (
                    f"{spec.translation_mm:.4f}"
                ),
                "applied_yaw_increment_deg_4dp": f"{spec.yaw_deg:.4f}",
                "baseline_SI_mm": spec.baseline_si_mm,
                "baseline_yaw_deg": spec.baseline_yaw_deg,
            }
        )
    return {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "source_cache_root": str(args.source_cache_root.resolve()),
        "raw_dataset_root": str(args.raw_dataset_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "source_split_required": "test",
        "view_key": VIEW_KEY,
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create two geometrically synchronized, test-only synthetic setup-error "
            "stress cases: one S-I-dominant and one yaw-dominant."
        )
    )
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument(
        "--raw-dataset-root",
        type=Path,
        default=DEFAULT_RAW_DATASET_ROOT,
        help=(
            "Root containing PATIENT/CASE/CBCT.nii.gz, Geometry, and all raw "
            "projections. Test membership still comes only from source-cache-root."
        ),
    )
    parser.add_argument(
        "--reference-cache-root", type=Path, default=DEFAULT_REFERENCE_CACHE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--formal-registration-results", type=Path, default=DEFAULT_FORMAL_RESULTS
    )
    parser.add_argument("--roi-reviewed", type=Path, default=DEFAULT_ROI_REVIEWED)
    parser.add_argument("--si-pair", type=parse_pair, default=parse_pair("6:0:2"))
    parser.add_argument("--yaw-pair", type=parse_pair, default=parse_pair("0:0:6"))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--si-target-mm",
        help=(
            "Exact non-integer total S-I setup-error target (<=4 decimals); "
            "overrides the random range."
        ),
    )
    parser.add_argument(
        "--yaw-target-deg",
        help=(
            "Exact non-integer total yaw setup-error target (<=4 decimals); "
            "overrides the random range."
        ),
    )
    parser.add_argument("--si-random-min-mm", default="7.2501")
    parser.add_argument("--si-random-max-mm", default="9.7500")
    parser.add_argument("--yaw-random-min-deg", default="8.2501")
    parser.add_argument("--yaw-random-max-deg", default="9.7500")
    parser.add_argument(
        "--si-sign",
        choices=("away-from-zero", "positive", "negative"),
        default="away-from-zero",
    )
    parser.add_argument(
        "--yaw-sign",
        choices=("away-from-zero", "positive", "negative"),
        default="away-from-zero",
    )
    parser.add_argument("--min-target-si-mm", type=float, default=7.0)
    parser.add_argument("--min-target-yaw-deg", type=float, default=8.0)
    parser.add_argument("--min-verified-si-mm", type=float, default=7.0)
    parser.add_argument("--min-verified-yaw-deg", type=float, default=8.0)
    parser.add_argument("--target-tolerance-si-mm", type=float, default=0.15)
    parser.add_argument("--target-tolerance-yaw-deg", type=float, default=0.20)
    parser.add_argument("--registration-itk-threads", type=int, default=4)
    parser.add_argument("--skip-registration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "source_cache_root",
        "raw_dataset_root",
        "reference_cache_root",
        "output_root",
        "formal_registration_results",
        "roi_reviewed",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (PROJECT_ROOT / value).resolve())
        else:
            setattr(args, name, value.resolve())

    source_manifest_path = args.source_cache_root / "cache_manifest.json"
    if not source_manifest_path.exists():
        parser.error(f"source cache manifest is missing: {source_manifest_path}")
    source_manifest = read_json(source_manifest_path)
    specs = _resolve_specs(args)
    preview = _resolved_preview(
        args=args,
        specs=specs,
        source_manifest=source_manifest,
    )
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(
            f"output root is not empty: {args.output_root}; choose a new --output-root"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = []
    summaries = []
    for spec in specs:
        source_sample = _find_source_sample(source_manifest, spec.pair)
        sample, summary = _build_stress_sample(
            spec=spec,
            source_cache_root=args.source_cache_root,
            source_manifest=source_manifest,
            source_sample=source_sample,
            raw_dataset_root=args.raw_dataset_root,
            output_root=args.output_root,
        )
        samples.append(sample)
        summaries.append(summary)
        print(
            f"[generated] {spec.experiment} source={spec.pair.source_case_key} "
            f"split=test target_SI={spec.target_si_mm:.4f}mm "
            f"target_yaw={spec.target_yaw_deg:.4f}deg "
            f"delta_SI={spec.translation_mm:.4f}mm "
            f"delta_yaw={spec.yaw_deg:.4f}deg",
            flush=True,
        )

    manifest_path = _write_cache_manifest(
        source_manifest=source_manifest,
        source_cache_root=args.source_cache_root,
        raw_dataset_root=args.raw_dataset_root,
        output_root=args.output_root,
        samples=samples,
        summaries=summaries,
        seed=args.seed,
        command=" ".join(sys.argv),
    )
    cache_manifest = read_json(manifest_path)
    report = None
    if not args.skip_registration:
        report = _validate_registration(
            specs=specs,
            summaries=summaries,
            output_root=args.output_root,
            reference_cache_root=args.reference_cache_root,
            cache_manifest=cache_manifest,
            roi_reviewed=args.roi_reviewed,
            min_verified_si_mm=args.min_verified_si_mm,
            min_verified_yaw_deg=args.min_verified_yaw_deg,
            target_tolerance_si_mm=args.target_tolerance_si_mm,
            target_tolerance_yaw_deg=args.target_tolerance_yaw_deg,
            itk_threads=args.registration_itk_threads,
        )
        cache_manifest["synthetic_setup_stress"]["cases"] = summaries
        cache_manifest["synthetic_setup_stress"]["registration_validation"] = report
        write_json(cache_manifest, manifest_path)

    result = {
        **preview,
        "dry_run": False,
        "cache_manifest": str(manifest_path.resolve()),
        "registration_validation": report,
    }
    write_json(result, args.output_root / "run_summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if report is None or bool(report["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
