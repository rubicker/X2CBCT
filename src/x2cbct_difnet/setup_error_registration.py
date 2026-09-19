from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from x2cbct_difnet.setup_error_roi import load_cbct, read_roi_csv, roi_mask_from_candidate


@dataclass(frozen=True)
class InitializationResult:
    initialization_id: str
    metric_value: float
    similarity_score: float
    optimizer_stop: str
    success: bool
    error: str
    LR_mm: float
    AP_mm: float
    SI_mm: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    setup_error_SI_mm: float
    setup_error_abs_SI_mm: float
    setup_error_yaw_deg: float
    setup_error_abs_yaw_deg: float


@dataclass(frozen=True)
class RegistrationResult:
    patient_id: str
    reference_case_id: str
    moving_case_id: str
    status: str
    LR_mm: float
    AP_mm: float
    SI_mm: float
    translation_norm_mm: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    rotation_norm_deg: float
    setup_error_SI_mm: float
    setup_error_abs_SI_mm: float
    setup_error_yaw_deg: float
    setup_error_abs_yaw_deg: float
    similarity_score: float
    selected_initialization_id: str
    roi_status: str
    notes: str


def physical_centroid(image: sitk.Image, mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        raise ValueError("ROI mask is empty")
    zz, yy, xx = np.where(mask)
    index_mean = (float(xx.mean()), float(yy.mean()), float(zz.mean()))
    return tuple(float(v) for v in image.TransformContinuousIndexToPhysicalPoint(index_mean))


def mask_to_sitk(mask: np.ndarray, reference: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.CopyInformation(reference)
    return img


def make_initial_transform(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    delta_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sitk.Euler3DTransform:
    fixed_center = physical_centroid(fixed_image, fixed_mask)
    moving_center = physical_centroid(moving_image, moving_mask)
    transform = sitk.Euler3DTransform()
    transform.SetCenter(fixed_center)
    transform.SetTranslation(
        tuple(moving_center[i] - fixed_center[i] + float(delta_xyz[i]) for i in range(3))
    )
    transform.SetRotation(*(math.radians(v) for v in delta_angles_deg))
    return transform


def initialization_specs() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    specs: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
        ("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    ]
    axes = ("LR", "AP", "SI")
    for axis_index, axis_name in enumerate(axes):
        for sign in (-1.0, 1.0):
            delta = [0.0, 0.0, 0.0]
            delta[axis_index] = sign * 3.0
            specs.append((f"{axis_name}_{sign * 3.0:+.0f}mm", tuple(delta), (0.0, 0.0, 0.0)))
    angles = ("pitch", "roll", "yaw")
    for axis_index, axis_name in enumerate(angles):
        for sign in (-1.0, 1.0):
            delta_angles = [0.0, 0.0, 0.0]
            delta_angles[axis_index] = sign * 2.0
            specs.append((f"{axis_name}_{sign * 2.0:+.0f}deg", (0.0, 0.0, 0.0), tuple(delta_angles)))
    return specs


def configure_registration(fixed_mask_img: sitk.Image, moving_mask_img: sitk.Image) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    method.SetMetricFixedMask(fixed_mask_img)
    method.SetMetricMovingMask(moving_mask_img)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.20, seed=20260531)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.25,
        minStep=0.001,
        numberOfIterations=80,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def crop_image_and_mask(
    image: sitk.Image,
    mask: np.ndarray,
    row: dict[str, Any],
    margin: int = 12,
) -> tuple[sitk.Image, np.ndarray]:
    z_size, y_size, x_size = mask.shape
    margin_x = int(row.get("crop_margin_x_voxels", margin))
    margin_y = int(row.get("crop_margin_y_voxels", margin))
    margin_z = int(row.get("crop_margin_z_voxels", margin))
    x0 = max(0, int(row["x_min"]) - margin_x)
    y0 = max(0, int(row["y_min"]) - margin_y)
    z0 = max(0, int(row["z_min"]) - margin_z)
    x1 = min(x_size - 1, int(row["x_max"]) + margin_x)
    y1 = min(y_size - 1, int(row["y_max"]) + margin_y)
    z1 = min(z_size - 1, int(row["z_max"]) + margin_z)
    size = [x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1]
    index = [x0, y0, z0]
    cropped_image = sitk.RegionOfInterest(image, size=size, index=index)
    cropped_mask = mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    return cropped_image, cropped_mask


def moving_to_reference_components(transform_fixed_to_moving: sitk.Transform) -> tuple[float, float, float, float, float, float]:
    inverse = transform_fixed_to_moving.GetInverse()
    params = inverse.GetParameters()
    if len(params) == 6:
        angle_x, angle_y, angle_z = [math.degrees(float(v)) for v in params[:3]]
        tx, ty, tz = [float(v) for v in params[3:6]]
        return tx, ty, tz, angle_x, angle_y, angle_z
    if len(params) == 12:
        matrix = np.asarray(params[:9], dtype=float).reshape(3, 3)
        angle_x, angle_y, angle_z = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
        tx, ty, tz = [float(v) for v in params[9:12]]
        return tx, ty, tz, float(angle_x), float(angle_y), float(angle_z)
    raise ValueError(f"Unsupported inverse transform parameter count: {len(params)}")


def primary_setup_error_components(si_mm: float, yaw_deg: float) -> dict[str, float]:
    return {
        "setup_error_SI_mm": float(si_mm),
        "setup_error_abs_SI_mm": abs(float(si_mm)),
        "setup_error_yaw_deg": float(yaw_deg),
        "setup_error_abs_yaw_deg": abs(float(yaw_deg)),
    }


def register_pair(
    pair: dict[str, Any],
    roi_by_case: dict[tuple[str, str], dict[str, Any]],
) -> tuple[RegistrationResult, list[InitializationResult]]:
    patient_id = str(pair["patient_id"])
    ref_case = str(pair["reference_case_id"])
    mov_case = str(pair["moving_case_id"])
    ref_roi = roi_by_case.get((patient_id, ref_case))
    mov_roi = roi_by_case.get((patient_id, mov_case))
    if ref_roi is None or mov_roi is None:
        return failed_result(pair, "missing_roi", "missing ROI row"), []
    if ref_roi.get("status") != "accepted" or mov_roi.get("status") != "accepted":
        return failed_result(pair, "skipped", "reference or moving ROI is not accepted"), []

    if not roi_boxes_compatible(ref_roi, mov_roi):
        return failed_result(pair, "skipped", "reference and moving ROI boxes are not compatible"), []

    fixed_image, fixed_array = load_cbct(Path(pair["reference_cbct_path"]))
    moving_image, moving_array = load_cbct(Path(pair["moving_cbct_path"]))
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    fixed_mask = roi_mask_from_candidate(fixed_array, ref_roi)
    moving_mask = roi_mask_from_candidate(moving_array, mov_roi)
    if int(fixed_mask.sum()) < 100 or int(moving_mask.sum()) < 100:
        return failed_result(pair, "failed", "ROI mask has fewer than 100 voxels"), []

    fixed_reg_image, fixed_reg_mask = crop_image_and_mask(fixed_image, fixed_mask, ref_roi)
    moving_reg_image, moving_reg_mask = crop_image_and_mask(moving_image, moving_mask, mov_roi)
    fixed_mask_img = mask_to_sitk(fixed_reg_mask, fixed_reg_image)
    moving_mask_img = mask_to_sitk(moving_reg_mask, moving_reg_image)
    results: list[InitializationResult] = []
    best_transform: sitk.Transform | None = None
    best_score = -float("inf")
    best_id = ""

    for init_id, delta_xyz, delta_angles in initialization_specs():
        try:
            initial = make_initial_transform(
                fixed_image,
                moving_image,
                fixed_mask,
                moving_mask,
                delta_xyz=delta_xyz,
                delta_angles_deg=delta_angles,
            )
            method = configure_registration(fixed_mask_img, moving_mask_img)
            method.SetInitialTransform(initial, inPlace=False)
            final = method.Execute(fixed_reg_image, moving_reg_image)
            metric = float(method.GetMetricValue())
            score = -metric
            tx, ty, tz, pitch, roll, yaw = moving_to_reference_components(final)
            stop = str(method.GetOptimizerStopConditionDescription())
            results.append(
                InitializationResult(
                    init_id,
                    metric,
                    score,
                    stop,
                    True,
                    "",
                    tx,
                    ty,
                    tz,
                    pitch,
                    roll,
                    yaw,
                    **primary_setup_error_components(tz, yaw),
                )
            )
            if score > best_score:
                best_score = score
                best_transform = final
                best_id = init_id
        except Exception as exc:
            results.append(
                InitializationResult(
                    init_id,
                    float("nan"),
                    -float("inf"),
                    "",
                    False,
                    str(exc),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    **primary_setup_error_components(0.0, 0.0),
                )
            )

    if best_transform is None:
        return failed_result(pair, "failed", "all initializations failed"), results
    tx, ty, tz, pitch, roll, yaw = moving_to_reference_components(best_transform)
    notes = anomaly_notes(tx, ty, tz, pitch, roll, yaw)
    return (
        RegistrationResult(
            patient_id=patient_id,
            reference_case_id=ref_case,
            moving_case_id=mov_case,
            status="ok",
            LR_mm=tx,
            AP_mm=ty,
            SI_mm=tz,
            translation_norm_mm=float(np.linalg.norm([tx, ty, tz])),
            pitch_deg=pitch,
            roll_deg=roll,
            yaw_deg=yaw,
            rotation_norm_deg=float(np.linalg.norm([pitch, roll, yaw])),
            **primary_setup_error_components(tz, yaw),
            similarity_score=best_score,
            selected_initialization_id=best_id,
            roi_status="accepted",
            notes=notes,
        ),
        results,
    )


def anomaly_notes(tx: float, ty: float, tz: float, pitch: float, roll: float, yaw: float) -> str:
    notes = []
    if float(np.linalg.norm([tx, ty, tz])) > 30.0:
        notes.append("large_translation_needs_review")
    if float(np.linalg.norm([pitch, roll, yaw])) > 10.0:
        notes.append("large_rotation_needs_review")
    return ";".join(notes)


def failed_result(pair: dict[str, Any], status: str, note: str) -> RegistrationResult:
    return RegistrationResult(
        patient_id=str(pair["patient_id"]),
        reference_case_id=str(pair["reference_case_id"]),
        moving_case_id=str(pair["moving_case_id"]),
        status=status,
        LR_mm=0.0,
        AP_mm=0.0,
        SI_mm=0.0,
        translation_norm_mm=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_deg=0.0,
        rotation_norm_deg=0.0,
        **primary_setup_error_components(0.0, 0.0),
        similarity_score=float("nan"),
        selected_initialization_id="",
        roi_status="",
        notes=note,
    )


def roi_boxes_compatible(ref_roi: dict[str, Any], mov_roi: dict[str, Any], *, max_extent_delta_mm: int = 25) -> bool:
    for axis in ("x", "y", "z"):
        ref_extent = int(ref_roi[f"{axis}_max"]) - int(ref_roi[f"{axis}_min"])
        mov_extent = int(mov_roi[f"{axis}_max"]) - int(mov_roi[f"{axis}_min"])
        if abs(ref_extent - mov_extent) > max_extent_delta_mm:
            return False
    return True


def write_registration_outputs(
    output_dir: Path,
    results: list[RegistrationResult],
    init_rows: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = output_dir / "registration_results.csv"
    init_csv = output_dir / "registration_initializations.csv"
    result_json = output_dir / "registration_results.json"
    write_dataclass_csv([asdict(r) for r in results], result_csv)
    write_dataclass_csv(init_rows, init_csv)
    result_json.write_text(
        json.dumps({"schema": "x2cbct_setup_error_registration.v1", "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"results_csv": str(result_csv), "initializations_csv": str(init_csv), "results_json": str(result_json)}


def write_dataclass_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_roi_review(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return read_roi_csv(path)
