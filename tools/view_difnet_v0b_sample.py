from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json, resolve_cache_path, write_json
from x2cbct_difnet.cached_dataset import load_cache_manifest, samples_for
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count
from x2cbct_difnet.volume_grid import plane_spacing, processed_grid_from_case_manifest, resize_plane_to_spacing


def _scalar(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def _load_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as raw:
        return {
            "sample_id": _scalar(raw["sample_id"]),
            "split": _scalar(raw["split"]),
            "view_key": _scalar(raw["view_key"]),
            "case_manifest_path": _scalar(raw["case_manifest_path"]),
            "cbct_payload_path": _scalar(raw["cbct_payload_path"]),
            "projection_payload_paths": [str(item) for item in raw["projection_payload_paths"].tolist()],
            "projection_files": [str(item) for item in raw["projection_files"].tolist()],
            "target_angles_deg": raw["target_angles_deg"].astype(float).tolist(),
            "gantry_angles_deg": raw["gantry_angles_deg"].astype(float).tolist(),
            "angular_residuals_deg": raw["angular_residuals_deg"].astype(float).tolist(),
        }


def _resolve_data_path(cache_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else resolve_cache_path(cache_root, str(value))


def _projection_display_array(array: np.ndarray, value_space: object = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if str(value_space or "").endswith("_0_255") or float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return arr


def _view_cache_sample_payload(sample: dict[str, object]) -> dict[str, object]:
    angles = [float(item) for item in sample.get("angles_deg", [])]  # type: ignore[arg-type]
    projection_paths = [str(item) for item in sample.get("projection_paths", [])]  # type: ignore[arg-type]
    return {
        "sample_id": str(sample["sample_id"]),
        "split": str(sample["split"]),
        "view_key": str(sample["view_key"]),
        "case_manifest_path": "",
        "cbct_payload_path": str(sample["processed_cbct_path"]),
        "projection_payload_paths": projection_paths,
        "projection_files": projection_paths,
        "target_angles_deg": angles,
        "gantry_angles_deg": angles,
        "angular_residuals_deg": [0.0 for _ in angles],
    }


def _to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _write_png(
    path: Path,
    array: np.ndarray,
    label: str = "",
    *,
    plane: str | None = None,
    spacing_xyz: list[float] | None = None,
) -> Path:
    img = _to_uint8(array)
    if plane is not None and spacing_xyz is not None:
        img = resize_plane_to_spacing(img, plane, spacing_xyz)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if label:
        cv2.putText(img, label[:70], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "sample"


def _slice_array(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"unknown plane: {plane}")


def _plane_size(volume: np.ndarray, plane: str) -> int:
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[plane]
    return int(volume.shape[axis])


def _export_plane(
    volume: np.ndarray,
    plane: str,
    out_dir: Path,
    root: Path,
    stride: int,
    spacing_xyz: list[float],
) -> dict[str, object]:
    count = _plane_size(volume, plane)
    indices = list(range(0, count, max(1, stride)))
    center = count // 2
    if center not in indices:
        indices.append(center)
        indices.sort()
    slices = []
    row_spacing, column_spacing = plane_spacing(plane, spacing_xyz)
    for index in indices:
        png = out_dir / "assets" / "cbct" / plane / f"{plane}_{index:03d}.png"
        _write_png(png, _slice_array(volume, plane, index), f"{plane} {index}", plane=plane, spacing_xyz=spacing_xyz)
        slices.append({"index": index, "png": _rel(png, root)})
    return {
        "count": count,
        "center_index": center,
        "row_spacing_mm": row_spacing,
        "column_spacing_mm": column_spacing,
        "slices": slices,
    }


def _array_summary(array: np.ndarray) -> dict[str, object]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _find_sample_for_path(cache_root: Path, manifest: dict[str, object], sample_path: Path) -> tuple[dict[str, object], str, str, int]:
    resolved = sample_path.resolve()
    samples = list(manifest["samples"])  # type: ignore[index]
    for index, sample in enumerate(samples):
        if "sample_path" not in sample:
            continue
        candidate = resolve_cache_path(cache_root, str(sample["sample_path"]))
        if candidate.resolve() == resolved:
            return sample, str(sample["split"]), str(sample["view_key"]), index
    raise ValueError(f"sample path is not listed in cache manifest: {sample_path}")


def _sample_identity(sample: dict[str, object], payload: dict[str, object]) -> tuple[str, str, str]:
    sample_id = str(payload["sample_id"])
    case_id = str(sample.get("case_id") or "")
    if case_id:
        patient_id = str(sample.get("patient_id") or sample.get("dataset_group") or sample.get("source") or "case")
        case_key = f"{patient_id}_{case_id}"
        return patient_id, case_id, case_key
    parts = sample_id.split("_")
    patient_id = parts[0].replace("patient", "") if len(parts) >= 1 else "unknown"
    case_id = parts[1].replace("case", "") if len(parts) >= 2 else "unknown"
    return patient_id, case_id, f"{patient_id}_{case_id}"


def _grid_from_view_cache_sample(sample: dict[str, object], cbct_shape: tuple[int, ...]) -> dict[str, object]:
    spacing = sample.get("processed_spacing_xyz") or [1.0, 1.0, 1.0]
    origin = sample.get("processed_origin_xyz") or [0.0, 0.0, 0.0]
    direction = sample.get("processed_direction") or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return {
        "shape_zyx": [int(item) for item in cbct_shape],
        "spacing_xyz": [float(item) for item in spacing],  # type: ignore[arg-type]
        "origin_xyz": [float(item) for item in origin],  # type: ignore[arg-type]
        "direction": [float(item) for item in direction],  # type: ignore[arg-type]
    }


def _viewer_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DIF-Net V0b Interactive Viewer</title>
  <style>
    :root {
      --bg: #f5f7f6;
      --panel: #ffffff;
      --ink: #17201c;
      --muted: #65716b;
      --line: #d7ded9;
      --accent: #0f766e;
      --warn: #a16207;
      --canvas: #0b0f0d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0 0 4px; font-size: 21px; font-weight: 650; }
    h2 { margin: 0 0 10px; font-size: 16px; font-weight: 650; }
    h3 { margin: 0 0 8px; font-size: 13px; color: var(--muted); font-weight: 600; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(430px, 1.45fr);
      gap: 12px;
      padding: 12px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .image-frame {
      background: var(--canvas);
      border-radius: 6px;
      min-height: 260px;
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .image-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: pixelated;
      filter: brightness(var(--brightness, 1)) contrast(var(--contrast, 1));
      transform: scale(var(--zoom, 1));
      transform-origin: center;
    }
    .controls {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 8px 10px;
      align-items: center;
      margin: 10px 0;
    }
    label { color: var(--muted); font-size: 13px; }
    select, input[type="range"] { width: 100%; }
    select {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 8px;
    }
    button {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    .volume-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .plane { min-width: 0; }
    .plane .image-frame { min-height: 190px; }
    .meta-line { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .kv {
      display: grid;
      grid-template-columns: 132px 1fr;
      gap: 6px 10px;
      font-size: 13px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    .pill {
      display: inline-block;
      padding: 3px 7px;
      border-radius: 999px;
      background: #e7f3f1;
      color: var(--accent);
      font-size: 12px;
      margin-right: 6px;
    }
    .warn { color: var(--warn); }
    code { font-family: Consolas, monospace; overflow-wrap: anywhere; }
    .nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      align-items: center;
      margin-top: 8px;
    }
    .nav a {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      text-decoration: none;
      font-size: 13px;
    }
    .nav a:hover { border-color: var(--accent); color: var(--accent); }
    .nav a.disabled {
      pointer-events: none;
      opacity: 0.42;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px 10px;
      padding: 10px 0 2px;
    }
    @media (max-width: 1080px) {
      main { grid-template-columns: 1fr; }
      .volume-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>DIF-Net V0b Interactive Viewer</h1>
      <div class="meta-line" id="caseTitle"></div>
      <div class="nav" id="sampleNav"></div>
    </div>
    <div class="meta-line" id="cacheTitle"></div>
  </header>
  <main>
    <section>
      <h2>Projection</h2>
      <div class="image-frame"><img id="projectionImage" alt="cached projection"></div>
      <div class="controls">
        <label for="projectionSlider">Projection</label>
        <input id="projectionSlider" type="range" min="0" value="0">
      </div>
      <div class="meta-line" id="projectionIndex"></div>
      <div class="kv" id="projectionMeta"></div>
    </section>
    <section>
      <h2>Target CBCT</h2>
      <div class="toolbar">
        <label for="brightnessSlider">Brightness</label>
        <input id="brightnessSlider" type="range" min="50" max="200" value="100">
        <label for="contrastSlider">Contrast</label>
        <input id="contrastSlider" type="range" min="50" max="220" value="100">
        <label for="zoomSlider">Zoom</label>
        <input id="zoomSlider" type="range" min="100" max="300" value="100">
      </div>
      <div class="volume-grid">
        <div class="plane" data-plane="axial">
          <h3>Axial</h3>
          <div class="image-frame"><img id="axialImage" alt="axial CBCT"></div>
          <div class="controls">
            <label for="axialSlider">Slice</label>
            <input id="axialSlider" type="range" min="0">
          </div>
          <div class="meta-line" id="axialIndex"></div>
        </div>
        <div class="plane" data-plane="coronal">
          <h3>Coronal</h3>
          <div class="image-frame"><img id="coronalImage" alt="coronal CBCT"></div>
          <div class="controls">
            <label for="coronalSlider">Slice</label>
            <input id="coronalSlider" type="range" min="0">
          </div>
          <div class="meta-line" id="coronalIndex"></div>
        </div>
        <div class="plane" data-plane="sagittal">
          <h3>Sagittal</h3>
          <div class="image-frame"><img id="sagittalImage" alt="sagittal CBCT"></div>
          <div class="controls">
            <label for="sagittalSlider">Slice</label>
            <input id="sagittalSlider" type="range" min="0">
          </div>
          <div class="meta-line" id="sagittalIndex"></div>
        </div>
      </div>
      <div class="kv" id="targetMeta"></div>
    </section>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const DATA = window.VIEWER_DATA;
    const projectionSlider = document.getElementById("projectionSlider");

    function text(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (Array.isArray(value)) return value.join(" x ");
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }

    function renderKv(id, rows) {
      const node = document.getElementById(id);
      node.innerHTML = rows.map(([k, v]) => `<div>${k}</div><div><code>${text(v)}</code></div>`).join("");
    }

    function renderNavigation() {
      const node = document.getElementById("sampleNav");
      const nav = DATA.navigation;
      if (!nav) {
        node.innerHTML = "";
        return;
      }
      const link = (item, fallback, disabledText) => {
        if (!item || !item.href) return `<a class="disabled" href="#">${disabledText}</a>`;
        const title = item.label || fallback;
        return `<a href="${item.href}">${title}</a>`;
      };
      const position = nav.total ? `<span class="meta-line">${nav.position} / ${nav.total}</span>` : "";
      node.innerHTML = [
        link(nav.index, "Index", "Index"),
        link(nav.previous, "Previous", "Previous"),
        link(nav.next, "Next", "Next"),
        position,
      ].join("");
      window.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft" && nav.previous && nav.previous.href) window.location.href = nav.previous.href;
        if (event.key === "ArrowRight" && nav.next && nav.next.href) window.location.href = nav.next.href;
      });
    }

    function setProjection() {
      const items = DATA.projections;
      projectionSlider.max = Math.max(items.length - 1, 0);
      const idx = Math.min(Number(projectionSlider.value), items.length - 1);
      projectionSlider.value = idx;
      const item = items[idx];
      document.getElementById("projectionImage").src = item.png;
      document.getElementById("projectionIndex").textContent = `${DATA.view_key} projection ${idx + 1} / ${items.length}`;
      renderKv("projectionMeta", [
        ["file", item.file],
        ["target angle", item.target_angle_deg],
        ["gantry angle", item.gantry_angle_deg],
        ["residual deg", item.angular_residual_deg],
        ["payload", item.payload_path],
        ["summary", item.summary],
      ]);
    }

    function closestSliceIndex(planeData, targetIndex) {
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < planeData.slices.length; i++) {
        const dist = Math.abs(planeData.slices[i].index - targetIndex);
        if (dist < bestDist) {
          best = i;
          bestDist = dist;
        }
      }
      return best;
    }

    function setPlane(plane) {
      const planeData = DATA.cbct.planes[plane];
      const slider = document.getElementById(`${plane}Slider`);
      const image = document.getElementById(`${plane}Image`);
      const index = document.getElementById(`${plane}Index`);
      slider.max = planeData.slices.length - 1;
      const idx = Math.min(Number(slider.value), planeData.slices.length - 1);
      slider.value = idx;
      const current = planeData.slices[idx];
      image.src = current.png;
      index.textContent = `${plane} slice index ${current.index} (${idx + 1} / ${planeData.slices.length})`;
    }

    function setImageFilters() {
      const brightness = Number(document.getElementById("brightnessSlider").value) / 100;
      const contrast = Number(document.getElementById("contrastSlider").value) / 100;
      const zoom = Number(document.getElementById("zoomSlider").value) / 100;
      document.documentElement.style.setProperty("--brightness", brightness);
      document.documentElement.style.setProperty("--contrast", contrast);
      document.documentElement.style.setProperty("--zoom", zoom);
    }

    document.getElementById("caseTitle").innerHTML =
      `<span class="pill">${DATA.split}</span><span class="pill">${DATA.view_key}</span> patient ${DATA.patient_id} / case ${DATA.case_id}`;
    document.getElementById("cacheTitle").innerHTML = `<code>${DATA.cache_root}</code>`;
    renderNavigation();
    projectionSlider.addEventListener("input", setProjection);

    for (const plane of ["axial", "coronal", "sagittal"]) {
      const planeData = DATA.cbct.planes[plane];
      const slider = document.getElementById(`${plane}Slider`);
      slider.value = closestSliceIndex(planeData, planeData.center_index);
      slider.addEventListener("input", () => setPlane(plane));
      setPlane(plane);
    }

    for (const id of ["brightnessSlider", "contrastSlider", "zoomSlider"]) {
      document.getElementById(id).addEventListener("input", setImageFilters);
    }
    setImageFilters();

    renderKv("targetMeta", [
      ["CBCT shape", DATA.cbct.shape],
      ["CBCT payload", DATA.cbct.payload_path],
      ["CBCT summary", DATA.cbct.summary],
      ["CBCT rule", DATA.normalization.cbct],
      ["projection rule", DATA.normalization.projection],
      ["sample path", DATA.sample_path],
      ["raw case path", DATA.raw_case_path],
    ]);
    setProjection();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a DIF-Net V0b cached sample viewer.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=2)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-path", type=Path, help="Exact .npz sample path under the cache. Overrides --split/--view-count/--index.")
    parser.add_argument("--slice-stride", type=int, default=1, help="Export every Nth CBCT slice per plane; center slice is always included.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    manifest = load_cache_manifest(cache_root)
    if args.sample_path is not None:
        sample, split, view_key, index = _find_sample_for_path(cache_root, manifest, args.sample_path)
    else:
        view_key = view_key_from_count(args.view_count)
        split = args.split
        index = args.index
        samples = samples_for(manifest, split, view_key)
        sample = samples[index]
    if "sample_path" in sample:
        sample_path = resolve_cache_path(cache_root, sample["sample_path"])
        payload = _load_npz(sample_path)
    else:
        sample_path = cache_root / "cache_manifest.json"
        payload = _view_cache_sample_payload(sample)
    default_name = _safe_name(f"{manifest['cache_version']}_{payload['sample_id']}_{split}_{view_key}")
    out_dir = args.output_dir or PROJECT_ROOT / "runs" / "viewers" / "difnet_v0b_sample" / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    projections = []
    projection_value_space = sample.get("projection_value_space", manifest.get("projection_value_space"))
    for idx, rel_path in enumerate(payload["projection_payload_paths"]):
        arr = np.load(_resolve_data_path(cache_root, rel_path)).astype(np.float32)
        label = f"{payload['projection_files'][idx]} angle={payload['gantry_angles_deg'][idx]:.2f}"
        png = _write_png(
            out_dir / "assets" / "projections" / f"projection_{idx:02d}.png",
            _projection_display_array(arr, projection_value_space),
            label,
        )
        projections.append(
            {
                "index": idx,
                "file": payload["projection_files"][idx],
                "payload_path": rel_path,
                "target_angle_deg": payload["target_angles_deg"][idx],
                "gantry_angle_deg": payload["gantry_angles_deg"][idx],
                "angular_residual_deg": payload["angular_residuals_deg"][idx],
                "summary": _array_summary(arr),
                "png": _rel(png, out_dir),
            }
        )

    if payload["case_manifest_path"]:
        case_manifest = read_json(resolve_cache_path(cache_root, str(payload["case_manifest_path"])))
        grid = processed_grid_from_case_manifest(case_manifest)
        raw_case_path = case_manifest.get("raw_case_path")
    else:
        case_manifest = {}
        raw_case_path = sample.get("raw_case_path") or sample.get("processed_cbct_path")
    cbct = np.load(_resolve_data_path(cache_root, payload["cbct_payload_path"])).astype(np.float32)
    if not case_manifest:
        grid = _grid_from_view_cache_sample(sample, tuple(int(item) for item in cbct.shape))
    spacing_xyz = [float(item) for item in grid["spacing_xyz"]]
    z, y, x = [dim // 2 for dim in cbct.shape]
    slice_paths = {
        "axial": str(_write_png(out_dir / "cbct_axial.png", cbct[z, :, :], "CBCT axial center", plane="axial", spacing_xyz=spacing_xyz)),
        "coronal": str(_write_png(out_dir / "cbct_coronal.png", cbct[:, y, :], "CBCT coronal center", plane="coronal", spacing_xyz=spacing_xyz)),
        "sagittal": str(_write_png(out_dir / "cbct_sagittal.png", cbct[:, :, x], "CBCT sagittal center", plane="sagittal", spacing_xyz=spacing_xyz)),
    }
    patient_id, case_id, case_key = _sample_identity(sample, payload)
    cbct_normalization = sample.get("cbct_normalization", manifest.get("cbct_normalization", {}))
    projection_normalization = manifest.get(
        "projection_normalization",
        {
            "projection_value_space": projection_value_space,
            "projection_loader_scale": sample.get("projection_loader_scale", manifest.get("projection_loader_scale")),
        },
    )
    viewer_data = {
        "schema": "x2cbct_difnet_v0b_interactive_viewer.v1",
        "cache_root": str(cache_root),
        "cache_version": manifest["cache_version"],
        "sample_path": str(sample_path),
        "sample_manifest_index": index,
        "sample_id": payload["sample_id"],
        "split": split,
        "view_key": view_key,
        "patient_id": patient_id,
        "case_id": case_id,
        "case_key": case_key,
        "raw_case_path": raw_case_path,
        "geometry_branch": case_manifest.get("geometry_branch", manifest.get("geometry_branch")),
        "projections": projections,
        "cbct": {
            "payload_path": payload["cbct_payload_path"],
            "shape": list(cbct.shape),
            "processed_spacing_xyz": spacing_xyz,
            "processed_origin_xyz": grid["origin_xyz"],
            "processed_direction": grid["direction"],
            "display_policy": "spacing_corrected_by_processed_spacing_xyz",
            "summary": _array_summary(cbct),
            "planes": {
                plane: _export_plane(cbct, plane, out_dir, out_dir, args.slice_stride, spacing_xyz)
                for plane in ("axial", "coronal", "sagittal")
            },
        },
        "normalization": {
            "cbct": cbct_normalization,
            "projection": projection_normalization,
        },
    }
    viewer_data_path = out_dir / "viewer_data.json"
    viewer_data_js_path = out_dir / "viewer_data.js"
    viewer_data_path.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    viewer_data_js_path.write_text(
        "window.VIEWER_DATA = " + json.dumps(viewer_data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    summary = {
        "ok": True,
        "viewer_kind": "interactive",
        "cache_root": str(cache_root),
        "sample_path": str(sample_path),
        "sample": payload,
        "case_manifest": case_manifest,
        "projection_pngs": [str(out_dir / item["png"]) for item in projections],
        "cbct_slice_pngs": slice_paths,
        "viewer_data": str(viewer_data_path),
        "viewer_data_js": str(viewer_data_js_path),
        "normalization": {
            "cbct": cbct_normalization,
            "projection": projection_normalization,
        },
    }
    summary_path = write_json(summary, out_dir / "viewer_summary.json")
    html_path = out_dir / "index.html"
    html_path.write_text(_viewer_html(), encoding="utf-8")
    print(json.dumps({"ok": True, "kind": "interactive", "summary": str(summary_path), "html": str(html_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
