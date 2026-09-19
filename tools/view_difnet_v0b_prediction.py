from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.volume_grid import plane_spacing, resize_plane_to_spacing


DISPLAY_WINDOW_PERCENTILE = 99.5
ERROR_WINDOW_PERCENTILE = 99.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "case"


def _summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _display_window(array: np.ndarray, error: bool = False) -> dict[str, Any]:
    arr = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return {"mode": "empty", "low": 0.0, "high": 1.0}
    low = 0.0
    percentile = ERROR_WINDOW_PERCENTILE if error else DISPLAY_WINDOW_PERCENTILE
    high = float(np.percentile(values, percentile))
    if high <= low + 1e-8:
        high = float(np.max(values))
    if high <= low + 1e-8:
        high = 1.0
    return {"mode": "auto_percentile", "low": low, "high": high, "percentile": percentile}


def _to_panel(array: np.ndarray, display_window: dict[str, Any] | None = None, error: bool = False) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    window = display_window or _display_window(arr, error=error)
    low = float(window["low"])
    high = float(window["high"])
    denom = max(high - low, 1e-8)
    gray = np.clip((arr - low) / denom, 0.0, 1.0)
    if error:
        panel = cv2.applyColorMap((gray * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    else:
        panel = cv2.cvtColor((gray * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return panel


def _write_png(
    path: Path,
    array: np.ndarray,
    display_window: dict[str, Any],
    *,
    plane: str,
    spacing_xyz: list[float],
    error: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = _to_panel(array, display_window=display_window, error=error)
    panel = resize_plane_to_spacing(panel, plane, spacing_xyz)
    cv2.imwrite(str(path), panel)
    return path


def _slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"unknown plane: {plane}")


def _plane_size(shape: tuple[int, int, int], plane: str) -> int:
    return int(shape[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]])


def _slice_indices(shape: tuple[int, int, int], plane: str, stride: int) -> list[int]:
    size = _plane_size(shape, plane)
    indices = list(range(0, size, max(1, stride)))
    center = size // 2
    if center not in indices:
        indices.append(center)
        indices.sort()
    return indices


def _export_volume(
    volume: np.ndarray,
    volume_key: str,
    case_dir: Path,
    root: Path,
    stride: int,
    spacing_xyz: list[float],
    error: bool = False,
) -> dict[str, Any]:
    shape = tuple(int(item) for item in volume.shape)
    display_window = _display_window(volume, error=error)
    planes = {}
    for plane in ("axial", "coronal", "sagittal"):
        plane_slices = []
        center = _plane_size(shape, plane) // 2
        row_spacing, column_spacing = plane_spacing(plane, spacing_xyz)
        for index in _slice_indices(shape, plane, stride):
            png = case_dir / volume_key / f"{plane}_{index:03d}.png"
            _write_png(
                png,
                _slice(volume, plane, index),
                display_window=display_window,
                plane=plane,
                spacing_xyz=spacing_xyz,
                error=error,
            )
            plane_slices.append({"index": index, "png": _rel(png, root)})
        planes[plane] = {
            "center_index": center,
            "row_spacing_mm": row_spacing,
            "column_spacing_mm": column_spacing,
            "slices": plane_slices,
        }
    return {
        "shape": list(shape),
        "processed_spacing_xyz": spacing_xyz,
        "summary": _summary(volume),
        "display_window": display_window,
        "planes": planes,
    }


def _viewer_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DIF-Net V0b Prediction Viewer</title>
  <style>
    :root { --bg:#f6f7f4; --panel:#fff; --ink:#17201c; --muted:#66736c; --line:#d9dfd8; --accent:#0f766e; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Segoe UI, Arial, sans-serif; letter-spacing:0; background:var(--bg); color:var(--ink); }
    header { display:flex; justify-content:space-between; gap:16px; padding:14px 18px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:2; }
    h1 { margin:0; font-size:21px; }
    main { padding:12px; }
    .toolbar, .meta, .comparison { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .toolbar { display:grid; grid-template-columns:110px minmax(240px,1fr) 90px minmax(160px,.4fr) 70px minmax(150px,.35fr); gap:10px; align-items:center; margin-bottom:12px; }
    label, .muted { color:var(--muted); font-size:13px; }
    select, input[type=range] { width:100%; height:32px; }
    .comparison { display:grid; grid-template-columns:92px repeat(3,minmax(0,1fr)); gap:10px; align-items:stretch; }
    .planeHead { color:var(--muted); font-size:13px; font-weight:700; text-transform:uppercase; }
    .volumeLabel { display:grid; align-items:center; color:var(--muted); font-size:14px; font-weight:700; }
    .frame { background:#08100d; border-radius:6px; aspect-ratio:1/1; display:grid; place-items:center; overflow:hidden; }
    img { width:100%; height:100%; object-fit:contain; image-rendering:pixelated; filter:brightness(var(--brightness,1)) contrast(var(--contrast,1)); transform:scale(var(--zoom,1)); }
    .sliceControl { min-width:0; }
    .meta { margin-top:12px; display:grid; grid-template-columns:150px 1fr; gap:7px 12px; font-size:13px; }
    code { overflow-wrap:anywhere; }
    @media (max-width:980px) {
      header, .toolbar { display:block; }
      .toolbar > * { margin:6px 0; }
      .comparison { grid-template-columns:68px repeat(3,minmax(120px,1fr)); overflow-x:auto; }
    }
  </style>
</head>
<body>
  <header>
    <div><h1>DIF-Net V0b Prediction Viewer</h1><div class="muted" id="caseTitle"></div></div>
    <code id="runTitle"></code>
  </header>
  <main>
    <div class="toolbar">
      <label for="caseSelect">Case</label><select id="caseSelect"></select>
      <label for="brightness">Brightness</label><input id="brightness" type="range" min="50" max="200" value="100">
      <label for="contrast">Contrast</label><input id="contrast" type="range" min="50" max="220" value="100">
      <label for="zoom">Zoom</label><input id="zoom" type="range" min="100" max="300" value="100">
    </div>
    <div class="comparison">
      <div></div><div class="planeHead">Axial</div><div class="planeHead">Coronal</div><div class="planeHead">Sagittal</div>
      <div class="volumeLabel">generate</div>
      <div class="frame"><img id="generatedAxialImg"></div>
      <div class="frame"><img id="generatedCoronalImg"></div>
      <div class="frame"><img id="generatedSagittalImg"></div>
      <div class="volumeLabel">target</div>
      <div class="frame"><img id="targetAxialImg"></div>
      <div class="frame"><img id="targetCoronalImg"></div>
      <div class="frame"><img id="targetSagittalImg"></div>
      <div class="volumeLabel">slice</div>
      <div class="sliceControl"><input id="axialSlider" type="range" min="0"><div class="muted" id="axialMeta"></div></div>
      <div class="sliceControl"><input id="coronalSlider" type="range" min="0"><div class="muted" id="coronalMeta"></div></div>
      <div class="sliceControl"><input id="sagittalSlider" type="range" min="0"><div class="muted" id="sagittalMeta"></div></div>
    </div>
    <div class="meta" id="metrics"></div>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const DATA = window.VIEWER_DATA;
    const cases = DATA.cases;
    const caseSelect = document.getElementById("caseSelect");
    function text(v) { return typeof v === "object" ? JSON.stringify(v) : String(v ?? "-"); }
    function nearest(slices, center) {
      let idx = 0, dist = Infinity;
      slices.forEach((item, i) => { const d = Math.abs(item.index - center); if (d < dist) { idx = i; dist = d; } });
      return idx;
    }
    function currentCase() { return cases[Number(caseSelect.value || 0)]; }
    function renderPlane(plane) {
      const item = currentCase();
      const planeData = item.volumes.generated.planes[plane];
      const slider = document.getElementById(`${plane}Slider`);
      slider.max = Math.max(planeData.slices.length - 1, 0);
      const idx = Math.min(Number(slider.value || 0), planeData.slices.length - 1);
      slider.value = idx;
      const suffix = plane.charAt(0).toUpperCase() + plane.slice(1);
      const generatedSlice = item.volumes.generated.planes[plane].slices[idx];
      const targetSlice = item.volumes.target.planes[plane].slices[idx];
      document.getElementById(`generated${suffix}Img`).src = generatedSlice.png;
      document.getElementById(`target${suffix}Img`).src = targetSlice.png;
      document.getElementById(`${plane}Meta`).textContent = `${generatedSlice.index} (${idx + 1} / ${planeData.slices.length})`;
    }
    function render() {
      const item = currentCase();
      document.getElementById("caseTitle").textContent = `${item.case_key} view=${DATA.view_count} checkpoint=${DATA.checkpoint}`;
      document.getElementById("runTitle").textContent = DATA.prediction_manifest;
      for (const plane of ["axial","coronal","sagittal"]) {
        const volume = item.volumes.generated;
        const slider = document.getElementById(`${plane}Slider`);
        slider.value = nearest(volume.planes[plane].slices, volume.planes[plane].center_index);
        renderPlane(plane);
      }
      document.getElementById("metrics").innerHTML = Object.entries(item.metrics).map(([k,v]) => `<div>${k}</div><div><code>${text(v)}</code></div>`).join("");
    }
    function filters() {
      document.documentElement.style.setProperty("--brightness", Number(document.getElementById("brightness").value) / 100);
      document.documentElement.style.setProperty("--contrast", Number(document.getElementById("contrast").value) / 100);
      document.documentElement.style.setProperty("--zoom", Number(document.getElementById("zoom").value) / 100);
    }
    cases.forEach((item, i) => caseSelect.add(new Option(`${item.case_key} MAE=${item.metrics.mae_norm.toFixed(4)}`, String(i))));
    caseSelect.addEventListener("change", render);
    for (const plane of ["axial","coronal","sagittal"]) document.getElementById(`${plane}Slider`).addEventListener("input", () => renderPlane(plane));
    for (const id of ["brightness","contrast","zoom"]) document.getElementById(id).addEventListener("input", filters);
    filters(); render();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an interactive viewer for DIF-Net V0b prediction manifests.")
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--slice-stride", type=int, default=4)
    args = parser.parse_args()

    manifest = _read_json(args.prediction_manifest)
    out_dir = args.output_dir or args.prediction_manifest.parent / "interactive_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = manifest["predictions"][: args.max_cases] if args.max_cases is not None else manifest["predictions"]
    cases = []
    for prediction in predictions:
        case_key = _safe_name(str(prediction["case_key"]))
        target = np.load(prediction["target_cbct_path"]).astype(np.float32)
        generated = np.load(prediction["generated_cbct_path"]).astype(np.float32)
        case_dir = out_dir / "assets" / case_key
        spacing_xyz = [float(item) for item in prediction.get("processed_spacing_xyz") or [1.0, 1.0, 1.0]]
        cases.append(
            {
                "case_key": prediction["case_key"],
                "patient_id": prediction["patient_id"],
                "case_id": prediction["case_id"],
                "geometry_branch": prediction.get("geometry_branch"),
                "processed_shape_zyx": prediction.get("processed_shape_zyx"),
                "processed_spacing_xyz": spacing_xyz,
                "processed_origin_xyz": prediction.get("processed_origin_xyz"),
                "processed_direction": prediction.get("processed_direction"),
                "metrics": prediction["metrics"],
                "generated_cbct_path": prediction["generated_cbct_path"],
                "target_cbct_path": prediction["target_cbct_path"],
                "volumes": {
                    "generated": _export_volume(generated, "generated", case_dir, out_dir, args.slice_stride, spacing_xyz),
                    "target": _export_volume(target, "target", case_dir, out_dir, args.slice_stride, spacing_xyz),
                },
            }
        )
    viewer_data = {
        "schema": "x2cbct_difnet_v0b_prediction_viewer.v1",
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "checkpoint": Path(manifest["checkpoint_path"]).name,
        "geometry_branch": manifest.get("geometry_branch"),
        "axis_convention": manifest.get("axis_convention"),
        "display_policy": "spacing_corrected_by_processed_spacing_xyz",
        "view_count": manifest["view_count"],
        "view_key": manifest["view_key"],
        "normalization": manifest["normalization"],
        "case_count": len(cases),
        "slice_stride": args.slice_stride,
        "cases": cases,
    }
    _write_json(viewer_data, out_dir / "viewer_data.json")
    (out_dir / "viewer_data.js").write_text("window.VIEWER_DATA = " + json.dumps(viewer_data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    (out_dir / "index.html").write_text(_viewer_html(), encoding="utf-8")
    summary = {"ok": True, "viewer_kind": "prediction_interactive", "html": str(out_dir / "index.html"), "case_count": len(cases)}
    summary_path = _write_json(summary, out_dir / "viewer_summary.json")
    print(json.dumps({"ok": True, "summary": str(summary_path), "html": str(out_dir / "index.html"), "case_count": len(cases)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
