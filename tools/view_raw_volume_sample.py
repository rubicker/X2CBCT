from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "volume"


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape_zyx": list(arr.shape),
        "dtype": str(array.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "p0_5": float(np.nanpercentile(arr, 0.5)),
        "p99_5": float(np.nanpercentile(arr, 99.5)),
    }


def _display_window(
    array: np.ndarray,
    *,
    low: float | None,
    high: float | None,
    percentile_low: float,
    percentile_high: float,
) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return {"mode": "empty", "low": 0.0, "high": 1.0}
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    hu_like = low is None and high is None and value_min <= -900.0 and value_max >= 100.0
    if hu_like:
        window_low = max(value_min, -1024.0)
        window_high = min(value_max, 1000.0)
    else:
        window_low = float(low) if low is not None else float(np.percentile(values, percentile_low))
        window_high = float(high) if high is not None else float(np.percentile(values, percentile_high))
    if window_high <= window_low + 1e-8:
        window_low = value_min
        window_high = value_max
    if window_high <= window_low + 1e-8:
        window_high = window_low + 1.0
    if low is not None or high is not None:
        mode = "manual"
    elif hu_like:
        mode = "auto_hu_like"
    else:
        mode = "auto_percentile"
    return {
        "mode": mode,
        "low": window_low,
        "high": window_high,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
    }


def _to_uint8(array: np.ndarray, window: dict[str, Any]) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=float(window["low"]))
    low = float(window["low"])
    high = float(window["high"])
    gray = np.clip((arr - low) / max(high - low, 1e-8), 0.0, 1.0)
    return (gray * 255.0).astype(np.uint8)


def _plane_spacing(plane: str, spacing_xyz: tuple[float, float, float]) -> dict[str, Any]:
    spacing_x, spacing_y, spacing_z = [float(item) for item in spacing_xyz]
    if plane == "axial":
        return {"row_axis": "y", "column_axis": "x", "row_spacing_mm": spacing_y, "column_spacing_mm": spacing_x}
    if plane == "coronal":
        return {"row_axis": "z", "column_axis": "x", "row_spacing_mm": spacing_z, "column_spacing_mm": spacing_x}
    if plane == "sagittal":
        return {"row_axis": "z", "column_axis": "y", "row_spacing_mm": spacing_z, "column_spacing_mm": spacing_y}
    raise ValueError(f"unknown plane: {plane}")


def _spacing_corrected_uint8(array: np.ndarray, window: dict[str, Any], plane_meta: dict[str, Any]) -> np.ndarray:
    gray = _to_uint8(array, window)
    row_spacing = float(plane_meta["row_spacing_mm"])
    column_spacing = float(plane_meta["column_spacing_mm"])
    base_spacing = min(row_spacing, column_spacing)
    target_height = max(1, int(round(gray.shape[0] * row_spacing / base_spacing)))
    target_width = max(1, int(round(gray.shape[1] * column_spacing / base_spacing)))
    if (target_height, target_width) == gray.shape:
        return gray
    return cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def _slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"unknown plane: {plane}")


def _plane_size(shape_zyx: tuple[int, int, int], plane: str) -> int:
    return int(shape_zyx[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]])


def _indices(shape_zyx: tuple[int, int, int], plane: str, stride: int) -> list[int]:
    size = _plane_size(shape_zyx, plane)
    indices = list(range(0, size, max(1, stride)))
    center = size // 2
    if center not in indices:
        indices.append(center)
        indices.sort()
    return indices


def _write_png(path: Path, array: np.ndarray, window: dict[str, Any], plane_meta: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = cv2.cvtColor(_spacing_corrected_uint8(array, window, plane_meta), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), panel)
    return path


def _export_plane(
    volume: np.ndarray,
    plane: str,
    out_dir: Path,
    window: dict[str, Any],
    stride: int,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    shape_zyx = tuple(int(item) for item in volume.shape)
    center = _plane_size(shape_zyx, plane) // 2
    plane_meta = _plane_spacing(plane, spacing_xyz)
    slices = []
    for index in _indices(shape_zyx, plane, stride):
        png = out_dir / "assets" / plane / f"{plane}_{index:04d}.png"
        _write_png(png, _slice(volume, plane, index), window, plane_meta)
        slices.append({"index": index, "png": _rel(png, out_dir)})
    return {
        "count": _plane_size(shape_zyx, plane),
        "center_index": center,
        "spacing_correction": "png_resampled_to_physical_mm",
        **plane_meta,
        "slices": slices,
    }


def _viewer_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw Volume Interactive Viewer</title>
  <style>
    :root { --bg:#f5f7f6; --panel:#fff; --ink:#17201c; --muted:#65716b; --line:#d7ded9; --accent:#0f766e; --canvas:#090f0d; --brightness:1; --contrast:1; --zoom:1; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Segoe UI,Arial,sans-serif; letter-spacing:0; }
    header { display:flex; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:2; }
    h1 { margin:0 0 4px; font-size:21px; font-weight:650; }
    h2 { margin:0 0 8px; font-size:16px; font-weight:650; }
    main { padding:12px; display:grid; grid-template-columns:minmax(360px,1fr) minmax(420px,.8fr); gap:12px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }
    .grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
    .frame { background:var(--canvas); border-radius:6px; aspect-ratio:1/1; display:grid; place-items:center; overflow:hidden; }
    img { width:100%; height:100%; object-fit:contain; image-rendering:pixelated; filter:brightness(var(--brightness)) contrast(var(--contrast)); transform:scale(var(--zoom)); transform-origin:center; }
    label,.muted { color:var(--muted); font-size:13px; }
    input[type=range] { width:100%; }
    .controls { display:grid; grid-template-columns:120px 1fr; gap:8px 10px; align-items:center; margin:10px 0 2px; }
    .kv { display:grid; grid-template-columns:150px 1fr; gap:7px 12px; font-size:13px; }
    .kv div:nth-child(odd) { color:var(--muted); }
    code { overflow-wrap:anywhere; }
    @media (max-width:1100px) { main,.grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>Raw Volume Interactive Viewer</h1><div class="muted" id="caseTitle"></div></div>
    <code id="sourcePath"></code>
  </header>
  <main>
    <section>
      <div class="controls">
        <label for="brightness">Brightness</label><input id="brightness" type="range" min="50" max="200" value="100">
        <label for="contrast">Contrast</label><input id="contrast" type="range" min="50" max="220" value="100">
        <label for="zoom">Zoom</label><input id="zoom" type="range" min="100" max="300" value="100">
      </div>
      <div class="grid">
        <div><h2>Axial</h2><div class="frame"><img id="axialImg"></div><input id="axialSlider" type="range" min="0"><div class="muted" id="axialMeta"></div></div>
        <div><h2>Coronal</h2><div class="frame"><img id="coronalImg"></div><input id="coronalSlider" type="range" min="0"><div class="muted" id="coronalMeta"></div></div>
        <div><h2>Sagittal</h2><div class="frame"><img id="sagittalImg"></div><input id="sagittalSlider" type="range" min="0"><div class="muted" id="sagittalMeta"></div></div>
      </div>
    </section>
    <section><h2>Metadata</h2><div class="kv" id="metadata"></div></section>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const DATA = window.VIEWER_DATA;
    const planes = ["axial", "coronal", "sagittal"];
    function text(value) {
      if (Array.isArray(value)) return value.join(" x ");
      if (value && typeof value === "object") return JSON.stringify(value);
      return String(value ?? "-");
    }
    function nearest(items, center) {
      let best = 0, dist = Infinity;
      items.forEach((item, i) => { const d = Math.abs(item.index - center); if (d < dist) { best = i; dist = d; } });
      return best;
    }
    function renderPlane(plane) {
      const planeData = DATA.planes[plane];
      const slider = document.getElementById(`${plane}Slider`);
      slider.max = Math.max(planeData.slices.length - 1, 0);
      const pos = Math.min(Number(slider.value || 0), planeData.slices.length - 1);
      slider.value = pos;
      const item = planeData.slices[pos];
      document.getElementById(`${plane}Img`).src = item.png;
      document.getElementById(`${plane}Meta`).textContent = `${plane} slice ${item.index} (${pos + 1} / ${planeData.slices.length}; source count ${planeData.count})`;
    }
    function filters() {
      document.documentElement.style.setProperty("--brightness", Number(document.getElementById("brightness").value) / 100);
      document.documentElement.style.setProperty("--contrast", Number(document.getElementById("contrast").value) / 100);
      document.documentElement.style.setProperty("--zoom", Number(document.getElementById("zoom").value) / 100);
    }
    document.getElementById("caseTitle").textContent = `${DATA.case_id} | ${DATA.summary.shape_zyx.join(" x ")} zyx`;
    document.getElementById("sourcePath").textContent = DATA.volume_path;
    document.getElementById("metadata").innerHTML = [
      ["case id", DATA.case_id], ["volume path", DATA.volume_path], ["shape zyx", DATA.summary.shape_zyx],
      ["spacing xyz", DATA.image.spacing_xyz], ["origin xyz", DATA.image.origin_xyz], ["direction", DATA.image.direction],
      ["dtype", DATA.summary.dtype], ["min", DATA.summary.min], ["max", DATA.summary.max], ["mean", DATA.summary.mean],
      ["display window", DATA.display_window], ["slice stride", DATA.slice_stride]
    ].map(([k,v]) => `<div>${k}</div><div><code>${text(v)}</code></div>`).join("");
    planes.forEach((plane) => {
      const slider = document.getElementById(`${plane}Slider`);
      slider.value = nearest(DATA.planes[plane].slices, DATA.planes[plane].center_index);
      slider.addEventListener("input", () => renderPlane(plane));
      renderPlane(plane);
    });
    ["brightness","contrast","zoom"].forEach((id) => document.getElementById(id).addEventListener("input", filters));
    filters();
  </script>
</body>
</html>
"""


def export_raw_volume_viewer(
    volume_path: Path,
    output_dir: Path,
    *,
    slice_stride: int = 1,
    window_low: float | None = None,
    window_high: float | None = None,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> dict[str, Any]:
    image = sitk.ReadImage(str(volume_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {volume.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spacing_xyz = tuple(float(item) for item in image.GetSpacing())
    window = _display_window(
        volume,
        low=window_low,
        high=window_high,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    viewer_data = {
        "schema": "x2cbct_raw_volume_viewer.v1",
        "volume_path": str(volume_path.resolve()),
        "case_id": volume_path.parent.name,
        "slice_stride": max(1, int(slice_stride)),
        "summary": _summary(volume),
        "display_window": window,
        "image": {
            "spacing_xyz": [float(item) for item in spacing_xyz],
            "origin_xyz": [float(item) for item in image.GetOrigin()],
            "direction": [float(item) for item in image.GetDirection()],
            "size_xyz": [int(item) for item in image.GetSize()],
        },
        "planes": {
            plane: _export_plane(volume, plane, output_dir, window, slice_stride, spacing_xyz)
            for plane in ("axial", "coronal", "sagittal")
        },
    }
    (output_dir / "viewer_data.json").write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "viewer_data.js").write_text("window.VIEWER_DATA = " + json.dumps(viewer_data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    (output_dir / "index.html").write_text(_viewer_html(), encoding="utf-8")
    summary = {
        "ok": True,
        "viewer_kind": "raw_volume_interactive",
        "html": str(output_dir / "index.html"),
        "viewer_data": str(output_dir / "viewer_data.json"),
        "volume_path": str(volume_path.resolve()),
        "shape_zyx": list(volume.shape),
        "display_window": window,
    }
    (output_dir / "viewer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an interactive viewer for a raw 3D medical volume.")
    parser.add_argument("--volume-path", type=Path, required=True, help="Raw .mha/.mhd/.nii/.nii.gz volume path.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--slice-stride", type=int, default=1, help="Export every Nth slice per plane; center slice is always included.")
    parser.add_argument("--window-low", type=float)
    parser.add_argument("--window-high", type=float)
    parser.add_argument("--percentile-low", type=float, default=0.5)
    parser.add_argument("--percentile-high", type=float, default=99.5)
    args = parser.parse_args()

    volume_path = args.volume_path.resolve()
    if not volume_path.exists():
        raise FileNotFoundError(volume_path)
    out_dir = args.output_dir or PROJECT_ROOT / "runs" / "viewers" / "raw_volume" / _safe_name(f"{volume_path.parent.name}_{volume_path.name}")
    summary = export_raw_volume_viewer(
        volume_path,
        out_dir,
        slice_stride=args.slice_stride,
        window_low=args.window_low,
        window_high=args.window_high,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    print(json.dumps({"ok": True, "summary": str(out_dir / "viewer_summary.json"), "html": summary["html"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
