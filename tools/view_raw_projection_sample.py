from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.translation_probe import (
    geometry_gantry_angles,
    parse_projection_ordinal,
    percentile_window,
    to_uint8_window,
)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _projection_sort_key(path: Path) -> tuple[int, str]:
    ordinal = parse_projection_ordinal(path)
    return (ordinal if ordinal is not None else 10**9, path.name)


def _projection_paths(case_path: Path) -> list[Path]:
    excluded = {"CBCT.nii.gz", "CT.nii.gz"}
    return sorted(
        [item for item in case_path.glob("*.nii.gz") if item.name not in excluded],
        key=_projection_sort_key,
    )


def _read_projection(path: Path) -> np.ndarray:
    array = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32))
    if array.ndim != 2:
        raise ValueError(f"projection must be 2D after squeeze, got {array.shape}: {path}")
    return array


def _array_summary(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    return {
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _write_png(path: Path, array: np.ndarray, window: dict[str, float], label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = to_uint8_window(array, window)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, label[:96], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), panel)
    return path


def _contact_sheet(items: list[dict[str, Any]], output: Path, *, thumb_size: int = 192, columns: int = 6) -> Path:
    thumbs: list[np.ndarray] = []
    for item in items:
        img = cv2.imread(str(output.parent / item["png"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        resized = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        thumbs.append(resized)
    if not thumbs:
        raise ValueError("no projection PNGs for contact sheet")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(thumbs) / columns))
    canvas = np.zeros((rows * thumb_size, columns * thumb_size, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        canvas[row * thumb_size : (row + 1) * thumb_size, col * thumb_size : (col + 1) * thumb_size] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


def _viewer_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw Projection Viewer</title>
  <style>
    :root{--bg:#f5f7f6;--panel:#fff;--ink:#17201c;--muted:#65716b;--line:#d7ded9;--canvas:#090f0d;--brightness:1;--contrast:1;--zoom:1}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}
    header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
    h1{margin:0 0 4px;font-size:21px;font-weight:650}
    h2{margin:0 0 10px;font-size:16px;font-weight:650}
    main{display:grid;grid-template-columns:minmax(420px,1fr) minmax(360px,.55fr);gap:12px;padding:12px}
    section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
    .frame{background:var(--canvas);border-radius:6px;min-height:420px;display:grid;place-items:center;overflow:hidden}
    img{max-width:100%;max-height:78vh;object-fit:contain;image-rendering:pixelated;filter:brightness(var(--brightness)) contrast(var(--contrast));transform:scale(var(--zoom));transform-origin:center}
    input[type=range]{width:100%}
    label,.muted{color:var(--muted);font-size:13px}
    code{overflow-wrap:anywhere}
    .controls{display:grid;grid-template-columns:120px 1fr;gap:8px 10px;align-items:center;margin:10px 0}
    .kv{display:grid;grid-template-columns:145px 1fr;gap:7px 12px;font-size:13px}
    .kv div:nth-child(odd){color:var(--muted)}
    .strip{display:flex;gap:6px;overflow-x:auto;padding:8px 0}
    .strip button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:3px;min-width:58px;height:58px;cursor:pointer}
    .strip button.active{border-color:#0f766e;outline:2px solid #99f6e4}
    .strip img{width:50px;height:50px;object-fit:cover;transform:none;filter:none}
    @media(max-width:980px){main{grid-template-columns:1fr}.frame{min-height:320px}}
  </style>
</head>
<body>
  <header>
    <div><h1>Raw Projection Viewer</h1><div class="muted" id="title"></div></div>
    <code id="source"></code>
  </header>
  <main>
    <section>
      <div class="frame"><img id="projectionImage" alt="projection"></div>
      <div class="controls">
        <label for="projectionSlider">Projection</label><input id="projectionSlider" type="range" min="0" value="0">
        <label for="brightness">Brightness</label><input id="brightness" type="range" min="50" max="200" value="100">
        <label for="contrast">Contrast</label><input id="contrast" type="range" min="50" max="220" value="100">
        <label for="zoom">Zoom</label><input id="zoom" type="range" min="100" max="300" value="100">
      </div>
      <div class="muted" id="projectionLine"></div>
      <div class="strip" id="strip"></div>
    </section>
    <section>
      <h2>Metadata</h2>
      <div class="kv" id="metadata"></div>
    </section>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const DATA = window.VIEWER_DATA;
    const slider = document.getElementById("projectionSlider");
    const img = document.getElementById("projectionImage");
    function text(value){
      if(Array.isArray(value)) return value.join(" x ");
      if(value && typeof value === "object") return JSON.stringify(value);
      return String(value ?? "-");
    }
    function renderKv(rows){
      document.getElementById("metadata").innerHTML = rows.map(([k,v]) => `<div>${k}</div><div><code>${text(v)}</code></div>`).join("");
    }
    function setProjection(index){
      const items = DATA.projections;
      const idx = Math.max(0, Math.min(Number(index), items.length - 1));
      slider.value = idx;
      const item = items[idx];
      img.src = item.png;
      document.getElementById("projectionLine").textContent = `${idx + 1} / ${items.length} | ${item.file} | ordinal ${item.ordinal ?? "-"} | gantry ${item.gantry_angle_deg ?? "-"} deg`;
      document.querySelectorAll(".strip button").forEach((btn, i) => btn.classList.toggle("active", i === idx));
      renderKv([
        ["case path", DATA.case_path], ["file", item.file], ["ordinal", item.ordinal],
        ["gantry angle deg", item.gantry_angle_deg], ["shape yx", item.shape_yx],
        ["spacing xy", item.spacing_xy], ["summary", item.summary], ["display window", DATA.display_window],
        ["contact sheet", DATA.contact_sheet]
      ]);
    }
    function filters(){
      document.documentElement.style.setProperty("--brightness", Number(document.getElementById("brightness").value) / 100);
      document.documentElement.style.setProperty("--contrast", Number(document.getElementById("contrast").value) / 100);
      document.documentElement.style.setProperty("--zoom", Number(document.getElementById("zoom").value) / 100);
    }
    document.getElementById("title").textContent = `${DATA.case_id} | ${DATA.projection_count} projections`;
    document.getElementById("source").textContent = DATA.case_path;
    slider.max = Math.max(DATA.projections.length - 1, 0);
    slider.addEventListener("input", () => setProjection(slider.value));
    ["brightness","contrast","zoom"].forEach((id) => document.getElementById(id).addEventListener("input", filters));
    document.getElementById("strip").innerHTML = DATA.projections.map((item, idx) => `<button data-idx="${idx}" title="${item.file}"><img src="${item.png}" alt="${item.file}"></button>`).join("");
    document.querySelectorAll(".strip button").forEach((btn) => btn.addEventListener("click", () => setProjection(btn.dataset.idx)));
    filters();
    setProjection(0);
  </script>
</body>
</html>
"""


def export_raw_projection_viewer(
    case_path: Path,
    output_dir: Path,
    *,
    max_projections: int | None = None,
    window_mode: str = "global",
) -> dict[str, Any]:
    paths = _projection_paths(case_path)
    if max_projections is not None:
        paths = paths[: max(0, int(max_projections))]
    if not paths:
        raise ValueError(f"no raw projection files under {case_path}")

    arrays = [(path, _read_projection(path)) for path in paths]
    if window_mode == "global":
        window = percentile_window(np.concatenate([array.ravel() for _, array in arrays]))
    elif window_mode == "first":
        window = percentile_window(arrays[0][1])
    else:
        raise ValueError(f"unsupported window mode: {window_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = case_path / "Geometry"
    gantry_angles = geometry_gantry_angles(geometry_path) if geometry_path.exists() else []

    projections: list[dict[str, Any]] = []
    for index, (path, array) in enumerate(arrays):
        ordinal = parse_projection_ordinal(path)
        gantry = None
        if ordinal is not None and ordinal < len(gantry_angles):
            gantry = float(gantry_angles[ordinal])
        image = sitk.ReadImage(str(path))
        png = _write_png(
            output_dir / "assets" / "projections" / f"projection_{index:04d}.png",
            array,
            window,
            f"{path.name} gantry={gantry:.2f}" if gantry is not None else path.name,
        )
        projections.append(
            {
                "index": index,
                "file": path.name,
                "path": str(path.resolve()),
                "ordinal": ordinal,
                "gantry_angle_deg": gantry,
                "shape_yx": [int(array.shape[0]), int(array.shape[1])],
                "spacing_xy": [float(item) for item in image.GetSpacing()[:2]],
                "summary": _array_summary(array),
                "png": _rel(png, output_dir),
            }
        )

    contact_sheet = _contact_sheet(projections, output_dir / "contact_sheet.png")
    viewer_data = {
        "schema": "x2cbct_raw_projection_viewer.v1",
        "case_path": str(case_path.resolve()),
        "case_id": f"patient{case_path.parent.name}_case{case_path.name}",
        "projection_count": len(projections),
        "display_window": window,
        "window_mode": window_mode,
        "contact_sheet": _rel(contact_sheet, output_dir),
        "projections": projections,
    }
    (output_dir / "viewer_data.json").write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "viewer_data.js").write_text("window.VIEWER_DATA = " + json.dumps(viewer_data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    (output_dir / "index.html").write_text(_viewer_html(), encoding="utf-8")
    summary = {
        "ok": True,
        "viewer_kind": "raw_projection_interactive",
        "html": str(output_dir / "index.html"),
        "viewer_data": str(output_dir / "viewer_data.json"),
        "contact_sheet": str(contact_sheet),
        "case_path": str(case_path.resolve()),
        "projection_count": len(projections),
        "display_window": window,
    }
    (output_dir / "viewer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an interactive viewer for raw self_projection1208 projection images.")
    parser.add_argument("--case-path", type=Path, required=True, help="Raw self_projection1208 patient/case directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-projections", type=int)
    parser.add_argument("--window-mode", choices=("global", "first"), default="global")
    args = parser.parse_args()

    case_path = args.case_path.resolve()
    if not case_path.exists():
        raise FileNotFoundError(case_path)
    summary = export_raw_projection_viewer(
        case_path,
        args.output_dir.resolve(),
        max_projections=args.max_projections,
        window_mode=args.window_mode,
    )
    print(json.dumps({"ok": True, "summary": str(Path(summary["viewer_data"]).with_name("viewer_summary.json")), "html": summary["html"], "projection_count": summary["projection_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
