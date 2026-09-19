from __future__ import annotations

import argparse
import json
import socket
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "data_manifest" / "luna16_c2rv_split_registry.json"
PLANES = ("axial", "coronal", "sagittal")
FIXED_WINDOWS = {
    "lung": (-1000.0, 400.0),
    "soft": (-160.0, 240.0),
    "bone": (-500.0, 1500.0),
    "body": (-1024.0, 1000.0),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _profile_window(registry: dict[str, Any]) -> tuple[float, float]:
    profiles = registry.get("hu_window", {}).get("profiles", {})
    profile = profiles.get("luna16") if isinstance(profiles, dict) else None
    if isinstance(profile, dict):
        low = float(profile.get("hu_min", -1024.0))
        high = float(profile.get("hu_max", 1000.0))
        if high > low:
            return low, high
    return -1000.0, 1024.0


def build_catalog(registry_path: Path = DEFAULT_REGISTRY, *, include_missing: bool = False) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = _read_json(registry_path)
    cases: list[dict[str, Any]] = []
    for ordinal, item in enumerate(registry.get("cases", [])):
        ct_path = Path(str(item["ct_path"]))
        mask_path = Path(str(item.get("mask_path", ""))) if item.get("mask_path") else None
        if not include_missing and not ct_path.exists():
            continue
        shape_zyx = [int(value) for value in item.get("shape_zyx", [])]
        spacing_xyz = [float(value) for value in item.get("spacing_xyz", [])]
        planes = {
            "axial": int(shape_zyx[0]),
            "coronal": int(shape_zyx[1]),
            "sagittal": int(shape_zyx[2]),
        }
        cases.append(
            {
                "index": len(cases),
                "source_ordinal": ordinal,
                "patient_id": str(item.get("patient_id", item.get("seriesuid", ct_path.stem))),
                "seriesuid": str(item.get("seriesuid", ct_path.stem)),
                "subset": str(item.get("subset", "")),
                "split": str(item.get("split", "")),
                "shape_zyx": shape_zyx,
                "spacing_xyz": spacing_xyz,
                "planes": planes,
                "ct_path": str(ct_path),
                "mask_path": str(mask_path) if mask_path else "",
                "mask_exists": bool(mask_path and mask_path.exists()),
                "relative_ct_path": str(item.get("relative_ct_path", "")),
                "compatibility_reasons": list(item.get("compatibility_reasons", [])),
            }
        )
    if not cases:
        raise ValueError(f"no usable LUNA16 cases in registry: {registry_path}")
    low, high = _profile_window(registry)
    return {
        "schema": "x2cbct_luna16_batch_ct_viewer.v1",
        "registry_path": str(registry_path),
        "case_count": len(cases),
        "source_case_count": len(registry.get("cases", [])),
        "hu_window_profile": {"low": low, "high": high},
        "windows": {**FIXED_WINDOWS, "profile": (low, high)},
        "cases": cases,
    }


@lru_cache(maxsize=3)
def _load_ct(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    return volume, tuple(float(value) for value in image.GetSpacing())


@lru_cache(maxsize=3)
def _load_mask(path: str) -> np.ndarray:
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)


def _plane_count(shape_zyx: tuple[int, int, int], plane: str) -> int:
    return int(shape_zyx[{"axial": 0, "coronal": 1, "sagittal": 2}[plane]])


def _slice(array: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return array[index, :, :]
    if plane == "coronal":
        return array[:, index, :]
    if plane == "sagittal":
        return array[:, :, index]
    raise ValueError(f"unknown plane: {plane}")


def _plane_spacing(plane: str, spacing_xyz: tuple[float, float, float]) -> tuple[float, float]:
    sx, sy, sz = spacing_xyz
    if plane == "axial":
        return sy, sx
    if plane == "coronal":
        return sz, sx
    if plane == "sagittal":
        return sz, sy
    raise ValueError(f"unknown plane: {plane}")


@lru_cache(maxsize=3)
def _auto_window(path: str) -> tuple[float, float]:
    volume, _spacing = _load_ct(path)
    values = volume[np.isfinite(volume)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [0.5, 99.5])
    if float(high) <= float(low):
        high = float(low) + 1.0
    return float(low), float(high)


def _window_values(catalog: dict[str, Any], case: dict[str, Any], name: str) -> tuple[float, float]:
    if name in FIXED_WINDOWS:
        return FIXED_WINDOWS[name]
    if name == "profile":
        profile = catalog["hu_window_profile"]
        return float(profile["low"]), float(profile["high"])
    return _auto_window(str(Path(case["ct_path"])))


def _to_uint8(array: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=low)
    scaled = np.clip((arr - low) / max(high - low, 1e-8), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _resize_display(array: np.ndarray, plane: str, spacing_xyz: tuple[float, float, float], max_edge: int) -> np.ndarray:
    row_spacing, col_spacing = _plane_spacing(plane, spacing_xyz)
    base = max(min(row_spacing, col_spacing), 1e-8)
    target_h = max(1, int(round(array.shape[0] * row_spacing / base)))
    target_w = max(1, int(round(array.shape[1] * col_spacing / base)))
    resized = cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    scale = min(1.0, float(max_edge) / max(resized.shape[:2]))
    if scale < 1.0:
        new_w = max(1, int(round(resized.shape[1] * scale)))
        new_h = max(1, int(round(resized.shape[0] * scale)))
        resized = cv2.resize(resized, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized


def _overlay_mask(panel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = panel.copy()
    if contours:
        cv2.drawContours(out, contours, -1, (49, 222, 132), 2)
    return out


def render_slice_png(
    catalog: dict[str, Any],
    case_index: int,
    plane: str,
    slice_index: int,
    *,
    window: str = "lung",
    show_mask: bool = True,
    max_edge: int = 900,
) -> tuple[bytes, dict[str, Any]]:
    if plane not in PLANES:
        raise ValueError(f"unknown plane: {plane}")
    case = catalog["cases"][int(case_index)]
    volume, spacing_xyz = _load_ct(str(Path(case["ct_path"])))
    shape_zyx = tuple(int(value) for value in volume.shape)
    count = _plane_count(shape_zyx, plane)
    index = max(0, min(int(slice_index), count - 1))
    low, high = _window_values(catalog, case, window)
    gray = _to_uint8(_slice(volume, plane, index), low, high)
    gray = _resize_display(gray, plane, spacing_xyz, max_edge=max(128, int(max_edge)))
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if show_mask and case.get("mask_exists"):
        mask = _slice(_load_mask(str(Path(case["mask_path"]))), plane, index)
        mask = _resize_display(mask, plane, spacing_xyz, max_edge=max(128, int(max_edge)))
        panel = _overlay_mask(panel, mask)
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("failed to encode slice png")
    meta = {
        "case_index": int(case_index),
        "plane": plane,
        "slice_index": index,
        "slice_count": count,
        "window": {"name": window, "low": low, "high": high},
        "display_shape_hw": [int(panel.shape[0]), int(panel.shape[1])],
    }
    return encoded.tobytes(), meta


VIEWER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LUNA16 CT 批量巡检</title>
<style>
:root{--bg:#f4f6f5;--panel:#fff;--ink:#111815;--muted:#66736c;--line:#d8ded9;--accent:#047857;--canvas:#060a08}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;letter-spacing:0}
main{height:100vh;display:grid;grid-template-columns:320px minmax(420px,1fr) 310px;gap:0}
aside,section{min-width:0;background:var(--panel);border-right:1px solid var(--line)}aside{display:flex;flex-direction:column}
header{padding:12px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:18px;font-weight:650}h2{margin:0 0 8px;font-size:15px}
button,select,input,textarea{font:inherit}button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:7px 9px;cursor:pointer}button.active{background:var(--accent);border-color:var(--accent);color:#fff}
.tools{display:flex;gap:7px;flex-wrap:wrap}.search{width:100%;margin-top:10px;padding:8px;border:1px solid var(--line);border-radius:6px}
.list{overflow:auto}.case{padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}.case.active{background:#e7f4ef}.case small{display:block;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.viewer{display:grid;grid-template-rows:auto 1fr auto;height:100vh;background:#101614;color:#e8f1ed}.bar{display:flex;gap:8px;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid #22302a}
.canvas{display:grid;place-items:center;overflow:hidden;background:var(--canvas)}#slice{max-width:100%;max-height:100%;object-fit:contain;image-rendering:pixelated}
.slider{padding:10px 12px;border-top:1px solid #22302a}#sliceRange{width:100%}.meta{color:#aebcb5;font-size:13px}
.side{padding:12px;overflow:auto}.kv{display:grid;grid-template-columns:95px 1fr;gap:7px 10px;font-size:13px;margin-bottom:14px}.kv div:nth-child(odd){color:var(--muted)}code{overflow-wrap:anywhere}
textarea{width:100%;min-height:92px;resize:vertical;border:1px solid var(--line);border-radius:6px;padding:8px}.flags{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:8px 0 12px}
@media(max-width:980px){main{grid-template-columns:1fr;height:auto}.viewer{height:78vh}aside,.side{border-right:0;border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<main>
<aside>
<header><h1>LUNA16 CT 批量巡检</h1><input id="q" class="search" placeholder="搜索 patient / subset / split"></header>
<div class="list" id="caseList"></div>
</aside>
<section class="viewer">
<div class="bar">
<div><strong id="title"></strong><div class="meta" id="sliceMeta"></div></div>
<div class="tools">
<button id="prevCase">上一例</button><button id="nextCase">下一例</button>
<button data-plane="axial" class="plane active">Axial</button><button data-plane="coronal" class="plane">Coronal</button><button data-plane="sagittal" class="plane">Sagittal</button>
<select id="window"><option value="lung">lung</option><option value="profile">profile</option><option value="body">body</option><option value="soft">soft</option><option value="bone">bone</option><option value="auto">auto</option></select>
<button id="mask" class="active">Mask</button>
</div>
</div>
<div class="canvas" id="canvas"><img id="slice" alt=""></div>
<div class="slider"><input id="sliceRange" type="range" min="0" value="0"></div>
</section>
<section class="side">
<h2>样本信息</h2><div class="kv" id="info"></div>
<h2>人工标记</h2>
<div class="flags"><button data-flag="ok">完整肺部</button><button data-flag="partial_lung">疑似半肺</button><button data-flag="other_region">疑似部位异常</button><button data-flag="bad_fov">视野异常</button></div>
<textarea id="note" placeholder="备注"></textarea>
<div class="tools" style="margin-top:8px"><button id="saveNote">保存标记</button><button id="download">导出 JSON</button></div>
</section>
</main>
<script>
const S={cases:[],filtered:[],caseIndex:0,plane:"axial",slice:0,mask:true,window:"lung",reviews:JSON.parse(localStorage.getItem("luna16Review")||"{}")};
const $=id=>document.getElementById(id);
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function key(c){return c.patient_id}
function current(){return S.cases[S.caseIndex]}
function planeCount(){return current().planes[S.plane]}
function review(){return S.reviews[key(current())]||{}}
function setReview(data){S.reviews[key(current())]={...review(),...data,patient_id:key(current()),updated_at:new Date().toISOString()};localStorage.setItem("luna16Review",JSON.stringify(S.reviews));renderFlags()}
function renderList(){const q=$("q").value.toLowerCase();S.filtered=S.cases.filter(c=>(c.patient_id+" "+c.subset+" "+c.split).toLowerCase().includes(q));$("caseList").innerHTML=S.filtered.map(c=>`<div class="case ${c.index===S.caseIndex?"active":""}" data-i="${c.index}"><b>${c.index+1}. ${c.subset} ${c.split}</b><small>${c.patient_id}</small><small>${c.shape_zyx.join(" x ")} | ${c.spacing_xyz.map(x=>Number(x).toFixed(3)).join(", ")}</small></div>`).join("");document.querySelectorAll(".case").forEach(e=>e.onclick=()=>loadCase(Number(e.dataset.i)))}
function renderFlags(){const r=review();document.querySelectorAll("[data-flag]").forEach(b=>b.classList.toggle("active",r.flag===b.dataset.flag));$("note").value=r.note||""}
function renderInfo(){const c=current();$("title").textContent=`${c.index+1} / ${S.cases.length} | ${c.subset} | ${c.split}`;$("info").innerHTML=[["patient",c.patient_id],["shape zyx",c.shape_zyx.join(" x ")],["spacing xyz",c.spacing_xyz.map(x=>Number(x).toFixed(6)).join(", ")],["plane",S.plane],["ct",c.relative_ct_path||c.ct_path],["mask",c.mask_exists?"yes":"no"]].map(([a,b])=>`<div>${a}</div><div><code>${b}</code></div>`).join("");renderFlags()}
function renderSlice(){const c=current();const n=planeCount();S.slice=clamp(S.slice,0,n-1);$("sliceRange").max=n-1;$("sliceRange").value=S.slice;$("sliceMeta").textContent=`${S.plane} slice ${S.slice+1} / ${n}`;$("slice").src=`/api/slice?case=${S.caseIndex}&plane=${S.plane}&slice=${S.slice}&window=${S.window}&mask=${S.mask?1:0}&t=${Date.now()}`}
function loadCase(i){S.caseIndex=clamp(i,0,S.cases.length-1);S.slice=Math.floor(planeCount()/2);renderList();renderInfo();renderSlice()}
function changePlane(p){S.plane=p;document.querySelectorAll(".plane").forEach(b=>b.classList.toggle("active",b.dataset.plane===p));S.slice=Math.floor(planeCount()/2);renderInfo();renderSlice()}
fetch("/api/cases").then(r=>r.json()).then(d=>{S.cases=d.cases;S.filtered=S.cases;loadCase(0)});
$("q").oninput=renderList;$("sliceRange").oninput=e=>{S.slice=Number(e.target.value);renderSlice()};
$("prevCase").onclick=()=>loadCase(S.caseIndex-1);$("nextCase").onclick=()=>loadCase(S.caseIndex+1);
document.querySelectorAll(".plane").forEach(b=>b.onclick=()=>changePlane(b.dataset.plane));
$("window").onchange=e=>{S.window=e.target.value;renderSlice()};$("mask").onclick=()=>{S.mask=!S.mask;$("mask").classList.toggle("active",S.mask);renderSlice()};
$("canvas").addEventListener("wheel",e=>{e.preventDefault();if(e.shiftKey){loadCase(S.caseIndex+(e.deltaY>0?1:-1));return}S.slice+=e.deltaY>0?1:-1;renderSlice()},{passive:false});
window.addEventListener("keydown",e=>{if(e.key==="ArrowDown"){S.slice++;renderSlice()}if(e.key==="ArrowUp"){S.slice--;renderSlice()}if(e.key==="ArrowRight")loadCase(S.caseIndex+1);if(e.key==="ArrowLeft")loadCase(S.caseIndex-1)});
document.querySelectorAll("[data-flag]").forEach(b=>b.onclick=()=>setReview({flag:b.dataset.flag}));
$("saveNote").onclick=()=>setReview({note:$("note").value});
$("download").onclick=()=>{const blob=new Blob([JSON.stringify(S.reviews,null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="luna16_ct_review.json";a.click();URL.revokeObjectURL(a.href)};
</script>
</body>
</html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    catalog: dict[str, Any]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send(VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/cases":
                payload = {key: value for key, value in self.catalog.items() if key != "windows"}
                payload["cases"] = self.catalog["cases"]
                self._json(payload)
                return
            if parsed.path == "/api/slice":
                png, _meta = render_slice_png(
                    self.catalog,
                    int(qs.get("case", ["0"])[0]),
                    qs.get("plane", ["axial"])[0],
                    int(qs.get("slice", ["0"])[0]),
                    window=qs.get("window", ["lung"])[0],
                    show_mask=qs.get("mask", ["1"])[0] != "0",
                    max_edge=int(qs.get("max_edge", ["1100"])[0]),
                )
                self._send(png, "image/png")
                return
            self._json({"ok": False, "error": f"unknown path: {parsed.path}"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)


def _choose_port(host: str, requested: int) -> int:
    if requested != 0:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def serve(catalog: dict[str, Any], host: str, port: int) -> None:
    ViewerHandler.catalog = catalog
    server = ThreadingHTTPServer((host, port), ViewerHandler)
    url = f"http://{host}:{server.server_address[1]}"
    print(json.dumps({"ok": True, "url": url, "case_count": catalog["case_count"]}, ensure_ascii=False), flush=True)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a batch interactive CT slice viewer for LUNA16.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 to auto-pick a free port.")
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()
    catalog = build_catalog(args.registry, include_missing=args.include_missing)
    serve(catalog, args.host, _choose_port(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
