# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.setup_error_roi import ROI_FIELDS, load_cbct, read_roi_csv, roi_mask_from_candidate


SERVER_STATE: dict[str, object] = {}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_shape_zyx(path: str) -> list[int]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    size_xyz = list(reader.GetSize())
    return [int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0])]


@lru_cache(maxsize=8)
def cached_volume(path: str) -> tuple[np.ndarray, dict]:
    image, array = load_cbct(Path(path))
    meta = {
        "spacing_xyz": list(image.GetSpacing()),
        "size_xyz": list(image.GetSize()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "shape_zyx": list(array.shape),
        "p1": float(np.percentile(array, 1)),
        "p99": float(np.percentile(array, 99)),
    }
    return array, meta


def slice_array(array: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return array[index, :, :]
    if plane == "coronal":
        return array[:, index, :]
    if plane == "sagittal":
        return array[:, :, index]
    raise ValueError(f"unknown plane: {plane}")


def draw_roi(panel: np.ndarray, row: dict, plane: str, index: int, color: tuple[int, int, int]) -> None:
    x0, x1 = int(row["x_min"]), int(row["x_max"])
    y0, y1 = int(row["y_min"]), int(row["y_max"])
    z0, z1 = int(row["z_min"]), int(row["z_max"])
    if plane == "axial" and z0 <= index <= z1:
        cv2.rectangle(panel, (x0, y0), (x1, y1), color, 1)
    elif plane == "coronal" and y0 <= index <= y1:
        cv2.rectangle(panel, (x0, z0), (x1, z1), color, 1)
    elif plane == "sagittal" and x0 <= index <= x1:
        cv2.rectangle(panel, (y0, z0), (y1, z1), color, 1)


def render_png(row: dict, plane: str, index: int, params: dict[str, str]) -> bytes:
    array, meta = cached_volume(row["cbct_path"])
    working = dict(row)
    working.update(params)
    shape = array.shape
    max_index = {"axial": shape[0], "coronal": shape[1], "sagittal": shape[2]}[plane] - 1
    index = max(0, min(index, max_index))
    low = float(meta["p1"])
    high = float(meta["p99"])
    image_slice = slice_array(array, plane, index)
    gray = np.clip((image_slice - low) / max(high - low, 1e-6), 0, 1)
    panel = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask = roi_mask_from_candidate(array, working)
    mask_slice = slice_array(mask.astype(np.uint8), plane, index).astype(bool)
    overlay = panel.copy()
    overlay[mask_slice] = (45, 45, 255)
    panel = cv2.addWeighted(overlay, 0.35, panel, 0.65, 0)
    draw_roi(panel, working, plane, index, (0, 255, 255))
    cv2.putText(panel, f"{plane} {index}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("failed to encode png")
    return encoded.tobytes()


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>颈椎 ROI 修正</title>
<style>
:root{--bg:#f4f7f6;--panel:#fff;--line:#d7dfdc;--ink:#17211d;--muted:#69756f;--accent:#0f766e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}
header{padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}
main{display:grid;grid-template-columns:260px minmax(520px,1fr) 320px;height:calc(100vh - 54px)}
aside,.controls{background:var(--panel);border-right:1px solid var(--line);overflow:auto}.controls{border-right:0;border-left:1px solid var(--line);padding:12px}
.case{padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}.case.active{background:#dff3ef}.case small{color:var(--muted)}
.viewer{padding:10px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.frame{background:#070b0a;display:grid;place-items:center;min-height:260px;border-radius:6px;overflow:hidden}
img{max-width:100%;max-height:58vh;image-rendering:pixelated}.row{display:grid;grid-template-columns:92px 1fr 64px;gap:8px;align-items:center;margin:8px 0}
input[type=range]{width:100%}input,select,textarea,button{font:inherit}button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:7px 9px;cursor:pointer}button.primary{background:var(--accent);color:#fff}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.hint{font-size:13px;color:var(--muted);line-height:1.45}.status{display:grid;grid-template-columns:1fr;gap:6px}
textarea{width:100%;min-height:64px}.meta{color:var(--muted);font-size:13px;margin:6px 0}
@media(max-width:1100px){main{grid-template-columns:1fr;height:auto}.grid{grid-template-columns:1fr}aside,.controls{border:0}}
</style>
</head>
<body>
<header><strong>颈椎 ROI 修正</strong><span id="path"></span></header>
<main>
<aside><div id="cases"></div></aside>
<section class="viewer">
  <div class="grid">
    <div><h3>轴向</h3><div class="frame"><img id="axialImg"></div><input id="axial" type="range"></div>
    <div><h3>冠状</h3><div class="frame"><img id="coronalImg"></div><input id="coronal" type="range"></div>
    <div><h3>矢状</h3><div class="frame"><img id="sagittalImg"></div><input id="sagittal" type="range"></div>
  </div>
  <p class="hint">目标：保留后方、中线、连续颈椎骨；排除下颌、牙齿、肩部、床板和边缘。</p>
</section>
<section class="controls">
  <div class="status"><label>状态</label><select id="status"><option>accepted</option><option>needs_fix</option><option>exclude</option></select></div>
  <div id="sliders"></div>
  <label>备注</label><textarea id="note"></textarea>
  <div class="bar"><button id="auto">使用自动候选</button><button id="resetAll">清空本地修改</button><button id="tight">收紧到当前骨性掩膜</button><button id="shrink">向内缩 5 mm</button><button id="expand">扩大 5 mm</button><button id="front">排除前方 10 mm</button><button id="bottom">排除底部 10 mm</button></div>
  <div class="bar"><button id="prev">上一例</button><button id="next">下一例</button><button id="saveProject">保存到项目文件</button><button class="primary" id="download">下载 roi_reviewed.csv</button></div>
  <div class="meta" id="meta"></div>
</section>
</main>
<script>
const fields=["threshold_low","threshold_high","x_min","x_max","y_min","y_max","z_min","z_max"];
let S={cases:[],idx:0,edits:{},auto:{},storageKey:"cervicalRoiReview"};
function key(c){return `${c.patient_id}/${c.case_id}`}
function cur(){return S.cases[S.idx]}
function row(){const c=cur();return {...c,...(S.edits[key(c)]||{})}}
function save(){const c=cur();if(!c)return;const r=row();fields.forEach(f=>r[f]=document.getElementById(f).value);r.status=document.getElementById("status").value;r.note=document.getElementById("note").value;S.edits[key(c)]=r;localStorage.setItem("cervicalRoiReview",JSON.stringify(S.edits));renderCases()}
function setField(f,v){document.getElementById(f).value=v;document.getElementById(f+"Num").value=v}
function renderControls(){const r=row();document.getElementById("status").value=r.status||"needs_fix";document.getElementById("note").value=r.note||"";const shape=r.shape_zyx||[264,270,270];const max={threshold_low:2500,threshold_high:4000,x_min:shape[2]-1,x_max:shape[2]-1,y_min:shape[1]-1,y_max:shape[1]-1,z_min:shape[0]-1,z_max:shape[0]-1};document.getElementById("sliders").innerHTML=fields.map(f=>`<div class=row><label>${f}</label><input id="${f}" type=range min=0 max="${max[f]}" step="${f.startsWith('threshold')?10:1}" value="${r[f]}"><input id="${f}Num" type=number value="${r[f]}"></div>`).join("");fields.forEach(f=>{const a=document.getElementById(f),b=document.getElementById(f+"Num");a.oninput=()=>{b.value=a.value;refresh()};b.oninput=()=>{a.value=b.value;refresh()}});document.getElementById("meta").textContent=`${r.patient_id}/${r.case_id} | ${r.cbct_path}`}
function renderCases(){document.getElementById("cases").innerHTML=S.cases.map((c,i)=>{const r={...c,...(S.edits[key(c)]||{})};return `<div class="case ${i===S.idx?'active':''}" onclick="go(${i})"><b>${c.patient_id}/${c.case_id}</b><br><small>${r.status||""} ${r.note||""}</small></div>`}).join("")}
function renderSlices(){const r=row(),shape=r.shape_zyx||[264,270,270];[["axial",0],["coronal",1],["sagittal",2]].forEach(([p,axis])=>{const s=document.getElementById(p);s.max=shape[axis]-1;if(!s.value)s.value=Math.floor(shape[axis]/2);document.getElementById(p+"Img").src=url(p,Number(s.value),r)})}
function url(p,i,r){const q=new URLSearchParams({case:S.idx,plane:p,index:i});fields.forEach(f=>q.set(f,r[f]));return `/api/slice?${q.toString()}&t=${Date.now()}`}
function refresh(){save();renderSlices()}
function go(i){save();S.idx=Math.max(0,Math.min(i,S.cases.length-1));renderCases();renderControls();renderSlices()}
function reviewedRows(){return S.cases.map(c=>({...c,...(S.edits[key(c)]||{})}))}
function csv(){const rows=reviewedRows();const cols=Object.keys(rows[0]);return [cols.join(","),...rows.map(r=>cols.map(c=>`"${String(r[c]??"").replaceAll('"','""')}"`).join(","))].join("\n")}
async function init(){const meta=await (await fetch("/api/meta")).json();S.storageKey=`cervicalRoiReview:${meta.dataset_key}`;S.edits=JSON.parse(localStorage.getItem(S.storageKey)||"{}");const res=await fetch("/api/cases");S.cases=await res.json();S.cases.forEach(c=>S.auto[key(c)]={...c});document.getElementById("path").textContent=`${S.cases.length} cases | ${meta.dataset_key}`;renderCases();renderControls();["axial","coronal","sagittal"].forEach(p=>document.getElementById(p).oninput=renderSlices);["status","note"].forEach(id=>document.getElementById(id).oninput=save);renderSlices()}
document.getElementById("prev").onclick=()=>go(S.idx-1);document.getElementById("next").onclick=()=>go(S.idx+1);document.getElementById("auto").onclick=()=>{S.edits[key(cur())]={...S.auto[key(cur())]};go(S.idx)};document.getElementById("resetAll").onclick=()=>{if(confirm("清空本地修改并重新使用当前候选文件？")){S.edits={};localStorage.removeItem(S.storageKey);go(S.idx)}};document.getElementById("shrink").onclick=()=>{["x_min","y_min","z_min"].forEach(f=>setField(f,Number(document.getElementById(f).value)+5));["x_max","y_max","z_max"].forEach(f=>setField(f,Number(document.getElementById(f).value)-5));refresh()};document.getElementById("expand").onclick=()=>{["x_min","y_min","z_min"].forEach(f=>setField(f,Math.max(0,Number(document.getElementById(f).value)-5)));["x_max","y_max","z_max"].forEach(f=>setField(f,Number(document.getElementById(f).value)+5));refresh()};document.getElementById("front").onclick=()=>{setField("y_min",Number(document.getElementById("y_min").value)+10);refresh()};document.getElementById("bottom").onclick=()=>{setField("z_max",Number(document.getElementById("z_max").value)-10);refresh()};document.getElementById("tight").onclick=()=>alert("第一版先使用自动候选或手动滑块；收紧按钮将在下一轮按当前阈值重算。");document.getElementById("saveProject").onclick=async()=>{save();const res=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({rows:reviewedRows()})});const data=await res.json();alert(data.ok?`已保存：${data.path}`:`保存失败：${data.error}`)};document.getElementById("download").onclick=()=>{save();const blob=new Blob([csv()],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="roi_reviewed.csv";a.click();URL.revokeObjectURL(a.href)};init();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/meta":
            self.send(json.dumps({"dataset_key": SERVER_STATE["dataset_key"]}, ensure_ascii=False).encode("utf-8"), "application/json")
        elif parsed.path == "/api/cases":
            self.send(json.dumps(SERVER_STATE["rows"], ensure_ascii=False).encode("utf-8"), "application/json")
        elif parsed.path == "/api/slice":
            qs = parse_qs(parsed.query)
            rows = SERVER_STATE["rows"]  # type: ignore[assignment]
            row = dict(rows[int(qs.get("case", ["0"])[0])])
            params = {k: qs[k][0] for k in qs if k not in {"case", "plane", "index", "t"}}
            png = render_png(row, qs.get("plane", ["axial"])[0], int(qs.get("index", ["0"])[0]), params)
            self.send(png, "image/png")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            rows = payload.get("rows", [])
            output_path = Path(str(SERVER_STATE["review_output"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(rows[0].keys()) if rows else ROI_FIELDS
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.send(json.dumps({"ok": True, "path": str(output_path)}, ensure_ascii=False).encode("utf-8"), "application/json")
        except Exception as exc:
            self.send(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, format: str, *args: object) -> None:
        return

    def send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve interactive cervical ROI reviewer.")
    parser.add_argument("--roi-candidates", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    rows = read_rows(args.roi_candidates)
    for row in rows:
        row["shape_zyx"] = read_shape_zyx(row["cbct_path"])
    SERVER_STATE["rows"] = rows
    SERVER_STATE["dataset_key"] = hashlib.sha1(args.roi_candidates.read_bytes()).hexdigest()[:12]
    SERVER_STATE["review_output"] = args.review_output or (args.roi_candidates.parent / "roi_reviewed.csv")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ok": True, "url": f"http://{args.host}:{args.port}", "case_count": len(rows), "review_output": str(SERVER_STATE["review_output"])}, ensure_ascii=False))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
