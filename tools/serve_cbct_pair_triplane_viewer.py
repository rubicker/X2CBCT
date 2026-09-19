# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
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

from x2cbct_difnet.setup_error_roi import load_cbct

SERVER_STATE: dict[str, object] = {}


def cbct_path(dataset_root: Path, patient_id: str, case_id: str) -> Path:
    return dataset_root / str(patient_id) / str(case_id) / "CBCT.nii.gz"


def scan_dataset(dataset_root: Path) -> dict[str, list[str]]:
    patients: dict[str, list[str]] = {}
    for patient_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        cases = []
        for case_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if (case_dir / "CBCT.nii.gz").exists():
                cases.append(case_dir.name)
        if cases:
            patients[patient_dir.name] = cases
    return patients


def read_shape_zyx(path: Path) -> list[int]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    sx, sy, sz = reader.GetSize()
    return [int(sz), int(sy), int(sx)]


def read_metrics(path: Path | None) -> dict[tuple[str, str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(str(row["patient_id"]), str(row["reference_case_id"]), str(row["moving_case_id"]))] = row
    return rows


@lru_cache(maxsize=6)
def cached_volume(path: str) -> tuple[np.ndarray, dict[str, object]]:
    image, array = load_cbct(Path(path))
    meta = {
        "shape_zyx": list(array.shape),
        "spacing_xyz": [float(v) for v in image.GetSpacing()],
        "p1": float(np.percentile(array, 1)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
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


def render_png(path: Path, plane: str, index: int, low: float | None, high: float | None) -> bytes:
    array, meta = cached_volume(str(path))
    shape = array.shape
    max_index = {"axial": shape[0], "coronal": shape[1], "sagittal": shape[2]}[plane] - 1
    index = max(0, min(int(index), max_index))
    low = float(meta["p1"] if low is None else low)
    high = float(meta["p99"] if high is None else high)
    slc = slice_array(array, plane, index)
    gray = np.clip((slc - low) / max(high - low, 1e-6), 0, 1)
    panel = (gray * 255).astype(np.uint8)
    panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, f"{plane} {index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1)
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return encoded.tobytes()


def metric_for(patient_id: str, case_a: str, case_b: str) -> dict[str, str] | None:
    metrics: dict[tuple[str, str, str], dict[str, str]] = SERVER_STATE["metrics"]  # type: ignore[assignment]
    return metrics.get((patient_id, case_a, case_b)) or metrics.get((patient_id, case_b, case_a))


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBCT Pair Triplane Viewer</title>
<style>
:root{--bg:#eef2f3;--panel:#fff;--line:#d9e1e3;--ink:#142022;--muted:#667276;--accent:#0f766e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:10px 14px;display:flex;justify-content:space-between;gap:12px;align-items:center}
main{display:grid;grid-template-columns:300px minmax(780px,1fr);height:calc(100vh - 52px)}
aside{background:var(--panel);border-right:1px solid var(--line);padding:12px;overflow:auto}.viewer{padding:12px;overflow:auto}
label{font-size:13px;color:var(--muted);display:block;margin:10px 0 4px}select,input,button{font:inherit}select,input[type=number]{width:100%;padding:6px;border:1px solid var(--line);border-radius:6px;background:#fff}
button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:7px 9px;cursor:pointer;margin-top:8px}button.primary{background:var(--accent);color:#fff}
.caseBlock{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:14px;padding:10px}.caseTitle{font-size:16px;font-weight:650;margin:0 0 8px}
.planes{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.plane h3{text-align:center;margin:0 0 6px;font-weight:500}
.frame{background:#050707;border:1px solid #111;min-height:250px;display:grid;place-items:center;overflow:hidden}.frame img{max-width:100%;max-height:42vh;image-rendering:pixelated}
.slider{display:grid;grid-template-columns:70px 1fr 54px;align-items:center;gap:8px;margin:9px 0}.slider input[type=range]{width:100%}
.metric{font-size:13px;line-height:1.5;color:#253033;background:#f7faf9;border:1px solid var(--line);border-radius:8px;padding:9px;margin:10px 0}.hint{font-size:13px;color:var(--muted);line-height:1.45}
@media(max-width:1050px){main{grid-template-columns:1fr;height:auto}.planes{grid-template-columns:1fr}aside{border-right:0}}
</style>
</head>
<body>
<header><strong>CBCT Pair Triplane Viewer</strong><span id="status"></span></header>
<main>
<aside>
  <label>Patient</label><select id="patient"></select>
  <label>Case A</label><select id="caseA"></select>
  <label>Case B</label><select id="caseB"></select>
  <button class="primary" id="loadPair">加载病例对</button>
  <div class="metric" id="metric"></div>
  <div class="slider"><span>Axial</span><input id="axial" type="range"><input id="axialNum" type="number"></div>
  <div class="slider"><span>Coronal</span><input id="coronal" type="range"><input id="coronalNum" type="number"></div>
  <div class="slider"><span>Sagittal</span><input id="sagittal" type="range"><input id="sagittalNum" type="number"></div>
  <label>Window low</label><input id="low" type="number">
  <label>Window high</label><input id="high" type="number">
  <button id="resetWindow">重置窗宽窗位</button>
  <p class="hint">上排为 Case A，下排为 Case B。三个切片滑块会同步控制上下两例的同一轴向位置。</p>
</aside>
<section class="viewer">
  <div class="caseBlock"><p class="caseTitle" id="titleA"></p><div class="planes" id="rowA"></div></div>
  <div class="caseBlock"><p class="caseTitle" id="titleB"></p><div class="planes" id="rowB"></div></div>
</section>
</main>
<script>
let meta=null,state={patient:"",caseA:"",caseB:"",shape:[1,1,1],window:{low:0,high:1}};
const planes=[["axial",0],["coronal",1],["sagittal",2]];
function el(id){return document.getElementById(id)}
function planeHtml(prefix){return planes.map(([p])=>`<div class="plane"><h3>${p[0].toUpperCase()+p.slice(1)}</h3><div class="frame"><img id="${prefix}_${p}"></div></div>`).join("")}
function fillSelect(sel, values, selected){sel.innerHTML=values.map(v=>`<option value="${v}" ${String(v)===String(selected)?"selected":""}>${v}</option>`).join("")}
async function init(){meta=await (await fetch("/api/meta")).json();state.patient=meta.default.patient_id;state.caseA=meta.default.case_a;state.caseB=meta.default.case_b;el("rowA").innerHTML=planeHtml("a");el("rowB").innerHTML=planeHtml("b");fillSelect(el("patient"),Object.keys(meta.patients),state.patient);updateCaseSelects();await loadPair()}
function updateCaseSelects(){const cases=meta.patients[el("patient").value]||[];fillSelect(el("caseA"),cases,state.caseA);fillSelect(el("caseB"),cases,state.caseB)}
async function loadPair(){state.patient=el("patient").value;state.caseA=el("caseA").value;state.caseB=el("caseB").value;const q=new URLSearchParams({patient_id:state.patient,case_a:state.caseA,case_b:state.caseB});const data=await (await fetch(`/api/pair?${q}`)).json();state.shape=data.shape_zyx;state.window={low:Math.floor(data.window.low),high:Math.ceil(data.window.high)};el("low").value=state.window.low;el("high").value=state.window.high;el("titleA").textContent=`Case A: patient ${state.patient}, CBCT ${state.caseA}`;el("titleB").textContent=`Case B: patient ${state.patient}, CBCT ${state.caseB}`;renderMetric(data.metric);setSliders(data.shape_zyx);refreshImages()}
function renderMetric(m){if(!m){el("metric").textContent="当前病例对没有 registration_fixed 指标记录。";return}const si=m.setup_error_SI_mm??m.SI_mm;const yaw=m.setup_error_yaw_deg??m.yaw_deg;el("metric").innerHTML=`Primary setup error<br><b>S-I=${fmt(si)} mm</b> | abs=${Math.abs(Number(si)).toFixed(3)} mm<br><b>yaw=${fmt(yaw)} deg</b> | abs=${Math.abs(Number(yaw)).toFixed(3)} deg<br><br>Diagnostic 6DOF<br>LR/AP/SI=${fmt(m.LR_mm)} / ${fmt(m.AP_mm)} / ${fmt(m.SI_mm)} mm<br>pitch/roll/yaw=${fmt(m.pitch_deg)} / ${fmt(m.roll_deg)} / ${fmt(m.yaw_deg)} deg`}
function fmt(v){return Number(v).toFixed(3)}
function setSliders(shape){planes.forEach(([p,axis])=>{const s=el(p),n=el(p+"Num");s.max=shape[axis]-1;s.value=Math.floor((shape[axis]-1)/2);n.value=s.value;s.oninput=()=>{n.value=s.value;refreshImages()};n.oninput=()=>{s.value=n.value;refreshImages()}})}
function imgUrl(which,plane){const caseId=which==="a"?state.caseA:state.caseB;const q=new URLSearchParams({patient_id:state.patient,case_id:caseId,plane,index:el(plane).value,low:el("low").value,high:el("high").value});return `/api/slice?${q}&t=${Date.now()}`}
function refreshImages(){planes.forEach(([p])=>{el(`a_${p}`).src=imgUrl("a",p);el(`b_${p}`).src=imgUrl("b",p)})}
el("patient").onchange=()=>{state.patient=el("patient").value;state.caseA=meta.patients[state.patient][0];state.caseB=meta.patients[state.patient][1]||meta.patients[state.patient][0];updateCaseSelects()}
el("loadPair").onclick=loadPair;el("low").onchange=refreshImages;el("high").onchange=refreshImages;el("resetWindow").onclick=async()=>loadPair();
init();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self.send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            if parsed.path == "/api/meta":
                metrics: dict[tuple[str, str, str], dict[str, str]] = SERVER_STATE["metrics"]  # type: ignore[assignment]
                first_metric = next(iter(metrics.values()), {})
                return self.send(json.dumps({
                    "patients": SERVER_STATE["patients"],
                    "default": SERVER_STATE["default"],
                    "registration_results": str(SERVER_STATE["registration_results"]),
                    "metric_fields": list(first_metric.keys()),
                }, ensure_ascii=False).encode("utf-8"), "application/json")
            if parsed.path == "/api/pair":
                qs = parse_qs(parsed.query)
                patient_id = qs["patient_id"][0]
                case_a = qs["case_a"][0]
                case_b = qs["case_b"][0]
                return self.send(json.dumps(pair_payload(patient_id, case_a, case_b), ensure_ascii=False).encode("utf-8"), "application/json")
            if parsed.path == "/api/slice":
                qs = parse_qs(parsed.query)
                path = cbct_path(Path(str(SERVER_STATE["dataset_root"])), qs["patient_id"][0], qs["case_id"][0])
                low = float(qs["low"][0]) if qs.get("low") else None
                high = float(qs["high"][0]) if qs.get("high") else None
                png = render_png(path, qs["plane"][0], int(qs["index"][0]), low, high)
                return self.send(png, "image/png")
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
            return self.send(body, "application/json", status=500)
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def pair_payload(patient_id: str, case_a: str, case_b: str) -> dict[str, object]:
    root = Path(str(SERVER_STATE["dataset_root"]))
    path_a = cbct_path(root, patient_id, case_a)
    path_b = cbct_path(root, patient_id, case_b)
    _, meta_a = cached_volume(str(path_a))
    _, meta_b = cached_volume(str(path_b))
    shape_a = list(meta_a["shape_zyx"])  # type: ignore[arg-type]
    shape_b = list(meta_b["shape_zyx"])  # type: ignore[arg-type]
    shape = [min(int(shape_a[i]), int(shape_b[i])) for i in range(3)]
    low = min(float(meta_a["p1"]), float(meta_b["p1"]))
    high = max(float(meta_a["p99"]), float(meta_b["p99"]))
    return {
        "shape_zyx": shape,
        "window": {"low": low, "high": high},
        "metric": metric_for(patient_id, case_a, case_b),
        "case_a": {"path": str(path_a), "meta": meta_a},
        "case_b": {"path": str(path_b), "meta": meta_b},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an interactive full-CBCT triplane viewer for two cases.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset_list" / "self_projection1208")
    parser.add_argument("--registration-results", type=Path, default=PROJECT_ROOT / "runs" / "setup_error" / "registration_si_yaw_fixed" / "registration_results.csv")
    parser.add_argument("--patient-id", default="24")
    parser.add_argument("--case-a", default="2")
    parser.add_argument("--case-b", default="8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    patients = scan_dataset(args.dataset_root)
    if args.patient_id not in patients:
        raise SystemExit(f"patient not found: {args.patient_id}")
    for case_id in (args.case_a, args.case_b):
        if case_id not in patients[args.patient_id]:
            raise SystemExit(f"case not found: patient={args.patient_id}, case={case_id}")

    SERVER_STATE["dataset_root"] = args.dataset_root
    SERVER_STATE["patients"] = patients
    SERVER_STATE["default"] = {"patient_id": args.patient_id, "case_a": args.case_a, "case_b": args.case_b}
    SERVER_STATE["registration_results"] = args.registration_results
    SERVER_STATE["metrics"] = read_metrics(args.registration_results)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ok": True, "url": f"http://{args.host}:{args.port}", "default": SERVER_STATE["default"]}, ensure_ascii=False))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
