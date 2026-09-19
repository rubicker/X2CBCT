from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import write_json
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


def _resolve(cache_root: Path, value: str) -> Path:
    path = Path(str(value).replace("/", "\\"))
    return path if path.is_absolute() else cache_root / path


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _load_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        return {
            "sample_id": _scalar(raw["sample_id"]),
            "split": _scalar(raw["split"]),
            "view_key": _scalar(raw["view_key"]),
            "case_key": _scalar(raw["case_key"]) if "case_key" in raw.files else "",
            "cbct_payload_path": _scalar(raw["cbct_payload_path"]),
            "projection_payload_paths": [str(item) for item in raw["projection_payload_paths"].tolist()],
            "projection_count": len(raw["projection_payload_paths"]),
            "sample_path": str(path),
            "mtime": path.stat().st_mtime,
        }


def _to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "sample"


def _write_preview(sample: dict[str, Any], cache_root: Path, output_dir: Path) -> dict[str, Any] | None:
    try:
        cbct = np.load(_resolve(cache_root, str(sample["cbct_payload_path"])))
        projection_paths = [str(item) for item in sample["projection_payload_paths"]]
        projection = np.load(_resolve(cache_root, projection_paths[0])) if projection_paths else None
    except Exception as exc:
        return {"sample_id": sample["sample_id"], "error": repr(exc)}

    volume = np.asarray(cbct, dtype=np.float32)
    if volume.ndim != 3:
        return {"sample_id": sample["sample_id"], "error": f"cbct payload is not 3D: {volume.shape}"}
    z, y, x = (dim // 2 for dim in volume.shape)
    panels = [
        ("axial", volume[z, :, :]),
        ("coronal", volume[:, y, :]),
        ("sagittal", volume[:, :, x]),
    ]
    if projection is not None:
        panels.append(("projection", np.squeeze(np.asarray(projection, dtype=np.float32))))

    tiles = []
    for label, array in panels:
        image = _to_uint8(array)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)

    while len(tiles) < 4:
        tiles.append(np.zeros((192, 192, 3), dtype=np.uint8))
    canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    png_path = asset_dir / f"{_safe_name(str(sample['sample_id']))}.png"
    cv2.imwrite(str(png_path), canvas)
    return {
        "sample_id": sample["sample_id"],
        "png": png_path.resolve().relative_to(output_dir.resolve()).as_posix(),
        "cbct_shape_zyx": [int(item) for item in volume.shape],
        "projection_count": len(projection_paths),
    }


def _latest_files(cache_root: Path, limit: int = 20) -> list[dict[str, Any]]:
    roots = [cache_root / "cases", cache_root / "projections", *cache_root.glob("view_*")]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [
        {
            "path": str(path.resolve()),
            "relative_path": path.resolve().relative_to(cache_root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for path in files[:limit]
    ]


def _scan(cache_root: Path, view_key: str, split: str) -> dict[str, Any]:
    case_payloads = list((cache_root / "cases").glob("*/cbct_norm.npy")) if (cache_root / "cases").exists() else []
    projection_payloads = list((cache_root / "projections").rglob("*.npy")) if (cache_root / "projections").exists() else []
    splits = ("train", "eval", "test") if split == "all" else (split,)
    sample_paths: list[Path] = []
    for name in splits:
        sample_paths.extend((cache_root / view_key / name).glob("*.npz"))
    samples = []
    for path in sample_paths:
        try:
            samples.append(_load_sample(path))
        except Exception as exc:
            samples.append({"sample_id": path.stem, "sample_path": str(path), "error": repr(exc), "mtime": path.stat().st_mtime})
    samples.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    status_samples = []
    for sample in samples[:200]:
        status_sample = dict(sample)
        projection_paths = status_sample.pop("projection_payload_paths", [])
        if projection_paths:
            status_sample["first_projection_payload_path"] = projection_paths[0]
        status_samples.append(status_sample)
    manifest_path = cache_root / "cache_manifest.json"
    return {
        "schema": "x2cbct_difnet_v0b_cache_build_watch.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(cache_root.resolve()),
        "view_key": view_key,
        "split": split,
        "manifest_exists": manifest_path.exists(),
        "counts": {
            "case_payloads": len(case_payloads),
            "projection_payloads": len(projection_payloads),
            "samples": len(sample_paths),
        },
        "recent_samples": status_samples,
        "_preview_samples": samples[:200],
        "latest_files": _latest_files(cache_root),
    }


def _write_html(status: dict[str, Any], output_dir: Path, previews: list[dict[str, Any]]) -> Path:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    counts = status["counts"]
    preview_html = "".join(
        f"<a class=\"preview\" href=\"{esc(item.get('png'))}\"><img src=\"{esc(item.get('png'))}\" alt=\"{esc(item.get('sample_id'))}\"><span>{esc(item.get('sample_id'))}</span></a>"
        for item in previews
        if item.get("png")
    )
    rows = "".join(
        f"<tr><td>{esc(item.get('modified_at'))}</td><td><code>{esc(item.get('relative_path'))}</code></td><td>{esc(item.get('bytes'))}</td></tr>"
        for item in status["latest_files"]
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DIF-Net V0b Cache Build Watch</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f5f7f6; color: #17201c; }}
    header, main {{ padding: 18px 22px; }}
    header {{ background: #fff; border-bottom: 1px solid #d7ded9; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: #65716b; font-size: 13px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin: 16px 0; }}
    .stat, table {{ background: #fff; border: 1px solid #d7ded9; }}
    .stat {{ padding: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .previews {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
    .preview {{ width: 192px; color: #0f766e; text-decoration: none; font-size: 12px; overflow-wrap: anywhere; }}
    .preview img {{ display: block; width: 192px; height: 192px; border: 1px solid #d7ded9; background: #0b0f0d; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e2e7e4; padding: 8px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #edf3f1; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>DIF-Net V0b Cache Build Watch</h1>
    <div class="muted">cache_root: <code>{esc(status["cache_root"])}</code></div>
    <div class="muted">updated_at: {esc(status["generated_at"])}</div>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="value">{counts["case_payloads"]}</div><div>case payloads</div></div>
      <div class="stat"><div class="value">{counts["projection_payloads"]}</div><div>projection payloads</div></div>
      <div class="stat"><div class="value">{counts["samples"]}</div><div>{esc(status["view_key"])} samples</div></div>
      <div class="stat"><div class="value">{esc(status["manifest_exists"])}</div><div>manifest exists</div></div>
    </section>
    <section class="previews">{preview_html}</section>
    <table>
      <thead><tr><th>modified</th><th>file</th><th>bytes</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </main>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    view_key = view_key_from_count(args.view_count)
    status = _scan(cache_root, view_key, args.split)
    previews = []
    for sample in status.pop("_preview_samples", [])[: max(0, int(args.max_preview_samples))]:
        if "error" not in sample:
            preview = _write_preview(sample, cache_root, output_dir)
            if preview is not None:
                previews.append(preview)
    status["previews"] = previews
    status_path = write_json(status, output_dir / "watch_status.json")
    html_path = _write_html(status, output_dir, previews)
    return {"ok": True, "status_json": str(status_path), "index_html": str(html_path), **status["counts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a DIF-Net V0b cache build and export a live HTML dashboard.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument("--split", choices=("train", "eval", "test", "all"), default="all")
    parser.add_argument("--max-preview-samples", type=int, default=6)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--stop-when-manifest", action="store_true")
    args = parser.parse_args()

    result = run_once(args)
    print(json.dumps(result, ensure_ascii=False))
    while args.watch:
        if args.stop_when_manifest and (args.cache_root / "cache_manifest.json").exists():
            break
        time.sleep(max(1.0, float(args.interval_seconds)))
        result = run_once(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
