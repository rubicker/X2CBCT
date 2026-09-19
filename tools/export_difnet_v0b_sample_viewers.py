from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from x2cbct_difnet.cache import read_json, resolve_cache_path, write_json
from x2cbct_difnet.cached_dataset import load_cache_manifest, samples_for
from x2cbct_difnet.normalization import SUPPORTED_VIEW_COUNTS, view_key_from_count


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "sample"


def _compact_token(value: Any, max_len: int = 28) -> str:
    text = _safe_name(str(value))
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[-(max_len - 9):]}_{digest}"


def _viewer_folder_name(row: dict[str, Any], ordinal: int) -> str:
    digest = hashlib.sha1(str(row.get("sample_id", "")).encode("utf-8")).hexdigest()[:8]
    patient = _compact_token(row.get("patient_id", "patient"), max_len=18)
    case = _compact_token(row.get("case_id", "case"), max_len=28)
    view = _safe_name(str(row.get("view_key", "view")))
    split = _safe_name(str(row.get("split", "split")))
    return _safe_name(f"{ordinal:03d}_{split}_{view}_patient{patient}_case{case}_{digest}")


def _sample_case_ids(sample: dict[str, Any]) -> tuple[str, str]:
    patient_id = str(sample.get("patient_id", "unknown"))
    case_id = str(sample.get("case_id", "unknown"))
    if patient_id != "unknown" and case_id != "unknown":
        return patient_id, case_id
    if case_id != "unknown":
        return str(sample.get("dataset_group") or sample.get("source") or "case"), case_id
    sample_id = str(sample.get("sample_id", ""))
    parts = sample_id.split("_")
    if len(parts) >= 2:
        return parts[0].replace("patient", ""), parts[1].replace("case", "")
    return patient_id, case_id


def _case_manifest(cache_root: Path, sample: dict[str, Any]) -> dict[str, Any]:
    rel = str(sample.get("case_manifest_path", ""))
    if not rel:
        return {}
    path = resolve_cache_path(cache_root, rel)
    return read_json(path) if path.exists() else {}


def _row_metadata(cache_root: Path, sample: dict[str, Any], ordinal: int) -> dict[str, Any]:
    patient_id, case_id = _sample_case_ids(sample)
    case_manifest = _case_manifest(cache_root, sample)
    selected = sample.get("selected_projections", [])
    projection_files = []
    gantry_angles = []
    target_angles = []
    angular_residuals = []
    if isinstance(selected, list):
        for item in selected:
            projection_files.append(str(item.get("file", "")))
            gantry_angles.append(item.get("gantry_angle_deg"))
            target_angles.append(item.get("target_angle_deg"))
            angular_residuals.append(item.get("angular_residual_deg"))
    if not projection_files and isinstance(sample.get("projection_paths"), list):
        projection_files = [str(item) for item in sample["projection_paths"]]
        gantry_angles = [float(item) for item in sample.get("angles_deg", [])]
        target_angles = [float(item) for item in sample.get("angles_deg", [])]
        angular_residuals = [0.0 for _ in target_angles]
    return {
        "ordinal": int(ordinal),
        "sample_id": str(sample.get("sample_id", "")),
        "split": str(sample.get("split", "")),
        "view_key": str(sample.get("view_key", "")),
        "patient_id": patient_id,
        "case_id": case_id,
        "sample_path": str(sample.get("sample_path", cache_root / "cache_manifest.json")),
        "case_manifest_path": str(sample.get("case_manifest_path", "")),
        "target_payload_path": case_manifest.get("raw_case_path") or sample.get("processed_cbct_path"),
        "geometry_branch": sample.get("geometry_branch") or case_manifest.get("geometry_branch"),
        "projection_files": projection_files,
        "target_angles_deg": target_angles,
        "gantry_angles_deg": gantry_angles,
        "angular_residuals_deg": angular_residuals,
    }


def _write_batch_html(
    rows: list[dict[str, Any]],
    output_root: Path,
    cache_root: Path,
    title: str,
    split_counts: dict[str, int],
) -> Path:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    row_html = []
    for row in rows:
        angles = ", ".join(f"{float(v):.2f}" for v in row["gantry_angles_deg"] if v is not None)
        residuals = ", ".join(f"{float(v):.3f}" for v in row["angular_residuals_deg"] if v is not None)
        row_html.append(
            "<tr>"
            f"<td>{row['display_order']:03d}</td>"
            f"<td><a href=\"{esc(row['viewer_relative_path'])}\">{esc(row['sample_id'])}</a></td>"
            f"<td>{esc(row['patient_id'])}</td>"
            f"<td>{esc(row['case_id'])}</td>"
            f"<td>{esc(row['split'])}</td>"
            f"<td>{esc(row['view_key'])}</td>"
            f"<td><code>{esc(row.get('target_payload_path'))}</code></td>"
            f"<td>{esc(angles)}</td>"
            f"<td>{esc(residuals)}</td>"
            "</tr>"
        )
    split_summary = ", ".join(f"{name}: {count}" for name, count in split_counts.items())
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f6f7f7; color: #17201c; }}
    header {{ padding: 16px 20px; background: #fff; border-bottom: 1px solid #d7ded9; }}
    main {{ padding: 16px 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: #66736d; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7ded9; }}
    th, td {{ border-bottom: 1px solid #e2e7e4; padding: 8px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #edf3f1; z-index: 1; }}
    code {{ font-family: Consolas, monospace; overflow-wrap: anywhere; }}
    a {{ color: #0f766e; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="muted">cache_root: <code>{esc(cache_root)}</code></div>
    <div class="muted">viewer_count: {len(rows)}</div>
    <div class="muted">split_count: {esc(split_summary)}</div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>#</th><th>sample</th><th>patient</th><th>case</th><th>split</th><th>view</th>
          <th>target payload path</th><th>GantryAngle</th><th>residual</th>
        </tr>
      </thead>
      <tbody>
        {''.join(row_html)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    path = output_root / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def _relative_href(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, from_dir).replace("\\", "/")


def _nav_item(current_dir: Path, row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    target = Path(str(row["viewer_dir"])) / "index.html"
    return {
        "href": _relative_href(current_dir, target),
        "sample_id": row["sample_id"],
        "split": row["split"],
        "patient_id": row["patient_id"],
        "case_id": row["case_id"],
        "label": f"{row['split']} patient {row['patient_id']} / case {row['case_id']}",
    }


def _inject_navigation(rows: list[dict[str, Any]], output_root: Path) -> None:
    total = len(rows)
    for idx, row in enumerate(rows):
        viewer_dir = Path(str(row["viewer_dir"]))
        data_path = viewer_dir / "viewer_data.json"
        js_path = viewer_dir / "viewer_data.js"
        data = json.loads(data_path.read_text(encoding="utf-8-sig"))
        data["navigation"] = {
            "position": idx + 1,
            "total": total,
            "index": {
                "href": _relative_href(viewer_dir, output_root / "index.html"),
                "label": "Index",
            },
            "previous": _nav_item(viewer_dir, rows[idx - 1] if idx > 0 else None),
            "next": _nav_item(viewer_dir, rows[idx + 1] if idx + 1 < total else None),
        }
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        js_path.write_text("window.VIEWER_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def _sort_token(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _viewer_order_key(row: dict[str, Any]) -> tuple[tuple[int, str], tuple[int, str], str]:
    return (_sort_token(row["patient_id"]), _sort_token(row["case_id"]), str(row["split"]))


def _splits_for_request(manifest: dict[str, Any], requested: str, view_key: str) -> tuple[str, ...]:
    if requested != "all":
        return (requested,)
    present = {str(item["split"]) for item in manifest.get("samples", []) if str(item.get("view_key")) == view_key}
    ordered = [split for split in ("train", "eval", "test") if split in present]
    ordered.extend(sorted(present.difference(ordered)))
    if not ordered:
        raise ValueError(f"no samples for view_key={view_key}")
    return tuple(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch export case-named DIF-Net V0b sample viewers.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval", "test", "all"), default="test")
    parser.add_argument("--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--slice-stride", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest = load_cache_manifest(cache_root)
    view_key = view_key_from_count(args.view_count)
    requested_splits = _splits_for_request(manifest, args.split, view_key)
    split_samples = [(split, samples_for(manifest, split, view_key)) for split in requested_splits]
    start = max(0, int(args.start_index))
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    viewer_script = PROJECT_ROOT / "tools" / "view_difnet_v0b_sample.py"
    split_counts: dict[str, int] = {}
    for split, samples in split_samples:
        selected_samples = samples[start:]
        if args.max_samples is not None:
            selected_samples = selected_samples[: max(0, int(args.max_samples))]
        split_counts[split] = len(selected_samples)
        for offset, sample in enumerate(selected_samples):
            ordinal = start + offset
            row = _row_metadata(cache_root, sample, ordinal)
            folder_name = _viewer_folder_name(row, ordinal)
            viewer_dir = output_root / split / folder_name if args.split == "all" else output_root / folder_name
            command = [
                sys.executable,
                str(viewer_script),
                "--cache-root",
                str(cache_root),
                "--slice-stride",
                str(args.slice_stride),
                "--output-dir",
                str(viewer_dir),
            ]
            if "sample_path" in sample:
                sample_path = resolve_cache_path(cache_root, str(sample["sample_path"]))
                command.extend(["--sample-path", str(sample_path)])
            else:
                sample_path = cache_root / "cache_manifest.json"
                command.extend(["--split", split, "--view-count", str(args.view_count), "--index", str(ordinal)])
            completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"viewer export failed for {sample_path}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            row["viewer_dir"] = str(viewer_dir)
            row["viewer_relative_path"] = f"{split}/{folder_name}/index.html" if args.split == "all" else f"{folder_name}/index.html"
            rows.append(row)

    rows.sort(key=_viewer_order_key)
    for display_order, row in enumerate(rows):
        row["display_order"] = display_order
    _inject_navigation(rows, output_root)
    index_payload = {
        "schema": "x2cbct_difnet_v0b_sample_viewer_batch.v1",
        "cache_root": str(cache_root),
        "cache_version": manifest.get("cache_version"),
        "split": args.split,
        "splits": list(requested_splits),
        "split_counts": split_counts,
        "view_key": view_key,
        "start_index": start,
        "viewer_count": len(rows),
        "slice_stride": int(args.slice_stride),
        "rows": rows,
    }
    json_path = write_json(index_payload, output_root / "viewer_batch_index.json")
    html_path = _write_batch_html(
        rows,
        output_root,
        cache_root,
        f"{manifest.get('cache_version')} {args.split} {view_key}",
        split_counts,
    )
    print(json.dumps({"ok": True, "viewer_count": len(rows), "index_json": str(json_path), "index_html": str(html_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
