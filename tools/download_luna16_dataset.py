from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


RECORD_IDS = ("3723295", "4121926")
DEFAULT_PROXY = "http://127.0.0.1:7897"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def md5_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_record(record_id: str, source_dir: Path, proxy: str, log_path: Path) -> Path:
    path = source_dir / f"zenodo_{record_id}.json"
    source_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    url = f"https://zenodo.org/api/records/{record_id}"
    run(
        [
            "curl.exe",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "60",
            "--proxy",
            proxy,
            url,
            "-o",
            str(path),
        ],
        log_path,
    )
    return path


def target_for_zip(name: str, root: Path) -> Path:
    stem = name[:-4]
    if stem.startswith("subset"):
        return root / stem
    if stem == "seg-lungs-LUNA16":
        return root / "seg-lungs-LUNA16"
    if stem == "evaluationScript":
        return root / "evaluationScript"
    if stem == "candidates_V2":
        return root
    return root / stem


def extraction_marker(root: Path, name: str) -> Path:
    return root / "_manifests" / f"{name}.done.json"


def is_done(root: Path, name: str, expected_md5: str) -> bool:
    marker = extraction_marker(root, name)
    if not marker.exists():
        return False
    try:
        data = load_json(marker)
    except Exception:
        return False
    return data.get("md5") == expected_md5 and data.get("status") == "done"


def write_done(root: Path, name: str, expected_md5: str, payload: dict[str, Any]) -> None:
    data = {"name": name, "md5": expected_md5, "status": "done", **payload}
    save_json(extraction_marker(root, name), data)


def download_file(file_info: dict[str, Any], archive_dir: Path, proxy: str, log_path: Path) -> Path:
    name = file_info["key"]
    out = archive_dir / name
    url = file_info["links"]["self"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "curl.exe",
            "-L",
            "--fail",
            "--retry",
            "20",
            "--retry-delay",
            "20",
            "--connect-timeout",
            "120",
            "--continue-at",
            "-",
            "--proxy",
            proxy,
            url,
            "-o",
            str(out),
        ],
        log_path,
    )
    return out


def extract_zip(zip_path: Path, root: Path, name: str) -> dict[str, Any]:
    target = target_for_zip(name, root)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt zip member in {zip_path}: {bad}")
        members = zf.namelist()
        zf.extractall(target if name.startswith("subset") else root)
    return {
        "target": str(target),
        "member_count": len(members),
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def handle_file(
    file_info: dict[str, Any],
    root: Path,
    archive_dir: Path,
    proxy: str,
    keep_archives: bool,
    log_path: Path,
) -> dict[str, Any]:
    name = file_info["key"]
    expected_md5 = file_info["checksum"].replace("md5:", "")
    if is_done(root, name, expected_md5):
        return {"name": name, "status": "skipped", "reason": "done marker exists"}

    archive_path = archive_dir / name
    if not archive_path.exists():
        archive_path = download_file(file_info, archive_dir, proxy, log_path)

    actual_md5 = md5_file(archive_path)
    if actual_md5 != expected_md5:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {name}: expected {expected_md5}, got {actual_md5}")

    if name.endswith(".zip"):
        payload = extract_zip(archive_path, root, name)
    else:
        target = root / name
        shutil.copy2(archive_path, target)
        payload = {"target": str(target), "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    write_done(root, name, expected_md5, payload)
    if name.endswith(".zip") and not keep_archives:
        archive_path.unlink()
    return {"name": name, "status": "done", **payload}


def collect_files(record_paths: list[Path]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in record_paths:
        record = load_json(path)
        for item in record["files"]:
            enriched = dict(item)
            enriched["record_id"] = record["id"]
            enriched["record_title"] = record["metadata"]["title"]
            files.append(enriched)
    return sorted(files, key=lambda item: item["key"])


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset_list/LUNA16")
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--only", nargs="*", help="Optional file names to process")
    args = parser.parse_args()

    root = Path(args.root)
    source_dir = root / "_source"
    archive_dir = root / "_archives"
    log_path = root / "_logs" / "download_luna16.log"
    root.mkdir(parents=True, exist_ok=True)

    record_paths = [fetch_record(record_id, source_dir, args.proxy, log_path) for record_id in RECORD_IDS]
    files = collect_files(record_paths)
    if args.only:
        wanted = set(args.only)
        files = [item for item in files if item["key"] in wanted]

    manifest = {
        "dataset": "LUNA16",
        "root": str(root),
        "records": [str(p) for p in record_paths],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keep_archives": args.keep_archives,
        "items": [],
    }
    save_json(root / "download_manifest.json", manifest)

    for item in files:
        print(f"[LUNA16] processing {item['key']}")
        result = handle_file(item, root, archive_dir, args.proxy, args.keep_archives, log_path)
        manifest["items"].append(result)
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json(root / "download_manifest.json", manifest)
        print(f"[LUNA16] {item['key']} -> {result['status']}")

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(root / "download_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
