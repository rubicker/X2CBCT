from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from .preprocess_pipeline import DATASET_ID, build_pipeline_stages, pipeline_layout


def bundle_sources(project_root: Path) -> tuple[Path, Path]:
    layout = pipeline_layout(project_root)
    return layout.manifest_root, layout.cache_root


def validate_bundle_sources(project_root: Path) -> None:
    stages = build_pipeline_stages(project_root)
    failures = []
    for stage in stages:
        result = stage.validator()
        if not result.ok:
            failures.append(f"{stage.name}: {result.reason}")
    if failures:
        raise RuntimeError("dataset family is incomplete:\n" + "\n".join(f"- {item}" for item in failures))


def inventory_sources(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    sources = bundle_sources(root)
    files = []
    total_bytes = 0
    for source in sources:
        if not source.is_dir():
            continue
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            size = path.stat().st_size
            files.append({"path": path.relative_to(root).as_posix(), "bytes": size})
            total_bytes += size
    return {
        "dataset_id": DATASET_ID,
        "source_roots": [str(path.relative_to(root).as_posix()) for path in sources],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_dataset_bundle(
    project_root: Path,
    output_root: Path,
    *,
    force: bool = False,
    runner: Callable[..., object] = subprocess.run,
) -> dict:
    root = Path(project_root).resolve()
    validate_bundle_sources(root)
    inventory = inventory_sources(root)
    if inventory["file_count"] == 0:
        raise RuntimeError("dataset bundle has no files")
    tar = shutil.which("tar")
    if tar is None:
        raise FileNotFoundError("tar executable was not found in PATH")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    package_root = Path(output_root).resolve() / f"{DATASET_ID}_{stamp}"
    if package_root.exists():
        if not force:
            raise FileExistsError(f"package output already exists: {package_root}")
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    archive_path = package_root / f"{DATASET_ID}.tar.gz"
    relative_sources = inventory["source_roots"]
    command = [tar, "-czf", str(archive_path), "-C", str(root), *relative_sources]
    runner(command, cwd=root, check=True)

    archive_hash = _sha256(archive_path)
    manifest = {
        "schema": "x2cbct_difnet_dataset_bundle.v1",
        "dataset_id": DATASET_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "archive": archive_path.name,
        "archive_sha256": archive_hash,
        "source_roots": relative_sources,
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "restore_command": f"tar -xzf {archive_path.name} -C ~/X2CBCT",
    }
    (package_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_root / "files.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_root / "SHA256SUMS.txt").write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="ascii",
    )
    return {**manifest, "package_root": str(package_root), "archive_path": str(archive_path)}
