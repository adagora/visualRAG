"""Snapshot identity and atomic publication shared by index builders."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


def load(index_dir: Path) -> dict | None:
    path = Path(index_dir) / "manifest.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("index_id"):
        raise ValueError(f"Invalid index manifest: {path}")
    return value


def fingerprint(index_dir: Path) -> str:
    root = Path(index_dir)
    manifest = load(root)
    digest = hashlib.sha256()
    if manifest:
        digest.update(manifest["index_id"].encode())
    names = ("manifest.json", "index.faiss", "articles.json", "summary.json", "text.json", "metadata.npz")
    paths = set(root / name for name in names)
    paths.update(root.glob("*gemini*"))
    for directory in root.glob("*gemini*"):
        if directory.is_dir():
            paths.update(directory.rglob("*"))
    for path in sorted(paths):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(str(root.resolve()).encode())
    return digest.hexdigest()


def source_id(index_dir: Path, article_id: int) -> str:
    manifest = load(index_dir)
    if manifest:
        for doc in manifest.get("documents", []):
            if doc["article_id"] == article_id:
                return doc["source_id"]
    return f"{fingerprint(index_dir)}:{article_id}"


def begin_snapshot(output: Path, index_id: str | None = None) -> Path:
    output = Path(output)
    index_id = index_id or uuid.uuid4().hex
    if Path(index_id).name != index_id or index_id in ("", ".", ".."):
        raise ValueError("Invalid snapshot ID")
    staging = output / "snapshots" / f".staging-{index_id}"
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def publish_snapshot(output: Path, staging: Path, manifest: dict) -> Path:
    output, staging = Path(output).resolve(), Path(staging).resolve()
    index_id = manifest["index_id"]
    if staging.parent != output / "snapshots" or staging.name != f".staging-{index_id}":
        raise ValueError("Staging path does not match snapshot identity")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    final = output / "snapshots" / index_id
    if final.exists():
        raise FileExistsError(final)
    summary_path = staging / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("index_file", "metadata_file"):
            if key in summary:
                summary[key] = str(final / Path(summary[key]).name)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(final)
    pointer = output / f".active-{index_id}.tmp"
    pointer.write_text(json.dumps({"schema_version": 1, "index_id": index_id,
                                   "path": f"snapshots/{index_id}"}), encoding="utf-8")
    os.replace(pointer, output / "active.json")
    return final
