"""Resolve immutable snapshots and atomically publish a complete build."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid


def resolve_snapshot(path: Path) -> Path:
    root = Path(path).resolve()
    if (root / "gemini.json").is_file():
        return root
    pointer = json.loads((root / "active.json").read_text(encoding="utf-8-sig"))
    identity = pointer.get("index_id")
    if not isinstance(identity, str) or not identity or Path(identity).name != identity or identity in (".", ".."):
        raise ValueError("Invalid active snapshot identity.")
    if pointer.get("schema_version") != 1 or pointer.get("path") != f"snapshots/{identity}":
        raise ValueError("Invalid active snapshot pointer.")
    snapshot = (root / pointer["path"]).resolve()
    if not snapshot.is_relative_to(root / "snapshots"):
        raise ValueError("Snapshot escapes collection.")
    manifest = load(snapshot)
    if manifest is None or manifest["index_id"] != identity:
        raise ValueError("Active pointer/snapshot identity mismatch.")
    return snapshot


def load(index_dir: Path) -> dict | None:
    path = Path(index_dir) / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != 1 or not value.get("index_id"):
        raise ValueError(f"Invalid index manifest: {path}")
    return value


def begin_snapshot(output: Path, index_id: str | None = None) -> Path:
    identity = index_id or uuid.uuid4().hex
    if Path(identity).name != identity or identity in ("", ".", ".."):
        raise ValueError("Invalid snapshot ID.")
    stage = Path(output) / "snapshots" / f".staging-{identity}"
    stage.mkdir(parents=True, exist_ok=False)
    return stage


def publish_snapshot(output: Path, staging: Path, manifest: dict) -> Path:
    root, stage = Path(output).resolve(), Path(staging).resolve()
    identity = manifest["index_id"]
    if stage.parent != root / "snapshots" or stage.name != f".staging-{identity}":
        raise ValueError("Staging path does not match snapshot identity.")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema.")
    final = root / "snapshots" / identity
    if final.exists():
        raise FileExistsError(final)
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    stage.rename(final)
    pointer = root / f".active-{identity}.tmp"
    try:
        pointer.write_text(json.dumps({"schema_version": 1, "index_id": identity,
                                       "path": f"snapshots/{identity}"}), encoding="utf-8")
        os.replace(pointer, root / "active.json")
    finally:
        pointer.unlink(missing_ok=True)
    return final
