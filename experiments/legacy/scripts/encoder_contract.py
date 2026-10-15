"""Fail explicitly when queries and document vectors use different encoders."""
from __future__ import annotations

import os

import layout

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."


class EncoderMismatch(RuntimeError):
    pass


def identity(index=None) -> dict:
    import index_manifest

    manifest = index_manifest.load((index or layout.DEFAULT).index_dir)
    recorded = dict(manifest.get("encoder", {})) if manifest else {}
    model = os.environ.get("PIXELRAG_EMBED_MODEL") or recorded.get("model") or DEFAULT_MODEL
    if recorded.get("model") and model != recorded["model"]:
        raise EncoderMismatch("PIXELRAG_EMBED_MODEL does not match the index manifest.")
    if recorded.get("adapter") or os.environ.get("PIXELRAG_ADAPTER"):
        raise EncoderMismatch("Adapter queries are disabled; rebuild with the base encoder.")
    return {"model": model, "revision": recorded.get("revision"),
            "dimensions": recorded.get("dimensions", 2048), "adapter": None,
            "instruction": recorded.get("instruction", DEFAULT_INSTRUCTION),
            "backend": recorded.get("backend", "qwen"),
            "index_id": manifest.get("index_id") if manifest else None}


def validate(actual: dict, expected: dict | None = None) -> None:
    expected = expected or identity()
    fields = ["model", "dimensions", "instruction", "adapter"]
    if expected.get("revision"):
        fields.append("revision")
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise EncoderMismatch(f"Query encoder {field} does not match the index; restart matching services.")
