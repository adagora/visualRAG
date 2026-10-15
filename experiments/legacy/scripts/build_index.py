#!/usr/bin/env python
"""Build fresh immutable visual snapshots and atomically select the active one."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import layout

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


BATCH_SIZE = int(os.environ.get("PIXELRAG_BATCH_SIZE", "16"))
INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOCUMENT_INSTRUCTION = "Represent the user's input."



def resolve_model(model, revision):
    local = layout._anchored(model, ROOT)
    if local.is_dir():
        if revision:
            raise ValueError("Local model cannot specify a Hugging Face revision")
        return str(local.resolve()), None
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=model, revision=revision))
    return str(path), path.name


def snapshot_embed(index, embed, model_path):
    import numpy as np
    emb = index.index_dir / "embeddings"
    emb.mkdir()
    ids = embed.get("gpu_ids", [0])
    if not isinstance(ids, list) or not ids or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("embed.gpu_ids must contain nonnegative integers")
    subprocess.run([PY, "-m", "pixelrag_embed.embed", "--shard-dir", str(index.tiles_dir),
                    "--output-dir", str(emb), "--model", model_path,
                    "--gpu-ids", ",".join(map(str, ids)), "--backend", "direct_gpu",
                    "--mode", "chunks", "--instruction", DOCUMENT_INSTRUCTION,
                    "--batch-size", str(embed.get("batch_size", BATCH_SIZE))], check=True)
    count, dims = 0, set()
    for shard in sorted(emb.glob("shard_*.npz")):
        with np.load(shard) as data:
            n, d = data["embeddings"].shape
            count += n
            dims.add(d)
    if not count or len(dims) != 1:
        raise ValueError("No embeddings or inconsistent dimensions")
    subprocess.run([PY, "-m", "pixelrag_embed.index", "build", "--embeddings-dir", str(emb),
                    "--output-dir", str(index.index_dir), "--nlist", str(min(4096, max(1, count // 40)))], check=True)
    return count, dims.pop()


def validate_snapshot(index, documents, count, dimensions):
    import json
    import faiss
    import numpy as np
    from PIL import Image
    expected = set()
    for doc in documents:
        if "snapshot_path" in doc:
            import hashlib
            source = (index.index_dir / doc["snapshot_path"]).resolve()
            if source.parent != (index.index_dir / "sources").resolve() or not source.is_file():
                raise ValueError("Missing snapshot source PDF")
            if hashlib.sha256(source.read_bytes()).hexdigest() != doc["sha256"]:
                raise ValueError("Snapshot source SHA256 mismatch")
        aid = doc["article_id"]
        pages = index.page_images(aid)
        if len(pages) != doc["pages"] or not pages:
            raise ValueError("Page count mismatch")
        chunks = json.loads(index.chunks_manifest(aid).read_text(encoding="utf-8"))["chunks"]
        for chunk in chunks:
            tile = chunk["tile_index"]
            if not 0 <= tile < len(pages) or not (index.tile_dir(aid) / chunk["file"]).is_file():
                raise ValueError("Missing page or crop")
            with Image.open(pages[tile]) as image:
                w, h = image.size
            x, y, cw, ch = (chunk[k] for k in ("x_offset", "y_offset", "width", "height"))
            if min(x, y) < 0 or min(cw, ch) <= 0 or x + cw > w or y + ch > h:
                raise ValueError("Chunk geometry outside page")
            key = (aid, tile, chunk["chunk_index"])
            if key in expected:
                raise ValueError("Duplicate chunk identity")
            expected.add(key)
    ix = faiss.read_index(str(index.faiss_index))
    with np.load(index.index_dir / "metadata.npz") as meta:
        actual = list(zip(meta["article_ids"].tolist(), meta["tile_indices"].tolist(), meta["chunk_indices"].tolist()))
    if count != len(expected) or ix.ntotal != count or ix.d != dimensions or len(actual) != count or set(actual) != expected:
        raise ValueError("Vectors and chunks do not match")
    if len(json.loads(index.articles_json.read_text(encoding="utf-8"))) != len(documents):
        raise ValueError("Article count mismatch")


def build(cfg, geometry, adapter=None):
    import shutil
    import chunk_multiscale
    import index_manifest
    import prepare_hires
    if cfg.get("source", {}).get("type", "pdf") != "pdf":
        raise ValueError("Snapshot renderer currently requires source.type=pdf")
    embed = cfg.get("embed", {})
    if adapter or embed.get("adapter"):
        raise ValueError("Adapters require matching query encoding and are unsupported")
    if embed.get("instruction", INSTRUCTION) != INSTRUCTION:
        raise ValueError("This encoder requires the standard retrieval instruction")
    if embed.get("document_instruction", DOCUMENT_INSTRUCTION) != DOCUMENT_INSTRUCTION:
        raise ValueError("Document embedding requires the native Qwen document instruction")
    if embed.get("device", "cuda") != "cuda" or embed.get("backend", "direct_gpu") != "direct_gpu":
        raise ValueError("Snapshot builds require cuda and direct_gpu")
    model = embed.get("model", "Qwen/Qwen3-VL-Embedding-2B")
    model_path, revision = resolve_model(model, embed.get("revision"))
    if revision is None:
        model = model_path
    output = layout.build_layout(cfg).index_dir
    staging = index_manifest.begin_snapshot(output)
    index_id = staging.name.removeprefix(".staging-")
    try:
        documents = prepare_hires.prepare({**cfg, "output": str(staging)})
        index = layout.IndexLayout(staging)
        for doc in documents:
            chunk_multiscale.process_article(index.tile_dir(doc["article_id"]), doc["article_id"], doc["path"], **geometry)
        count, dimensions = snapshot_embed(index, embed, model_path)
        import lexical
        import json
        lexical.build_sidecar(json.loads(index.articles_json.read_text(encoding="utf-8")),
                              out_path=index.text_sidecar)
        validate_snapshot(index, documents, count, dimensions)
        manifest = {"schema_version": 1, "index_id": index_id,
                    "encoder": {"backend": "qwen", "model": model, "revision": revision,
                                "adapter": None, "dimensions": dimensions,
                                "instruction": INSTRUCTION,
                                "document_instruction": DOCUMENT_INSTRUCTION},
                    "render": {"dpi": 200, "jpeg_quality": 85},
                    "chunking": {**geometry, "blank_ink": chunk_multiscale.BLANK_INK,
                                 "min_side": chunk_multiscale.MIN_SIDE},
                    "documents": documents, "vector_count": count}
        return index_manifest.publish_snapshot(output, staging, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    import chunk_multiscale
    ap = argparse.ArgumentParser(description="Build and publish a fresh immutable snapshot")
    ap.add_argument("--config", type=Path, help="Index configuration (default: PIXELRAG_CONFIG or pixelrag.yaml)")
    ap.add_argument("--output", type=Path, help="Snapshot collection; preserve existing published builds")
    ap.add_argument("--revision", help="Pinned Hugging Face encoder revision")
    ap.add_argument("--stock", action="store_true", help="Alias for --geometry page")
    ap.add_argument("--geometry", choices=["page", "regions", "multiscale"], default="regions")
    ap.add_argument("--width", type=int, default=875)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--stride-y", type=int, default=640)
    ap.add_argument("--stride-x", type=int, default=875)
    ap.add_argument("--force", action="store_true", help="Every snapshot build is already fresh")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--adapter")
    args = ap.parse_args()
    if args.skip_render:
        ap.error("--skip-render cannot guarantee reproducible snapshots")
    chunk_multiscale._offsets(1, args.width, args.stride_x)
    chunk_multiscale._offsets(1, args.height, args.stride_y)
    geometry = {"mode": "page" if args.stock else args.geometry, "width": args.width,
                "height": args.height, "stride_y": args.stride_y, "stride_x": args.stride_x}
    if args.config:
        os.environ["PIXELRAG_CONFIG"] = str(layout._anchored(args.config, ROOT / "pixelrag.yaml"))
    cfg = layout.build_config()
    if args.output:
        cfg["output"] = str(layout._anchored(args.output, ROOT / "index"))
    if args.revision:
        cfg.setdefault("embed", {})["revision"] = args.revision
    print(f"Published {build(cfg, geometry, args.adapter)}")


if __name__ == "__main__":
    main()
