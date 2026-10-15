"""The small FAISS service: vector queries, snapshot identity, and crop images.

Models live in encoder.py. This process never imports torch or encodes text.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import layout

app = FastAPI(title="PixelRAG vector search")
_state = {}


class VectorQuery(BaseModel):
    embedding: list[float] = Field(min_length=1, max_length=4096)


class SearchRequest(BaseModel):
    queries: list[VectorQuery] = Field(min_length=1, max_length=16)
    n_docs: int = Field(default=24, ge=1, le=10000)


def load(index=None):
    import faiss
    import index_manifest

    idx = index or layout.DEFAULT
    vectors = faiss.read_index(str(idx.faiss_index))
    with np.load(idx.index_dir / "metadata.npz", allow_pickle=False) as stored:
        meta = {key: stored[key].copy() for key in stored.files}
    required = ("article_ids", "tile_indices", "chunk_indices")
    if any(key not in meta or len(meta[key]) != vectors.ntotal for key in required):
        raise ValueError("FAISS vectors and page metadata disagree; rebuild this snapshot.")
    manifest = index_manifest.load(idx.index_dir)
    if manifest and (manifest["vector_count"] != vectors.ntotal or
                     manifest["encoder"]["dimensions"] != vectors.d):
        raise ValueError("FAISS vectors do not match the completed snapshot manifest.")
    if hasattr(vectors, "nprobe"):
        vectors.nprobe = min(int(os.environ.get("PIXELRAG_NPROBE", "128")), vectors.nlist)
    _state.update(index=vectors, meta=meta, layout=idx, manifest=manifest,
                  fingerprint=index_manifest.fingerprint(idx.index_dir))


@app.get("/health")
def health():
    manifest = _state.get("manifest") or {}
    return {"status": "ok" if _state else "loading",
            "process_id": os.getpid(), "owner_token": os.environ.get("PIXELRAG_OWNER_TOKEN"),
            "index_id": manifest.get("index_id"),
            "fingerprint": _state.get("fingerprint"),
            "encoder": manifest.get("encoder"),
            "dimensions": _state["index"].d if _state else None}


@app.post("/search")
def search(req: SearchRequest):
    if not _state:
        raise HTTPException(503, "Search index is not loaded.")
    index, meta = _state["index"], _state["meta"]
    try:
        queries = np.asarray([q.embedding for q in req.queries], dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(400, "Query embeddings must have equal dimensions.") from exc
    if queries.ndim != 2 or queries.shape[1] != index.d or not np.isfinite(queries).all():
        raise HTTPException(400, "Invalid query embedding dimensions or values.")
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise HTTPException(400, "A query vector must have a finite nonzero norm.")
    scores, ids = index.search((queries / norms).astype(np.float32), min(req.n_docs, index.ntotal))
    results = []
    for values, found in zip(scores, ids):
        results.append({"hits": [{"score": float(score), "vector_id": int(vid),
            "article_id": int(meta["article_ids"][vid]),
            "tile_index": int(meta["tile_indices"][vid]),
            "chunk_index": int(meta["chunk_indices"][vid])}
            for score, vid in zip(values, found) if vid >= 0]})
    return {"results": results}


@app.get("/tile/{article_id}/{tile_index}/{chunk_index}")
def tile(article_id: int, tile_index: int, chunk_index: int):
    if min(article_id, tile_index, chunk_index) < 0 or not _state:
        raise HTTPException(404, "No such region.")
    folder = _state["layout"].tile_dir(article_id)
    path = folder / f"chunk_{tile_index:04d}_{chunk_index:02d}.png"
    if not path.exists():
        raise HTTPException(404, "No such region.")
    return FileResponse(path, media_type="image/png")


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--port", type=int, default=30001)
    args = parser.parse_args()
    idx = layout.IndexLayout(layout.resolve_index_dir(layout._anchored(args.index_dir, layout.DEFAULT.index_dir))) if args.index_dir else None
    load(idx)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
