"""Page-image embeddings and exact cosine search for small Gemini corpora.

The model, dimensions and query format come from the index manifest. Never
fall back to Qwen vectors: those live in a different embedding space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np
from dotenv import load_dotenv

import layout

QUERY_PREFIX = "task: search result | query: "
MODEL = "gemini-embedding-2"
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def enabled() -> bool:
    backend = os.environ.get("PIXELRAG_EMBED_BACKEND", "auto").lower()
    if backend != "auto":
        return backend == "gemini"
    return (layout.DEFAULT.index_dir / "gemini.json").exists()


def client():
    # Query variants encode concurrently. lru_cache alone allows duplicate
    # first constructions; discarding one Client can close an in-flight request.
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from google import genai
            load_dotenv(layout.ROOT / ".env", override=False)
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Gemini embeddings require GEMINI_API_KEY or GOOGLE_API_KEY.")
            _CLIENT = genai.Client(api_key=key)
        return _CLIENT


def embed(contents, model: str, dimensions: int) -> np.ndarray:
    from google.genai import types
    owner = client()
    result = owner.models.embed_content(
        model=model, contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=dimensions))
    if not result.embeddings or len(result.embeddings) != 1:
        raise ValueError("Expected one Gemini embedding per page/query.")
    vector = np.asarray(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.shape != (dimensions,) or not np.isfinite(vector).all() or norm == 0:
        raise ValueError("Invalid Gemini embedding dimensions or values.")
    return vector / norm


@lru_cache(maxsize=4)
def _load(index_dir: str, stamp: tuple):
    root = Path(index_dir)
    manifest = json.loads((root / "gemini.json").read_text())
    if manifest.get("format") != 1 or not manifest.get("model", "").startswith("gemini-embedding-2"):
        raise ValueError("Unsupported Gemini index; rebuild with scripts/gemini_index.py.")
    vectors = np.load(root / "gemini.npy", allow_pickle=False)
    if vectors.shape != (len(manifest["pages"]), manifest["dimensions"]):
        raise ValueError("Gemini index manifest/vector mismatch; rebuild the index.")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Gemini index contains invalid or unnormalized vectors.")
    return manifest, vectors


def load(index: layout.IndexLayout | None = None):
    index = index or layout.DEFAULT
    paths = [index.index_dir / name for name in ("gemini.json", "gemini.npy")]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("Gemini index missing; run scripts/gemini_index.py first.")
    stamp = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    return _load(str(index.index_dir), stamp)


@lru_cache(maxsize=256)
def _query(text: str, model: str, dimensions: int, prefix: str) -> tuple:
    return tuple(embed(prefix + text, model, dimensions).tolist())


def embed_query(text: str, *, image: bytes | None = None) -> list[float]:
    manifest, _ = load()
    if image is not None:
        from google.genai import types
        contents = [types.Part.from_bytes(data=image, mime_type="image/png")]
        if text.strip():
            # Gemini's task prefix is for text-only retrieval. Its multimodal
            # guidance recommends plain text alongside an image.
            contents.append(text)
        return embed(contents, manifest["model"], manifest["dimensions"]).tolist()
    return list(_query(text, manifest["model"], manifest["dimensions"], manifest["query_prefix"]))


def search(query: str, n_results: int, *, image: bytes | None = None) -> list[dict]:
    manifest, vectors = load()
    q = np.asarray(embed_query(query, image=image), dtype=np.float32) if image is not None else np.asarray(embed_query(query), dtype=np.float32)
    scores = vectors @ q
    order = np.argsort(-scores, kind="stable")[:max(1, n_results)]
    return [{"article_id": manifest["pages"][i]["article_id"],
             "tile_index": manifest["pages"][i]["tile_index"],
             "chunk_index": 0, "score": float(scores[i])} for i in order]


def build(pdfs: list[Path], output: Path, *, model: str = MODEL,
          dimensions: int = 768, titles: list[str] | None = None) -> dict:
    """Render and embed each page independently; publish only a complete build."""
    from google.genai import types
    from pixelrag_render.render import render_pdf
    import index_manifest
    if not pdfs or dimensions not in (768, 1536, 3072):
        raise ValueError("Provide PDFs and dimensions 768, 1536 or 3072.")
    if titles is not None and len(titles) != len(pdfs):
        raise ValueError("Provide one title per PDF.")
    output = output.resolve()
    stage = index_manifest.begin_snapshot(output)
    idx = layout.IndexLayout(stage)
    articles, pages, vectors, documents = [], [], [], []
    try:
        for aid, source in enumerate(pdfs):
            source = source.resolve()
            title = titles[aid] if titles else source.stem
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            saved = stage / "sources" / (digest + ".pdf")
            saved.parent.mkdir(exist_ok=True)
            shutil.copy2(source, saved)
            articles.append({"title": title, "url": "sources/" + saved.name,
                             "document_type": "document",
                             "sha256": digest, "source_id": "doc-" + digest[:24]})
            # Upstream's renderer writes the page JPEGs and a one-chunk-per-page
            # chunks.json; only the embedding is ours.
            render_pdf(source, idx.tiles_dir, stem=str(aid))
            images = idx.page_images(aid)
            documents.append({"article_id": aid, "source_id": "doc-" + digest[:24],
                              "sha256": digest, "path": str(source), "snapshot_path": "sources/" + saved.name,
                              "title": title, "pages": len(images)})
            for ti, image in enumerate(images):
                # Multiple images in a request aggregate into ONE vector.
                # One request/page preserves citation-level retrieval.
                vector = embed(types.Part.from_bytes(data=image.read_bytes(), mime_type="image/jpeg"),
                               model, dimensions)
                vectors.append(vector)
                pages.append({"article_id": aid, "tile_index": ti})
                print(f"{title}: page {ti + 1}/{len(images)}", flush=True)
        if not vectors:
            raise ValueError("PDFs contain no pages.")
        matrix = np.stack(vectors)
        if matrix.shape != (len(pages), dimensions) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError("Gemini build contains invalid dimensions or unnormalized vectors.")
        idx.articles_json.write_text(json.dumps(articles, ensure_ascii=False))
        import lexical
        lexical.build_sidecar(articles, out_path=idx.text_sidecar)
        manifest = {"format": 1, "model": model, "dimensions": dimensions,
                    "query_prefix": QUERY_PREFIX, "input": "page-image", "pages": pages}
        (stage / "gemini.json").write_text(json.dumps(manifest))
        np.save(stage / "gemini.npy", matrix, allow_pickle=False)
        idx.summary_json.write_text(json.dumps({"backend": "gemini", "pages": len(pages), "model": model}))
        snapshot = {"schema_version": 1, "index_id": stage.name.removeprefix(".staging-"),
                    "encoder": {"backend": "gemini", "model": model, "revision": model,
                                "dimensions": dimensions, "adapter": None, "instruction": QUERY_PREFIX},
                    "render": {"dpi": 200, "jpeg_quality": 85},
                    "chunking": {"mode": "page"}, "documents": documents,
                    "vector_count": len(vectors)}
        final = index_manifest.publish_snapshot(output, stage, snapshot)
        manifest["index_dir"] = str(final)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdfs", type=Path, nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dimensions", type=int, default=768, choices=[768, 1536, 3072])
    ap.add_argument("--title", action="append",
                    help="document title; repeat once per PDF (default: file name)")
    args = ap.parse_args()
    build(args.pdfs, args.output, model=args.model, dimensions=args.dimensions,
          titles=args.title)


if __name__ == "__main__":
    main()
