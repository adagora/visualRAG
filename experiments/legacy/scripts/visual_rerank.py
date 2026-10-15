"""Optional bounded multimodal reranking, following the official Qwen model card.

https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B
Uses its documented Sentence Transformers CrossEncoder multimodal pair API.
Requires installed optional sentence-transformers and a predownloaded local model
directory. Importing this module never loads a model or downloads weights.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

from pagehit import PageHit

CANDIDATE_LIMIT = 12
IMAGE_LONG_EDGE = 1024
INSTRUCTION = "Retrieve images or text relevant to the user's query."
_LOCK = threading.Lock()


def enabled():
    return os.environ.get("PIXELRAG_VISUAL_RERANK") == "1"


def model_path():
    value = os.environ.get("PIXELRAG_RERANK_MODEL", "")
    path = Path(value)
    if not value or not path.is_dir() or not (path / "config.json").is_file():
        raise RuntimeError("Set PIXELRAG_RERANK_MODEL to a local Qwen3-VL-Reranker-2B model directory; automatic downloads are disabled")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Expected Qwen3-VL reranker configuration")
    return path.resolve()


@lru_cache(maxsize=1)
def load_model(path, device):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(str(path), device=device, trust_remote_code=False,
                        local_files_only=True, max_length=4096, model_kwargs={"torch_dtype": "auto"})


def image_url(source):
    """Bound candidate/query image pixels before the model sees them."""
    from PIL import Image, ImageOps
    if isinstance(source, bytes) and len(source) > 5 * 1024 * 1024:
        raise ValueError("Reranker query image exceeds 5 MB")
    with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Reranker image exceeds 16 million pixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def rerank_with_stats(question: str, pages: list[PageHit], query_image: bytes | None = None,
                      *, scorer=None, page_path=None) -> tuple[list[PageHit], dict]:
    if not enabled():
        raise RuntimeError("Visual reranker experiment is disabled; set PIXELRAG_VISUAL_RERANK=1")
    started = time.perf_counter()
    candidates = list(pages[:CANDIDATE_LIMIT])
    if len({p.key for p in candidates}) != len(candidates):
        raise ValueError("Reranker candidate pages must be unique")
    if page_path is None:
        import corpus
        page_path = corpus.page_path
    query = {"text": question}
    if query_image is not None:
        query["image"] = image_url(query_image)
    pairs = [(query, {"image": image_url(page_path(p.article_id, p.tile_index))}) for p in candidates]
    if not candidates:
        return [], {"candidate_limit": CANDIDATE_LIMIT, "scored_pages": 0, "scores": [], "latency_ms": 0.0}
    with _LOCK:
        if scorer is None:
            scorer = load_model(model_path(), os.environ.get("PIXELRAG_RERANK_DEVICE", "cuda")).predict
        raw = scorer(pairs, prompt=INSTRUCTION, batch_size=1, show_progress_bar=False)
    scores = [float(s) for s in raw]
    if len(scores) != len(candidates) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Reranker returned incomplete or nonfinite scores")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    # Preserve retrieval evidence and scores; reranker scores are separate provenance.
    ranked = [candidates[i].with_(rank=rank + 1) for rank, i in enumerate(order)] + list(pages[CANDIDATE_LIMIT:])
    stats = {"model": "Qwen/Qwen3-VL-Reranker-2B", "model_path": os.environ.get("PIXELRAG_RERANK_MODEL"),
             "candidate_limit": CANDIDATE_LIMIT, "scored_pages": len(scores), "query_image": query_image is not None,
             "batch_size": 1, "image_long_edge": IMAGE_LONG_EDGE, "instruction": INSTRUCTION,
             "scores": [{"article_id": candidates[i].article_id, "page": candidates[i].page, "score": scores[i]} for i in order],
             "latency_ms": (time.perf_counter() - started) * 1000}
    return ranked, stats


def rerank(question: str, pages: list[PageHit], query_image: bytes | None = None) -> list[PageHit]:
    return rerank_with_stats(question, pages, query_image)[0]
