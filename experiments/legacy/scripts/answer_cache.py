#!/usr/bin/env python
"""Cache answers by exact normalized question and pipeline namespace.

Vector similarity is available only through the explicit fuzzy experiment.
Normalization changes case and whitespace while preserving product names,
numerals, punctuation and negation. Namespace identity must cover source data,
reader prompts and pipeline options. Sequence numbers provide deterministic LRU
ordering even when timestamps tie.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

import layout

log = logging.getLogger("pixelrag.answer_cache")

ENABLED = os.environ.get("PIXELRAG_ANSWER_CACHE", "1") != "0"
THRESHOLD = float(os.environ.get("PIXELRAG_CACHE_THRESHOLD", "0.95"))
MAX_ENTRIES = int(os.environ.get("PIXELRAG_CACHE_MAX", "2000"))


def normalize_question(question: str) -> str:
    """Normalize presentation only; preserve products, numerals and negation."""
    return " ".join(unicodedata.normalize("NFC", question).casefold().split())

# Beside the index by default, because an entry is only valid for the index it
# was read out of (see index_fingerprint). PIXELRAG_CACHE_DIR moves it; a
# relative value is anchored to the repo root, not to the working directory.
_CACHE_LAYOUT = layout.IndexLayout(
    layout._anchored(os.environ.get("PIXELRAG_CACHE_DIR", ""),
                     layout.DEFAULT.index_dir))
CACHE_DIR = _CACHE_LAYOUT.index_dir
VEC_PATH = _CACHE_LAYOUT.answer_cache_vectors
META_PATH = _CACHE_LAYOUT.answer_cache_meta


def index_fingerprint(index_dir: Path) -> str:
    """Identity of the index the cached answers were read out of.

    Size and mtime rather than a content hash: index.faiss is hundreds of MB
    and this runs on every question. A rebuild always moves both.
    """
    parts = []
    for name in ("index.faiss", "articles.json", "summary.json", "gemini.json", "gemini.npy", "manifest.json", "text.json", "pages.json", "lexical.json"):
        p = Path(index_dir) / name
        try:
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{name}:-")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class AnswerCache:
    """Vectors in one array, payloads in a parallel list, both rewritten on put.

    Not a database on purpose. A few hundred entries of a few KB each is small
    enough that the simplest thing that survives a restart is the right thing,
    and a corrupt cache is recoverable by deleting two files.
    """

    def __init__(self, vec_path: Path = VEC_PATH, meta_path: Path = META_PATH):
        self.vec_path = Path(vec_path)
        self.meta_path = Path(meta_path)
        self._lock = threading.Lock()
        self._vecs: np.ndarray | None = None
        self._meta: list[dict] = []
        self.hits = 0
        self.misses = 0
        self._load()
        self._sequence = max((m.get("sequence", 0) for m in self._meta), default=0)

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            vecs = np.load(self.vec_path)["v"].astype(np.float32)
        except FileNotFoundError:
            return                      # no cache yet, which is the normal case
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("answer cache at %s is unreadable (%s) — starting "
                        "empty; delete both files to silence this",
                        self.meta_path.parent, e)
            return
        # A half-written pair is worse than none: drop both rather than pair a
        # question with someone else's vector.
        if isinstance(meta, list) and len(meta) == len(vecs):
            self._meta, self._vecs = meta, vecs
        else:
            log.warning("answer cache is inconsistent (%d entries, %d vectors)"
                        " — discarding it rather than pairing a question with "
                        "another question's vector", len(meta), len(vecs))

    def _save(self) -> None:
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            if self._vecs is None or not len(self._meta):
                self.meta_path.unlink(missing_ok=True)
                self.vec_path.unlink(missing_ok=True)
                return
            # Vectors first: a crash between the two leaves extra vectors, which
            # the length check discards. The reverse would orphan answers.
            np.savez_compressed(self.vec_path, v=self._vecs)
            self.meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("could not persist the answer cache to %s (%s) — it "
                        "will still work in memory for this process",
                        self.meta_path.parent, e)

    # -- api ---------------------------------------------------------------

    def get(self, embedding, namespace: str, question: str | None = None,
            *, fuzzy: bool = False) -> dict | None:
        """Exact normalized question by default; fuzzy lookup is experimental."""
        if not ENABLED:
            return None
        with self._lock:
            if self._vecs is None or not len(self._meta):
                self.misses += 1
                return None
            if not fuzzy:
                matches = [i for i, m in enumerate(self._meta)
                           if question is not None and m.get("namespace") == namespace
                           and normalize_question(m.get("question", "")) == normalize_question(question)]
                if not matches:
                    self.misses += 1
                    return None
                i = matches[-1]
                similarity = 1.0
            else:
                return self._fuzzy_get(embedding, namespace)
            return self._hit(i, similarity)

    def _hit(self, i: int, similarity: float) -> dict:
        self.hits += 1
        entry = self._meta[i]
        entry["used"] = entry.get("used", 0) + 1
        entry["last_used"] = time.time()
        self._sequence += 1
        entry["sequence"] = self._sequence
        return {"similarity": similarity, "question": entry["question"], "result": entry["result"]}

    def _fuzzy_get(self, embedding, namespace):
        # Caller holds the lock; this path is never enabled implicitly.
        q = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if not n or not np.isfinite(q).all() or q.shape != (self._vecs.shape[1],):
            self.misses += 1
            return None
        q /= n

        sims = self._vecs @ q
        # Namespace is exact-match, so a near-miss in the wrong namespace
        # must not win. Masking beats filtering: the array stays contiguous.
        mask = np.array([m.get("namespace") == namespace for m in self._meta])
        if not mask.any():
            self.misses += 1
            return None
        sims = np.where(mask, sims, -np.inf)

        i = int(np.argmax(sims))
        if sims[i] < THRESHOLD:
            self.misses += 1
            return None

        return self._hit(i, float(sims[i]))

    def put(self, embedding, question: str, namespace: str, result: dict) -> None:
        if not ENABLED:
            return
        q = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if not n or q.ndim != 1 or not np.isfinite(q).all():
            return
        q = (q / n).reshape(1, -1)

        with self._lock:
            self._sequence += 1
            if self._vecs is None or not len(self._meta):
                self._vecs = q
                self._meta = []
            elif self._vecs.shape[1] != q.shape[1]:
                # A different embedding model was configured. Its vectors are
                # not comparable to these, and silently mixing spaces would
                # make every similarity meaningless.
                log.warning("embedding width changed (%d -> %d): the answer "
                            "cache has been reset, since vectors from two "
                            "models cannot be compared",
                            self._vecs.shape[1], q.shape[1])
                self._vecs, self._meta = q, []
            else:
                self._vecs = np.vstack([self._vecs, q])
            self._meta.append({
                "question": question,
                "namespace": namespace,
                "result": result,
                "created": time.time(),
                "used": 0,
                "sequence": self._sequence,
            })
            if len(self._meta) > MAX_ENTRIES:
                # Evict least-recently-useful: never-used entries by age.
                keep = sorted(
                    range(len(self._meta)),
                    key=lambda i: (self._meta[i].get("sequence", 0), i),
                    reverse=True,
                )[:MAX_ENTRIES]
                keep.sort()
                self._vecs = self._vecs[keep]
                self._meta = [self._meta[i] for i in keep]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._vecs, self._meta = None, []
            self._save()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._meta),
                "hits": self.hits,
                "misses": self.misses,
                "threshold": THRESHOLD,
                "enabled": ENABLED,
            }


_default: AnswerCache | None = None


def default() -> AnswerCache:
    global _default
    if _default is None:
        _default = AnswerCache()
    return _default


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or clear the answer cache.")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    c = default()
    if args.clear:
        c.clear()
        print("cleared")
        return
    print(json.dumps(c.stats(), indent=2))
    for m in c._meta:
        print(f"  [{m.get('used', 0):3d} uses] {m['question'][:70]}")


if __name__ == "__main__":
    _main()
