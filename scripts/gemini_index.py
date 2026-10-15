"""Immutable PDF page images, Gemini embeddings, and NumPy search."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

import numpy as np
from dotenv import load_dotenv
import index_manifest

QUERY_PREFIX = "task: search result | query: "
MODEL = "gemini-embedding-2"
DIMENSIONS = (768, 1536, 3072)
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def client():
    """One shared SDK client; environment variables take precedence over .env."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            load_dotenv(Path.cwd() / ".env", override=False)
            load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or .env.")
            from google import genai
            from google.genai import types
            _CLIENT = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
        return _CLIENT


def embed(contents, model: str, dimensions: int) -> np.ndarray:
    from google.genai import types
    owner = client()
    result = owner.models.embed_content(model=model, contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=dimensions))
    if not result.embeddings or len(result.embeddings) != 1:
        raise ValueError("Expected one Gemini embedding per page/query.")
    vector = np.asarray(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.shape != (dimensions,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid Gemini embedding dimensions or values.")
    return vector / norm


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_vectors(vectors, count, dimensions):
    if vectors.shape != (count, dimensions):
        raise ValueError("Gemini manifest/vector dimensions mismatch.")
    if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid or unnormalized Gemini vectors.")


class Index:
    """Load an explicit snapshot or resolve a collection's active pointer once."""
    def __init__(self, path: Path | str):
        self.snapshot_dir = index_manifest.resolve_snapshot(Path(path))
        root = self.snapshot_dir
        self.manifest = _json(root / "gemini.json")
        meta = index_manifest.load(root)
        if meta is None:
            raise ValueError("Missing manifest.json; rebuild the index.")
        dimensions = self.manifest.get("dimensions")
        if self.manifest.get("format") != 1 or self.manifest.get("model") != MODEL or dimensions not in DIMENSIONS:
            raise ValueError("Unsupported Gemini model or dimensions.")
        self._pages = self.manifest["pages"]
        self._vectors = np.load(root / "gemini.npy", allow_pickle=False)
        _validate_vectors(self._vectors, len(self._pages), dimensions)
        if not self._pages or meta.get("vector_count") != len(self._pages):
            raise ValueError("Index page count mismatch.")
        encoder = meta.get("encoder", {})
        if encoder.get("model") != MODEL or encoder.get("dimensions") != dimensions:
            raise ValueError("Index encoder metadata mismatch.")
        articles = _json(root / "articles.json")
        self._sources, self._hashes, self._documents = {}, {}, {}
        self.documents = []
        for doc in meta.get("documents", []):
            aid, count, digest = doc["article_id"], doc["pages"], doc["sha256"]
            if type(aid) is not int or aid < 0 or aid in self._documents or type(count) is not int or count < 1:
                raise ValueError("Invalid document/page identity.")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid source SHA256.")
            if doc["source_id"] != "doc-" + digest[:24]:
                raise ValueError("Source identity mismatch.")
            if aid >= len(articles) or articles[aid].get("sha256") != digest or articles[aid].get("source_id") != doc["source_id"]:
                raise ValueError("Article/source identity mismatch.")
            source = (root / doc["snapshot_path"]).resolve()
            if not source.is_relative_to(root) or source.parent != (root / "sources").resolve() or source.suffix.lower() != ".pdf":
                raise ValueError("Source must be a PDF inside snapshot sources/.")
            self._sources[aid], self._hashes[aid] = source, digest
            row = {"article_id": aid, "source_id": doc["source_id"], "title": doc["title"], "pages": count}
            self.documents.append(row)
            self._documents[aid] = row
            self.source_pdf(aid)
        if not self.documents or len(articles) != len(self.documents):
            raise ValueError("Index document count mismatch.")
        expected = {(d["article_id"], p) for d in self.documents for p in range(d["pages"])}
        actual = []
        for row in self._pages:
            aid, page = row["article_id"], row["tile_index"]
            if type(aid) is not int or type(page) is not int:
                raise ValueError("Invalid indexed page identity.")
            actual.append((aid, page))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Indexed pages do not match source page counts.")
        for aid, page in actual:
            self.page_path(aid, page + 1)
        self._vectors.flags.writeable = False

    def _document(self, article_id):
        if type(article_id) is not int or article_id not in self._documents:
            raise ValueError("Unknown document.")
        return self._documents[article_id]

    def page_path(self, article_id: int, page: int) -> Path:
        doc = self._document(article_id)
        if type(page) is not int or not 1 <= page <= doc["pages"]:
            raise ValueError("Page is outside the document.")
        path = self.snapshot_dir / "tiles" / f"{article_id}.png.tiles" / f"tile_{page - 1:04d}.jpg"
        if not path.is_file() or not path.resolve().is_relative_to(self.snapshot_dir):
            raise ValueError("Missing or invalid indexed page image.")
        return path

    def source_pdf(self, article_id: int) -> Path:
        doc = self._document(article_id)
        path = self._sources[article_id]
        if not path.resolve().is_relative_to(self.snapshot_dir):
            raise ValueError("Source PDF escapes its snapshot.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self._hashes[article_id]:
            raise ValueError("Source PDF hash mismatch.")
        import pymupdf
        with pymupdf.open(path) as pdf:
            if len(pdf) != doc["pages"]:
                raise ValueError("Source PDF page count mismatch.")
        return path

    def search(self, question: str, *, image: bytes | None = None, limit: int = 4) -> list[dict]:
        if type(limit) is not int or limit < 1:
            raise ValueError("Search limit must be a positive integer.")
        if not isinstance(question, str) or (not question.strip() and image is None):
            raise ValueError("Provide a question or reference image.")
        if image is None:
            contents = self.manifest.get("query_prefix", QUERY_PREFIX) + question
        else:
            from google.genai import types
            contents = [types.Part.from_bytes(data=image, mime_type="image/png")]
            if question.strip():
                contents.append(question)
        query = embed(contents, self.manifest["model"], self.manifest["dimensions"])
        _validate_vectors(np.asarray(query)[None, :], 1, self.manifest["dimensions"])
        scores = self._vectors @ query
        order = np.argsort(-scores, kind="stable")[:limit]
        return [{"article_id": self._pages[i]["article_id"],
                 "source_id": self._documents[self._pages[i]["article_id"]]["source_id"],
                 "page": self._pages[i]["tile_index"] + 1,
                 "document": self._documents[self._pages[i]["article_id"]]["title"],
                 "score": float(scores[i])} for i in order]


def build(pdfs: list[Path], output: Path, *, model: str = MODEL,
          dimensions: int = 3072, titles: list[str] | None = None) -> dict:
    """Render and embed one page per request, then atomically publish."""
    import pymupdf
    from google.genai import types
    if not pdfs or dimensions not in DIMENSIONS or model != MODEL:
        raise ValueError("Provide PDFs, gemini-embedding-2, and dimensions 768, 1536 or 3072.")
    if titles is not None and len(titles) != len(pdfs):
        raise ValueError("Provide one title per PDF.")
    output = Path(output).resolve()
    stage = index_manifest.begin_snapshot(output)
    articles, pages, vectors, documents = [], [], [], []
    try:
        for aid, source in enumerate(pdfs):
            source = Path(source).resolve()
            title = titles[aid] if titles else source.stem
            raw = source.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            sid = "doc-" + digest[:24]
            saved = stage / "sources" / (digest + ".pdf")
            saved.parent.mkdir(exist_ok=True)
            saved.write_bytes(raw)
            tiles = stage / "tiles" / f"{aid}.png.tiles"
            tiles.mkdir(parents=True)
            with pymupdf.open(saved) as pdf:
                if not len(pdf) or pdf.needs_pass:
                    raise ValueError("Provide a nonempty, unencrypted PDF.")
                count = len(pdf)
                for number, page in enumerate(pdf):
                    image = tiles / f"tile_{number:04d}.jpg"
                    page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False).save(image, jpg_quality=85)
                    vectors.append(embed(types.Part.from_bytes(data=image.read_bytes(), mime_type="image/jpeg"), model, dimensions))
                    pages.append({"article_id": aid, "tile_index": number})
                    print(f"{title}: page {number + 1}/{count}", flush=True)
            articles.append({"title": title, "url": "sources/" + saved.name,
                             "document_type": "document", "sha256": digest, "source_id": sid})
            documents.append({"article_id": aid, "source_id": sid, "sha256": digest,
                              "snapshot_path": "sources/" + saved.name, "title": title, "pages": count})
        matrix = np.stack(vectors)
        _validate_vectors(matrix, len(pages), dimensions)
        manifest = {"format": 1, "model": model, "dimensions": dimensions,
                    "query_prefix": QUERY_PREFIX, "input": "page-image", "pages": pages}
        (stage / "articles.json").write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
        (stage / "gemini.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.save(stage / "gemini.npy", matrix, allow_pickle=False)
        snapshot = {"schema_version": 1, "index_id": stage.name.removeprefix(".staging-"),
                    "encoder": {"backend": "gemini", "model": model, "dimensions": dimensions},
                    "render": {"engine": "pymupdf", "dpi": 200, "jpeg_quality": 85}, "chunking": {"mode": "page"},
                    "documents": documents, "vector_count": len(pages)}
        (stage / "manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        Index(stage)
        final = index_manifest.publish_snapshot(output, stage, snapshot)
        return {**manifest, "index_dir": str(final)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=3072, choices=DIMENSIONS)
    parser.add_argument("--title", action="append")
    args = parser.parse_args()
    print(json.dumps(build(args.pdfs, args.output, dimensions=args.dimensions, titles=args.title)))


if __name__ == "__main__":
    main()
