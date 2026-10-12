"""The Gemini page-embedding index: what a build publishes and how search reads it.

No network and no poppler: the embedding call and the upstream renderer are
replaced, so these pin the part that is ours — the file layout, the refusal to
publish a partial build, and the ranking.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("google.genai")
render_module = pytest.importorskip("pixelrag_render.render")

import gemini_index  # noqa: E402
import layout  # noqa: E402

DIMS = 768


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def source_pdf(tmp_path):
    import fitz

    path = tmp_path / "paper.pdf"
    with fitz.open() as doc:
        for text in ("scaled dot-product attention", "multi-head attention"):
            doc.new_page().insert_text((72, 72), text)
        doc.save(str(path))
    return path


@pytest.fixture
def fakes(monkeypatch):
    """Upstream's renderer writes the page JPEGs; the model returns basis vectors."""
    calls = []

    def render_pdf(path, output_dir, *, stem, **_):
        tiles = output_dir / f"{stem}.png.tiles"
        tiles.mkdir(parents=True)
        for i in range(2):
            (tiles / f"tile_{i:04d}.jpg").write_bytes(b"\xff\xd8 page %d" % i)
        return [tiles]

    def embed(contents, model, dimensions):
        calls.append((model, dimensions))
        return basis(len(calls) - 1)

    monkeypatch.setattr(render_module, "render_pdf", render_pdf)
    monkeypatch.setattr(gemini_index, "embed", embed)
    return calls


def test_build_publishes_a_consistent_index(tmp_path, source_pdf, fakes):
    out = tmp_path / "index"
    manifest = gemini_index.build([source_pdf], out, titles=["The Paper"])

    vectors = np.load(out / "gemini.npy", allow_pickle=False)
    assert vectors.shape == (2, DIMS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert manifest["pages"] == [{"article_id": 0, "tile_index": 0},
                                 {"article_id": 0, "tile_index": 1}]
    assert fakes == [(gemini_index.MODEL, DIMS)] * 2  # one request per page
    assert json.loads((out / "articles.json").read_text())[0]["title"] == "The Paper"
    assert (out / "text.json").exists()
    loaded, _ = gemini_index.load(layout.IndexLayout(out))
    assert loaded["model"] == gemini_index.MODEL
    assert not list(tmp_path.glob(".gemini-build-*"))


def test_build_refuses_to_overwrite(tmp_path, source_pdf, fakes):
    out = tmp_path / "index"
    out.mkdir()
    with pytest.raises(FileExistsError):
        gemini_index.build([source_pdf], out)
    assert fakes == []  # nothing was embedded, so nothing was paid for


def test_a_failed_build_publishes_nothing(tmp_path, source_pdf, fakes, monkeypatch):
    def fail(*_):
        raise RuntimeError("429")

    monkeypatch.setattr(gemini_index, "embed", fail)
    out = tmp_path / "index"
    with pytest.raises(RuntimeError):
        gemini_index.build([source_pdf], out)
    assert not out.exists()
    assert not list(tmp_path.glob(".gemini-build-*"))


def test_search_ranks_by_cosine(monkeypatch):
    manifest = {"pages": [{"article_id": 0, "tile_index": i} for i in range(3)]}
    vectors = np.stack([basis(0), basis(1), basis(2)])
    monkeypatch.setattr(gemini_index, "load", lambda index=None: (manifest, vectors))
    monkeypatch.setattr(gemini_index, "embed_query", lambda text: list(basis(1)))

    hits = gemini_index.search("anything", 2)

    assert [h["tile_index"] for h in hits][0] == 1
    assert len(hits) == 2
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["chunk_index"] == 0


def test_load_rejects_vectors_that_do_not_match_the_manifest(tmp_path):
    (tmp_path / "gemini.json").write_text(json.dumps({
        "format": 1, "model": "gemini-embedding-2", "dimensions": DIMS,
        "pages": [{"article_id": 0, "tile_index": i} for i in range(3)]}))
    np.save(tmp_path / "gemini.npy", np.stack([basis(0), basis(1)]), allow_pickle=False)
    with pytest.raises(ValueError, match="mismatch"):
        gemini_index.load(layout.IndexLayout(tmp_path))


def test_backend_is_opt_in(monkeypatch):
    monkeypatch.delenv("PIXELRAG_EMBED_BACKEND", raising=False)
    assert not gemini_index.enabled()
    monkeypatch.setenv("PIXELRAG_EMBED_BACKEND", "gemini")
    assert gemini_index.enabled()
