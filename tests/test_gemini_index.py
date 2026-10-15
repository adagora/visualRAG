"""Offline integration tests with real rendered PDFs and fake Gemini vectors."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pymupdf
import numpy as np
import pytest
import gemini_index as gi
import index_manifest


def vector(dimensions, position=0):
    result = np.zeros(dimensions, dtype=np.float32)
    result[position] = 1
    return result


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "zażółć.pdf"
    with pymupdf.open() as doc:
        for text in ("first", "second"):
            doc.new_page(width=120, height=160).insert_text((10, 30), text)
        doc.save(path)
    return path


@pytest.fixture
def built(tmp_path, pdf, monkeypatch):
    calls = []
    def fake(contents, model, dimensions):
        calls.append(contents)
        return vector(dimensions, (len(calls) - 1) % 2)
    monkeypatch.setattr(gi, "embed", fake)
    output = tmp_path / "index"
    report = gi.build([pdf], output, titles=["Żółć"])
    return output, Path(report["index_dir"]), calls


def test_real_render_build_and_immutable_sources(built, pdf):
    output, snapshot, calls = built
    index = gi.Index(output)
    assert index.snapshot_dir == snapshot
    assert index.documents[0]["title"] == "Żółć"
    assert index.documents[0]["pages"] == 2
    assert len(calls) == 2
    assert index.manifest["dimensions"] == 3072
    assert json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))["render"]["engine"] == "pymupdf"
    assert np.load(snapshot / "gemini.npy").shape == (2, 3072)
    from PIL import Image
    with Image.open(index.page_path(0, 1)) as image:
        assert image.format == "JPEG"
        assert image.width == 334  # 120 points at 200 DPI.
    original = pdf.read_bytes()
    pdf.write_bytes(b"changed external source")
    assert index.source_pdf(0).read_bytes() == original
    assert not (snapshot / "text.json").exists()
    assert not list((output / "snapshots").glob(".staging-*"))
    assert gi.Index(snapshot).documents == index.documents


@pytest.mark.parametrize("dimensions", [768, 1536])
def test_old_snapshot_schema_and_dimensions(tmp_path, pdf, monkeypatch, dimensions):
    monkeypatch.setattr(gi, "embed", lambda contents, model, dimensions: vector(dimensions))
    snapshot = Path(gi.build([pdf], tmp_path / "old", dimensions=dimensions)["index_dir"])
    meta = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    meta["documents"][0]["path"] = "C:/old/unavailable/original.pdf"
    meta["encoder"].update(revision=gi.MODEL, instruction=gi.QUERY_PREFIX, adapter=None)
    (snapshot / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    assert gi.Index(snapshot).manifest["dimensions"] == dimensions


def test_rank_and_one_embedding_for_each_question(built, monkeypatch):
    output, _, _ = built
    index = gi.Index(output)
    calls = []
    def fake(contents, model, dimensions):
        calls.append(contents)
        return vector(dimensions, 1)
    monkeypatch.setattr(gi, "embed", fake)
    for _ in range(2):
        hits = index.search("find second", limit=1)
        assert hits == [{"article_id": 0, "source_id": index.documents[0]["source_id"],
                         "document": "Żółć", "page": 2, "score": 1.0}]
    assert calls == [gi.QUERY_PREFIX + "find second"] * 2
    hits = index.search("plain query", image=b"reference", limit=2)
    assert len(calls) == 3
    assert calls[-1][0].inline_data.data == b"reference"
    assert calls[-1][1] == "plain query"
    assert len(hits) == 2
    index.search("", image=b"reference")
    assert len(calls[-1]) == 1


@pytest.mark.parametrize("article,page", [(0, 0), (0, -1), (0, 3), (-1, 1), (True, 1), (0, True), (0, 1.0)])
def test_page_bounds(built, article, page):
    with pytest.raises(ValueError):
        gi.Index(built[0]).page_path(article, page)


def test_failed_build_keeps_active(built, pdf, monkeypatch):
    output, old, _ = built
    pointer = (output / "active.json").read_bytes()
    def fail(*args):
        raise RuntimeError("simulated API failure")
    monkeypatch.setattr(gi, "embed", fail)
    with pytest.raises(RuntimeError, match="API failure"):
        gi.build([pdf], output)
    assert (output / "active.json").read_bytes() == pointer
    assert old.is_dir()
    assert not list((output / "snapshots").glob(".staging-*"))


@pytest.mark.parametrize("bad", [np.zeros(3072), np.ones(3), np.full(3072, np.nan)])
def test_invalid_build_vectors_never_publish(tmp_path, pdf, monkeypatch, bad):
    monkeypatch.setattr(gi, "embed", lambda *args: bad)
    output = tmp_path / "broken"
    with pytest.raises(ValueError):
        gi.build([pdf], output)
    assert not (output / "active.json").exists()
    assert not list((output / "snapshots").glob(".staging-*"))


@pytest.mark.parametrize("corruption", ["hash", "source_path", "pages", "dimensions", "rows"])
def test_corrupt_snapshot_rejected(built, corruption):
    output, snapshot, _ = built
    path = snapshot / "manifest.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "hash":
        (snapshot / meta["documents"][0]["snapshot_path"]).write_bytes(b"tampered")
    elif corruption == "source_path":
        meta["documents"][0]["snapshot_path"] = "../../original.pdf"
    elif corruption == "pages":
        meta["documents"][0]["pages"] = 3
    elif corruption == "dimensions":
        np.save(snapshot / "gemini.npy", np.ones((2, 768)))
    else:
        page_path = snapshot / "gemini.json"
        manifest = json.loads(page_path.read_text(encoding="utf-8"))
        manifest["pages"][1]["tile_index"] = 0
        page_path.write_text(json.dumps(manifest), encoding="utf-8")
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        gi.Index(output)


def test_source_revalidated_before_serving(built):
    index = gi.Index(built[0])
    source = index.source_pdf(0)
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        index.source_pdf(0)


def test_pointer_traversal(tmp_path):
    (tmp_path / "active.json").write_text(json.dumps({"schema_version": 1, "index_id": "../../escape", "path": "../../escape"}), encoding="utf-8")
    with pytest.raises(ValueError):
        index_manifest.resolve_snapshot(tmp_path)


def test_pointer_manifest_identity_must_match(built):
    output, snapshot, _ = built
    path = snapshot / "manifest.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["index_id"] = "another-snapshot"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        gi.Index(output)


def test_one_lazy_threadsafe_client(monkeypatch):
    from google import genai
    created = []
    def construct(**kwargs):
        assert kwargs["http_options"].timeout == 60_000
        assert kwargs["http_options"].retry_options.attempts == 1
        created.append(object())
        return created[-1]
    monkeypatch.setattr(gi, "_CLIENT", None)
    monkeypatch.setattr(gi, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(genai, "Client", construct)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: gi.client(), range(24)))
    assert len(created) == 1
    assert all(result is created[0] for result in results)


def test_embed_sdk_contract_and_normalization(monkeypatch):
    calls = []
    def request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=np.full(768, 2))])
    monkeypatch.setattr(gi, "client", lambda: SimpleNamespace(models=SimpleNamespace(embed_content=request)))
    result = gi.embed("query", gi.MODEL, 768)
    assert result.shape == (768,)
    assert np.linalg.norm(result) == pytest.approx(1)
    assert calls[0]["config"].output_dimensionality == 768


def test_import_has_no_heavy_dependencies():
    scripts = str(Path(gi.__file__).parent)
    code = ("import sys; sys.path.insert(0, " + repr(scripts) + "); import gemini_index, visual_query; "
            "assert not ({'torch', 'faiss', 'transformers', 'layout', 'corpus', 'lexical', 'google.genai', 'pixelrag_render'} & set(sys.modules))")
    subprocess.run([sys.executable, "-c", code], check=True)
