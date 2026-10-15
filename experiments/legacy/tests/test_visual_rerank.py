import io
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from pagehit import PageHit
import visual_rerank as vr


def png():
    out = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(out, format="PNG")
    return out.getvalue()


def test_disabled_never_loads_model(monkeypatch):
    monkeypatch.delenv("PIXELRAG_VISUAL_RERANK", raising=False)
    monkeypatch.setattr(vr, "load_model", lambda *a: pytest.fail("model loading forbidden"))
    with pytest.raises(RuntimeError, match="disabled"):
        vr.rerank("question", [])


def test_missing_local_model_never_downloads(monkeypatch):
    monkeypatch.delenv("PIXELRAG_RERANK_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="automatic downloads are disabled"):
        vr.model_path()


def test_lazy_loader_forces_offline_model_files(monkeypatch, tmp_path):
    calls = []
    fake = object()
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return fake
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=constructor))
    vr.load_model.cache_clear()
    try:
        assert vr.load_model(tmp_path, "cuda") is fake
        assert calls[0][1]["local_files_only"] is True
        assert calls[0][1]["trust_remote_code"] is False
        assert calls[0][1]["max_length"] == 4096
    finally:
        vr.load_model.cache_clear()


def test_bounded_image_aware_scoring_preserves_evidence_and_tail(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELRAG_VISUAL_RERANK", "1")
    image = tmp_path / "page.png"
    image.write_bytes(png())
    pages = [PageHit(0, i, score=.5, rank=i+1) for i in range(15)]
    calls = []
    def scorer(pairs, **kwargs):
        calls.append((pairs, kwargs))
        return list(range(12))
    ranked, stats = vr.rerank_with_stats("question", pages, png(), scorer=scorer, page_path=lambda *a: image)
    assert [p.tile_index for p in ranked[:12]] == list(reversed(range(12)))
    assert ranked[12:] == pages[12:]
    assert all(p.score == .5 for p in ranked)
    assert pages[0].rank == 1
    assert stats["scored_pages"] == 12 and stats["query_image"]
    pairs, kwargs = calls[0]
    assert len(pairs) == 12 and pairs[0][0]["image"].startswith("data:image/png")
    assert pairs[0][1]["image"].startswith("data:image/png")
    assert kwargs["batch_size"] == 1 and kwargs["prompt"] == vr.INSTRUCTION


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")]])
def test_invalid_scores_are_visible_failures(monkeypatch, tmp_path, scores):
    monkeypatch.setenv("PIXELRAG_VISUAL_RERANK", "1")
    path = tmp_path / "page.png"
    path.write_bytes(png())
    with pytest.raises(ValueError, match="incomplete or nonfinite"):
        vr.rerank_with_stats("q", [PageHit(0, 0)], scorer=lambda *a, **k: scores, page_path=lambda *a: path)
