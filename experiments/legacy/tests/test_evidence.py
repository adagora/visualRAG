"""Evidence identity and geometry are independent of a PDF text layer."""
import io
import json

import pytest
from PIL import Image

import citeparse
import corpus
import citations
import evidence
import prompts


@pytest.fixture
def pages(monkeypatch):
    monkeypatch.setattr(corpus, "source_pdf", lambda _: None)
    return [{"source_id": "a", "article_id": 0, "document": "A", "page": 7},
            {"source_id": "b", "article_id": 1, "document": "B", "page": 7}]


def test_visual_evidence_on_scan_needs_no_quote_or_text_layer(pages):
    text = 'Answer\n---CYTATY---\n' + json.dumps({
        "source_id": "b", "page": 7, "rects": [{"x": 10, "y": 20, "w": 30, "h": 40}],
        "claim": "diagram dimensions"})
    body, cites = citeparse.split_citations(text)
    out = citeparse.resolve_citations(body, cites, pages)
    assert out[0]["document"] == "B" and out[0]["source_id"] == "b"
    assert out[0]["rects"] == [{"left": 10, "top": 20, "width": 30, "height": 40}]
    assert out[0]["status"] == "model-region" and not out[0]["text_layer"]
    assert out[0]["verified"] is False


def test_page_identity_never_falls_back_from_unknown_source(pages):
    out = citeparse.resolve_citations("", [{"source_id": "unknown", "page": 7,
                                           "quote": "a label"}], pages)
    assert out[0]["status"] == "unsupported" and "document" not in out[0]


def test_legacy_single_document_remains_supported(pages):
    out = citeparse.resolve_citations("", [{"page": 7, "quote": "a label"}], pages[:1])
    assert out[0]["source_id"] == "a" and out[0]["status"] == "unlocated"


def test_empty_visual_evidence_remains_visible_as_unlocated(pages):
    body, cites = citeparse.split_citations('A\n---CYTATY---\n'
        '{"source_id":"a","page":7,"rects":[],"supports":"a claim"}')
    out = citeparse.resolve_citations(body, cites, pages)
    assert out[0]["status"] == "unlocated" and out[0]["rects"] == []


def test_malformed_source_identifier_is_unsupported_not_an_exception(pages):
    out = citeparse.resolve_citations("", [{"source_id": [], "page": 7,
                                           "quote": "a label"}], pages)
    assert out[0]["status"] == "unsupported"


def test_text_location_is_not_fact_verification(monkeypatch, pages):
    monkeypatch.setattr(corpus, "source_pdf", lambda _: "dummy.pdf")
    monkeypatch.setattr(citations, "locate_quote", lambda *a: [{"rects": [
        {"left": 1, "top": 2, "width": 3, "height": 4}], "coverage": 1}])
    monkeypatch.setattr(citations, "_page_words", lambda *a: ("word",))
    monkeypatch.setattr(citations, "locate_numbers", lambda *a: ([], []))
    out = citeparse.resolve_citations("wrong conclusion", [{"source_id": "a", "page": 7,
                    "quote": "existing label", "supports": "wrong conclusion"}], pages)
    assert out[0]["status"] == "text-located" and out[0]["verified"] is False


@pytest.mark.parametrize("rect", [
    {"x": -1, "y": 0, "w": 5, "h": 5},
    {"x": 99, "y": 0, "w": 5, "h": 5},
    {"x": 0, "y": 0, "w": 0, "h": 5},
    {"x": 0, "y": 0, "w": float("nan"), "h": 5},
    {"x": 0, "y": 0, "w": float("inf"), "h": 5},
    {"x": True, "y": 0, "w": 5, "h": 5},
    {"x": 0},
])
def test_invalid_rectangles_are_flagged_without_highlight(rect, pages):
    assert evidence.validate_rect(rect) is None
    out = citeparse.resolve_citations("", [{"source_id": "a", "page": 7,
                                           "rects": [rect]}], pages)
    assert out[0]["status"] == "unlocated" and out[0]["rects"] == []
    assert out[0]["source_id"] == "a" and out[0]["reason"] == "invalid-rectangles"


def test_invalid_model_geometry_preserves_independently_located_quote(monkeypatch, pages):
    monkeypatch.setattr(corpus, "source_pdf", lambda _: "dummy.pdf")
    located = {"left": 1, "top": 2, "width": 3, "height": 4}
    monkeypatch.setattr(citations, "locate_quote", lambda *a: [{"rects": [located], "coverage": 1}])
    monkeypatch.setattr(citations, "_page_words", lambda *a: ("word",))
    monkeypatch.setattr(citations, "locate_numbers", lambda *a: ([], []))
    out = citeparse.resolve_citations("", [{"source_id": "a", "page": 7,
        "quote": "existing label", "rects": [{"x": 461, "y": 214, "w": 38, "h": 11}]}], pages)
    assert out[0]["status"] == "text-located"
    assert out[0]["rects"] == [located] and out[0]["reason"] == "invalid-rectangles"
    assert out[0]["verified"] is False


def test_zoom_is_one_request_to_attached_source(pages):
    raw = '---ZOOM---\n{"source_id":"b","page":7,"rect":{"x":10,"y":20,"w":30,"h":40}}'
    request = evidence.parse_zoom(raw)
    target, rect = evidence.resolve_zoom(request, pages)
    assert target["source_id"] == "b" and rect["width"] == 30
    assert evidence.resolve_zoom({**request, "page": 8}, pages) is None
    assert evidence.parse_zoom(raw + '\n' + raw) is None
    assert evidence.parse_zoom(raw.replace('"page":7', '"page":7.5')) is None


def test_zoom_crop_includes_context_and_header_maps_original_page(pages):
    buffer = io.BytesIO()
    Image.new("RGB", (1000, 1000)).save(buffer, format="PNG")
    rect = {"x": 10, "y": 20, "w": 30, "h": 40}
    crop = evidence.crop_with_context(buffer.getvalue(), rect)
    with Image.open(io.BytesIO(crop)) as img:
        assert img.size == (460, 560)
    page = {**pages[0], "zoom_rect": evidence.context_rect(rect)}
    assert "source_id=a" in prompts.page_header(page)
    assert '"left": 2.0' in prompts.zoom_header(page)
    assert "never crop percentages" in prompts.zoom_header(page)
