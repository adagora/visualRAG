"""Screenshot inputs are bounded, normalized and forwarded without network."""
import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as webapp
import visual_query


def encoded_image(size=(64, 32), *, fmt="PNG", orientation=None):
    out = io.BytesIO()
    image = Image.new("RGB", size, (10, 20, 30))
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(out, format=fmt, **options)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(out.getvalue()).decode()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_accepted_screenshot_formats_normalize_to_rgb_png(fmt):
    decoded = visual_query.decode_image(encoded_image(fmt=fmt))
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (64, 32) and image.mode == "RGB" and image.format == "PNG"


def test_screenshot_orientation_is_applied_before_forwarding():
    decoded = visual_query.decode_image(encoded_image((32, 64), fmt="JPEG", orientation=6))
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (64, 32)
        assert not image.getexif().get(274)


def test_screenshot_dimensions_are_bounded_and_preserve_aspect_ratio():
    decoded = visual_query.decode_image(encoded_image((3000, 1500)))
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (2048, 1024)


@pytest.mark.parametrize("value", [
    "https://example.com/image.png", "data:image/svg+xml;base64,PHN2Zz4=",
    "data:image/png;base64,%%%", "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
])
def test_bad_screenshot_payload_is_rejected(value):
    with pytest.raises(ValueError):
        visual_query.decode_image(value)


def test_tiny_and_excessive_pixel_images_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="16 px"):
        visual_query.decode_image(encoded_image((15, 64)))
    monkeypatch.setattr(visual_query, "MAX_PIXELS", 1000)
    with pytest.raises(ValueError, match="million pixels"):
        visual_query.decode_image(encoded_image((32, 32)))


def test_encoded_payload_size_is_bounded_before_image_decode(monkeypatch):
    monkeypatch.setattr(visual_query, "MAX_BYTES", 32)
    with pytest.raises(ValueError, match="at most 5 MB"):
        visual_query.decode_image("data:image/png;base64," + "A" * 100)


def test_optional_screenshot_is_absent_when_unset():
    assert visual_query.decode_image(None) is None
    assert visual_query.decode_image("") is None


@pytest.mark.parametrize("route", ["/api/ask", "/api/compare"])
def test_invalid_screenshot_is_http400_before_retrieval(monkeypatch, route):
    monkeypatch.setattr(webapp.rag, "run_agent", lambda *a, **k: pytest.fail("reader called"))
    monkeypatch.setattr(webapp.compare_modes, "compare", lambda *a, **k: pytest.fail("retrieval called"))
    with TestClient(webapp.app) as client:
        response = client.post(route, json={"question":"q", "query_image":"data:image/png;base64,%%%"})
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_ask_passes_normalized_reference_pixels_to_reader(monkeypatch):
    seen = {}
    monkeypatch.setattr(webapp.chat_categories, "plan", lambda *a, **k: pytest.fail("planner called"))
    monkeypatch.setattr(webapp.chat_context, "resolve", lambda *a, **k: pytest.fail("context called"))
    monkeypatch.setattr(webapp.rag, "run_agent", lambda q, **k: seen.update(question=q, **k) or {"answer":"a"})
    with TestClient(webapp.app) as client:
        response = client.post("/api/ask", json={"question":"Which component is shown?",
            "query_image":encoded_image((32, 64), fmt="JPEG", orientation=6), "pages":2})
    assert response.status_code == 200 and seen["n_pages"] == 2
    assert seen["question"] == "Which component is shown?"
    with Image.open(io.BytesIO(seen["query_image"])) as image:
        assert image.size == (64, 32) and image.format == "PNG"
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["kind"] == "done"


def test_compare_passes_normalized_image_to_retriever(monkeypatch):
    seen = {}
    def compare(question, modes, pages, **kwargs):
        seen.update(question=question, modes=modes, pages=pages, **kwargs)
        return iter([])
    monkeypatch.setattr(webapp.compare_modes, "compare", compare)
    monkeypatch.setattr(webapp.compare_modes, "agreement", lambda rows: [])
    with TestClient(webapp.app) as client:
        response = client.post("/api/compare", json={"question":"Find diagram", "query_image":encoded_image(),
                                                       "modes":["visual"], "pages":3})
    assert response.status_code == 200 and seen["pages"] == 3
    with Image.open(io.BytesIO(seen["query_image"])) as image:
        assert image.format == "PNG" and image.size == (64, 32)
