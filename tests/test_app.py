"""HTTP/SSE boundaries; no embedding or reader calls."""
import base64
import io
import json
from fastapi.testclient import TestClient
from PIL import Image
import pytest
import app

@pytest.fixture
def web(tmp_path):
    pdf = tmp_path / "Pergola-żółta.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 100)
    image = tmp_path / "page.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    class Index:
        documents = [{"article_id": 0, "source_id": "doc-1", "title": "Pergola", "pages": 2}]
        def page_path(self, aid, page):
            if aid != 0 or not 1 <= page <= 2: raise ValueError("Unknown page")
            return image
        def source_pdf(self, aid):
            if aid != 0: raise ValueError("Unknown document")
            return pdf
    index = Index()
    app.app.dependency_overrides[app.get_index] = lambda: index
    with TestClient(app.app) as client:
        yield client, index
    app.app.dependency_overrides.clear()

def frames(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

def test_documents_images_and_pdf_ranges(web):
    client, index = web
    assert client.get("/api/docs").json() == {"docs": index.documents}
    assert client.get("/api/page/0/1").headers["content-type"] == "image/jpeg"
    response = client.get("/api/pdf/0", headers={"Range": "bytes=0-8"})
    assert response.status_code == 206 and response.content == b"%PDF-1.7\n"
    assert "filename*=UTF-8''Pergola-" in response.headers["content-disposition"]
    for url in ("/api/page/-1/1", "/api/page/0/0", "/api/page/0/3", "/api/pdf/1"):
        assert client.get(url).status_code == 404


def test_source_identity_controls_links_and_caching(web):
    client, _ = web
    for path in ("/api/pdf/0", "/api/page/0/1"):
        assert client.get(path).headers["cache-control"] == "no-store"
        response = client.get(path, params={"source": "doc-1"})
        assert response.status_code == 200
        assert "max-age=86400" in response.headers["cache-control"]
        assert client.get(path, params={"source": "old-document"}).status_code == 409
    assert client.get("/").headers["cache-control"] == "no-store"

def test_evidence_arrives_before_answer(web, monkeypatch):
    client, index = web
    hit = {"article_id": 0, "source_id": "doc-1", "document": "Pergola", "page": 2, "score": .8}
    calls = []
    def answer(actual_index, question, *, image, on_event, cancelled):
        calls.append((actual_index, question, image))
        assert not cancelled()
        on_event({"type": "retrieved", "pages": [hit]})
        on_event({"type": "reading"})
        return {"answer": "726", "citations": [hit], "pages": [hit], "reader_calls": 1}
    monkeypatch.setattr(app.rag, "answer", answer)
    response = client.post("/api/ask", json={"question": "  Jaka wartość?  "})
    assert response.status_code == 200
    result = frames(response)
    assert [f["kind"] for f in result] == ["event", "event", "done"]
    assert result[0]["payload"]["pages"] == [hit]
    assert result[-1]["payload"]["answer"] == "726"
    assert calls == [(index, "Jaka wartość?", None)]

def test_image_only_request(web, monkeypatch):
    source = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(source, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(source.getvalue()).decode()
    calls = []
    def answer(index, question, **kwargs):
        calls.append((question, kwargs["image"]))
        return {"answer": "Element.", "citations": [], "pages": []}
    monkeypatch.setattr(app.rag, "answer", answer)
    assert frames(web[0].post("/api/ask", json={"query_image": uri}))[-1]["kind"] == "done"
    assert calls[0][0] and calls[0][1].startswith(b"\x89PNG")

@pytest.mark.parametrize("payload,status", [
    ({"question": " "}, 400), ({"question": "x", "query_image": "invalid"}, 400),
    ({"question": "x", "retrieval": "jev"}, 422), ({"question": "x", "api_key": "unused"}, 422),
    ({"question": "x" * 4001}, 422),
])
def test_invalid_requests_do_not_call_models(web, monkeypatch, payload, status):
    monkeypatch.setattr(app.rag, "answer", lambda *a, **k: pytest.fail("Invalid input reached reader"))
    assert web[0].post("/api/ask", json=payload).status_code == status

def test_reader_error_is_streamed_without_success(web, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError("Reader unavailable")
    monkeypatch.setattr(app.rag, "answer", fail)
    result = frames(web[0].post("/api/ask", json={"question": "Test"}))
    assert [f["kind"] for f in result] == ["error"]
    assert "Reader unavailable" in result[0]["payload"]

def test_missing_index_fails_without_fallback(tmp_path, monkeypatch):
    app.get_index.cache_clear()
    monkeypatch.setenv("PIXELRAG_INDEX_DIR", str(tmp_path / "missing"))
    with TestClient(app.app) as client: response = client.get("/api/docs")
    app.get_index.cache_clear()
    assert response.status_code == 503
    assert "Cannot open Gemini index" in response.json()["detail"]

def test_research_routes_removed(web):
    for path in ("/api/modes", "/api/compare", "/api/tile/0/0/0", "/api/snip/0/1"):
        assert web[0].get(path).status_code == 404
