"""Both adapters attach the full page and a mapped crop without API calls."""
import base64
import sys
from types import ModuleType, SimpleNamespace
import pytest

import prompts
import providers


def page(query_image=False):
    return {"source_id": "source-a", "document": "A", "page": 1,
            "image": b"full-page", "mime": "image/jpeg", "zoom_image": b"crop",
            "zoom_rect": {"left": 5, "top": 10, "width": 50, "height": 60},
            **({"query_image": b"user-reference"} if query_image else {})}


@pytest.mark.parametrize("query_image", [False, True])
def test_anthropic_attaches_full_page_before_mapped_crop(monkeypatch, query_image):
    captured = {}
    class Stream:
        text_stream = ["answer"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get_final_message(self):
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    def stream(**kwargs):
        captured.update(kwargs)
        return Stream()
    reader = providers.AnthropicReader()
    monkeypatch.setattr(reader, "_client", lambda: SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    reply = reader.read_pages("system", "question", [page(query_image)], prompts.page_header, lambda _: None)
    content = captured["messages"][0]["content"]
    assert reply.text == "answer"
    expected = ([b"user-reference"] if query_image else []) + [b"full-page", b"crop"]
    assert [base64.b64decode(p["source"]["data"]) for p in content if p["type"] == "image"] == expected
    if query_image:
        assert content[1]["text"] == providers.QUERY_IMAGE_HEADER
        assert content[2]["source"]["media_type"] == "image/png"
    assert content[-2]["text"] == prompts.zoom_header(page())
    assert content[-1]["source"]["media_type"] == "image/png"


@pytest.mark.parametrize("query_image", [False, True])
def test_gemini_attaches_full_page_before_mapped_crop(monkeypatch, query_image):
    captured = {}
    class Part:
        @staticmethod
        def from_text(*, text):
            return {"text": text}
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime": mime_type}
    types = SimpleNamespace(Part=Part, Content=lambda **kw: kw,
                            GenerateContentConfig=lambda **kw: kw)
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.types = types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setattr(providers, "_thinking_config", lambda _: None)
    def stream(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
            parts=[SimpleNamespace(text="answer")]))])]
    reader = providers.GeminiReader()
    monkeypatch.setattr(reader, "_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream)))
    reply = reader.read_pages("system", "question", [page(query_image)], prompts.page_header, lambda _: None)
    parts = captured["contents"][0]["parts"]
    assert reply.text == "answer"
    expected = ([b"user-reference"] if query_image else []) + [b"full-page", b"crop"]
    assert [p["data"] for p in parts if "data" in p] == expected
    if query_image:
        assert parts[1]["text"] == providers.QUERY_IMAGE_HEADER
        assert parts[2]["mime"] == "image/png"
    assert parts[-2]["text"] == prompts.zoom_header(page())
    assert parts[-1]["mime"] == "image/png"
