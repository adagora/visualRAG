"""Offline contract tests: no embeddings or reader requests reach the network."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rag


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELRAG_JEV", "off")
    hits = [{"article_id": i // 2, "source_id": "source-" + str(i // 2),
             "page": i % 2 + 1, "document": "Katalog " + str(i // 2),
             "score": 1 - i / 10} for i in range(6)]
    paths = {}
    for hit in hits:
        path = tmp_path / (str(hit["article_id"]) + "-" + str(hit["page"]) + ".jpg")
        path.write_bytes(b"\xff\xd8original full page " + path.name.encode() + b"\xff\xd9")
        paths[hit["article_id"], hit["page"]] = path
    index = SimpleNamespace(
        search=Mock(return_value=hits),
        page_path=Mock(side_effect=lambda article_id, page: paths[article_id, page]))
    response = SimpleNamespace(
        text=json.dumps({"answer": "Cena wynosi 20 zł/m².",
                         "citations": [{"source_id": "source-0", "page": 1}]}),
        usage_metadata=SimpleNamespace(prompt_token_count=100,
                                       candidates_token_count=20, thoughts_token_count=5))
    generate = Mock(return_value=response)
    client = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(rag.gemini_index, "client", client)
    return SimpleNamespace(index=index, hits=hits, paths=paths, generate=generate,
                           client=client, response=response)


def test_single_search_reader_and_original_pages(setup):
    events = []
    result = rag.answer(setup.index, "Jaka cena?", on_event=events.append)
    setup.index.search.assert_called_once_with("Jaka cena?", image=None, limit=4)
    setup.generate.assert_called_once()
    assert result["reader_calls"] == 1
    assert result["pages"] == setup.hits[:4]
    assert result["citations"] == [{key: setup.hits[0][key] for key in
                                      ("source_id", "article_id", "document", "page")}]
    assert result["usage"] == {"input": 100, "output": 20, "thoughts": 5}
    assert all(value >= 0 for value in result["timings"].values())
    assert events == [{"type": "retrieved", "pages": setup.hits[:4]}, {"type": "reading"}]
    request = setup.generate.call_args.kwargs
    parts = request["contents"][0].parts
    images = [part.inline_data for part in parts if part.inline_data]
    assert [blob.mime_type for blob in images] == ["image/jpeg"] * 4
    assert [blob.data for blob in images] == [
        setup.paths[hit["article_id"], hit["page"]].read_bytes() for hit in setup.hits[:4]]
    labels = [json.loads(part.text.removeprefix("EVIDENCE ")) for part in parts
              if part.text and part.text.startswith("EVIDENCE ")]
    assert labels == [{key: hit[key] for key in ("source_id", "document", "page")}
                      for hit in setup.hits[:4]]
    config = request["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == rag.RESPONSE_SCHEMA
    assert config.max_output_tokens == 4096
    assert config.thinking_config.thinking_level.value == "LOW"
    assert "jednostki" in config.system_instruction
    assert "m²" in config.system_instruction
    assert "brakuje dowodu" in config.system_instruction
    assert "nie jest ceną za sztukę" in config.system_instruction


def test_reference_is_separate_png_context_and_never_cited(setup):
    screenshot = b"\x89PNG\r\nreference"
    setup.response.text = json.dumps({"answer": "Brak czytelnej ceny.", "citations": [
        {"source_id": "REFERENCE", "page": 1},
        {"source_id": "source-2", "page": 1},
        {"source_id": "source-0", "page": 2},
        {"source_id": "source-0", "page": 1},
        {"source_id": "source-0", "page": 1}]})
    result = rag.answer(setup.index, "", image=screenshot, pages=1)
    setup.index.search.assert_called_once_with("", image=screenshot, limit=1)
    images = [part.inline_data for part in setup.generate.call_args.kwargs["contents"][0].parts
              if part.inline_data]
    assert [(blob.mime_type, blob.data) for blob in images] == [
        ("image/png", screenshot), ("image/jpeg", setup.paths[0, 1].read_bytes())]
    assert [(citation["source_id"], citation["page"]) for citation in result["citations"]] == [
        ("source-0", 1)]
    assert len(result["pages"]) == 1


def test_empty_retrieval_abstains_without_reader(setup):
    setup.index.search.return_value = []
    result = rag.answer(setup.index, "Ile kosztuje?")
    setup.index.search.assert_called_once()
    setup.client.assert_not_called()
    assert result["reader_calls"] == 0
    assert result["citations"] == []
    assert "Nie znaleziono" in result["answer"]


@pytest.mark.parametrize("budget", [0, 5, -1, True, 1.5, "4"])
def test_invalid_budget_stops_before_retrieval(setup, budget):
    with pytest.raises(ValueError, match="budget"):
        rag.answer(setup.index, "Cena?", pages=budget)
    setup.index.search.assert_not_called()
    setup.client.assert_not_called()


def test_duplicate_search_results_are_attached_once(setup):
    setup.index.search.return_value = [setup.hits[0], setup.hits[0], *setup.hits[1:]]
    result = rag.answer(setup.index, "Cena?", pages=2)
    assert result["pages"] == setup.hits[:2]
    assert setup.index.page_path.call_count == 2


@pytest.mark.parametrize("text", ["not json", "[]", "{}",
    '{"answer":"","citations":[]}',
    '{"answer":"Cena","citations":[{"source_id":"source-0","page":true}]}',
    '{"answer":"Cena","citations":[{"source_id":"source-0","page":"1"}]}',
    '{"answer":"Cena","citations":[{"source_id":"source-0","page":0}]}'])
def test_invalid_response_fails_clearly_without_retry(setup, text):
    setup.response.text = text
    with pytest.raises(ValueError, match="Gemini reader"):
        rag.answer(setup.index, "Cena?")
    setup.generate.assert_called_once()
    setup.index.search.assert_called_once()


def test_reader_error_propagates_without_retry(setup):
    setup.generate.side_effect = RuntimeError("reader unavailable")
    with pytest.raises(RuntimeError, match="reader unavailable"):
        rag.answer(setup.index, "Cena?")
    setup.generate.assert_called_once()


def test_cancel_before_search(setup):
    with pytest.raises(rag.CancelledError):
        rag.answer(setup.index, "Cena?", cancelled=lambda: True)
    setup.index.search.assert_not_called()
    setup.client.assert_not_called()


def test_cancel_after_search_prevents_reader(setup):
    state = {"cancelled": False}
    def search(*args, **kwargs):
        state["cancelled"] = True
        return setup.hits
    setup.index.search.side_effect = search
    with pytest.raises(rag.CancelledError):
        rag.answer(setup.index, "Cena?", cancelled=lambda: state["cancelled"])
    setup.index.search.assert_called_once()
    setup.client.assert_not_called()


def test_cancel_on_reading_event_prevents_sdk_call(setup):
    state = {"cancelled": False}
    def on_event(event):
        state["cancelled"] = event["type"] == "reading"
    with pytest.raises(rag.CancelledError):
        rag.answer(setup.index, "Cena?", on_event=on_event,
                   cancelled=lambda: state["cancelled"])
    setup.client.assert_not_called()


def test_cancel_during_reader_discards_completed_response(setup):
    state = {"cancelled": False}
    def generate(**kwargs):
        state["cancelled"] = True
        return setup.response
    setup.generate.side_effect = generate
    with pytest.raises(rag.CancelledError):
        rag.answer(setup.index, "Cena?", cancelled=lambda: state["cancelled"])
    setup.generate.assert_called_once()


def test_model_from_environment_and_missing_usage(setup, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "configured-model")
    setup.response.usage_metadata = None
    result = rag.answer(setup.index, "Cena?")
    assert result["model"] == "configured-model"
    assert setup.generate.call_args.kwargs["model"] == "configured-model"
    assert result["usage"] == {"input": 0, "output": 0, "thoughts": 0}


def test_visual_decisions_select_pages_before_one_original_reader_call(setup, monkeypatch):
    from PIL import Image
    import jev
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    for path in setup.paths.values(): Image.new("RGB", (300, 400), "white").save(path)
    response = {"model": jev.MODEL, "usage": {"input_tokens": 100, "output_tokens": 6},
                "answers": {f"page_{i}": {"type": "noul", "noul": i / 10} for i in range(6)}}
    request = Mock(return_value=response)
    monkeypatch.setattr(jev, "_request", request)
    events = []
    result = rag.answer(setup.index, "Question", jev_enabled=True, on_event=events.append)
    setup.index.search.assert_called_once_with("Question", image=None, limit=8)
    assert [p["page"] for p in result["pages"]] == [2, 1, 2, 1]
    assert [p["article_id"] for p in result["pages"]] == [2, 2, 1, 1]
    assert result["retrieval"]["jev"]["status"] == "applied"
    assert [e["type"] for e in events] == ["reranking", "retrieved", "reading"]
    setup.generate.assert_called_once()
    blobs = [p.inline_data.data for p in setup.generate.call_args.kwargs["contents"][0].parts if p.inline_data]
    assert blobs == [setup.paths[h["article_id"], h["page"]].read_bytes() for h in result["pages"]]
    request.assert_called_once()


def test_jev_failure_keeps_dense_pages_and_one_reader(setup, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = rag.answer(setup.index, "Question", jev_enabled=True)
    assert result["pages"] == setup.hits[:4]
    assert result["retrieval"]["jev"]["status"] == "fallback"
    assert result["reader_calls"] == 1
    setup.generate.assert_called_once()


def test_disabled_jev_makes_no_decisions_even_with_key(setup, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(rag.jev, "_request", lambda *a: pytest.fail("Unrequested Jev call"))
    assert rag.answer(setup.index, "Question")["retrieval"]["jev"]["status"] == "off"
