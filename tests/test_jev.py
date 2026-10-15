"""Real image preparation and mocked HTTP; no paid decisions in the test suite."""
import base64
import io
from http.client import IncompleteRead
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from PIL import Image
import pytest
import jev


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    hits, paths = [], {}
    for number in range(8):
        path = tmp_path / f"{number}.jpg"
        Image.new("RGB", (2000, 2400), (number * 20, 10, 20)).save(path)
        hits.append({"article_id": 0, "source_id": "source", "document": "Catalog",
                     "page": number + 1, "score": 1 - number / 10})
        paths[0, number + 1] = path
    index = SimpleNamespace(page_path=lambda aid, page: paths[aid, page])
    response = {"model": jev.MODEL, "usage": {"input_tokens": 100, "output_tokens": 8},
                "answers": {f"page_{i}": {"type": "noul", "noul": i / 10} for i in range(8)}}
    request = Mock(return_value=response)
    monkeypatch.setattr(jev, "_request", request)
    return index, hits, response, request


def test_one_image_question_per_candidate_and_actual_model(sample):
    index, hits, response, request = sample
    reference = io.BytesIO()
    Image.new("RGB", (40, 600), "white").save(reference, format="PNG")
    ranked, report = jev.rerank(index, "Which product?", hits, image=reference.getvalue())
    assert [p["page"] for p in ranked] == [8, 7, 6, 5]
    assert ranked[0]["score"] == hits[7]["score"]  # Preserve dense similarity separately.
    assert "jev_score" not in hits[7]
    assert report["status"] == "applied" and report["scored_count"] == 8
    assert report["model"] == report["requested_model"] == "jev-1.14.0"
    assert report["usage"]["input_tokens"] == 100
    request.assert_called_once()
    payload, key = request.call_args.args
    assert key == "test-key" and payload["model"] == jev.MODEL
    assert set(payload["state"]) == {"question", "reference_image"}
    assert len(payload["assets"]) == 9 and len(payload["questions"]) == 8
    refs = [q["instructions"]["candidate"]["image"] for q in payload["questions"].values()]
    assert len(set(refs)) == 8
    assert payload["state"]["reference_image"] not in refs
    for asset in payload["assets"].values():
        assert asset["format"] == "jpg"
        with Image.open(io.BytesIO(base64.b64decode(asset["data_base64"]))) as image:
            assert 256 <= min(image.size) <= max(image.size) <= jev.MAX_SIDE
            assert image.width * image.height >= 78400
            assert not image.info.get("progressive")


@pytest.mark.parametrize("value", [True, -0.1, 1.01, float("nan"), float("inf"), "0.9", None])
def test_invalid_probability_preserves_original_ranking(sample, value):
    index, hits, response, request = sample
    response["answers"]["page_3"]["noul"] = value
    ranked, report = jev.rerank(index, "Question", hits)
    assert ranked == hits[:4] and report["status"] == "fallback"
    assert report["scored_count"] == 0 and report["calls"] == 1


@pytest.mark.parametrize("change", ["missing_answer", "wrong_type", "no_model", "invalid_usage"])
def test_malformed_response_is_not_partly_applied(sample, change):
    index, hits, response, request = sample
    if change == "missing_answer": del response["answers"]["page_0"]
    if change == "wrong_type": response["answers"]["page_0"]["type"] = "score"
    if change == "no_model": del response["model"]
    if change == "invalid_usage": response["usage"]["input_tokens"] = True
    ranked, report = jev.rerank(index, "Question", hits)
    assert ranked == hits[:4] and report["status"] == "fallback"


def test_ties_retain_original_order(sample):
    index, hits, response, request = sample
    for answer in response["answers"].values(): answer["noul"] = .5
    ranked, report = jev.rerank(index, "Question", hits)
    assert [p["page"] for p in ranked] == [1, 2, 3, 4]


def test_missing_key_and_empty_retrieval_make_no_request(sample, monkeypatch):
    index, hits, response, request = sample
    monkeypatch.delenv("TYPESAFE_API_KEY")
    ranked, report = jev.rerank(index, "Question", hits)
    assert ranked == hits[:4] and report["calls"] == 0
    assert report["status"] == "fallback"
    assert jev.rerank(index, "Question", [])[1]["status"] == "skipped"
    request.assert_not_called()


def test_transport_failure_preserves_dense_order(sample):
    index, hits, response, request = sample
    request.side_effect = jev.JevError("Jev request failed (TimeoutError).")
    ranked, report = jev.rerank(index, "Question", hits)
    assert ranked == hits[:4] and report["status"] == "fallback"
    assert "TimeoutError" in report["reason"]


def test_cancellation_is_not_treated_as_fallback(sample):
    index, hits, response, request = sample
    class Cancelled(RuntimeError): pass
    def cancel(): raise Cancelled()
    with pytest.raises(Cancelled):
        jev.rerank(index, "Question", hits, check_cancelled=cancel)
    request.assert_not_called()


def test_http_contract(monkeypatch):
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'{"model":"jev-1.14.0"}'
    def open_request(request, timeout):
        seen.append((request, timeout))
        return Response()
    monkeypatch.setattr(jev, "urlopen", open_request)
    assert jev._request({"test": 1}, "secret")["model"] == jev.MODEL
    request, timeout = seen[0]
    assert request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert request.method == "POST" and timeout == jev.TIMEOUT
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["typesafe-danger-zone-images"] == "true"
    assert headers["authorization"] == "Bearer secret"
    assert json.loads(request.data) == {"test": 1}


@pytest.mark.parametrize("error", [HTTPError(jev.ENDPOINT, 429, "private body", {}, io.BytesIO(b"private body")),
                                    URLError("secret provider detail"), TimeoutError("secret"), IncompleteRead(b"private")])
def test_http_errors_expose_no_request_or_provider_details(monkeypatch, error):
    monkeypatch.setattr(jev, "urlopen", Mock(side_effect=error))
    with pytest.raises(jev.JevError) as raised:
        jev._request({}, "private-key")
    assert "private" not in str(raised.value) and "secret" not in str(raised.value)


def test_invalid_decisions_retain_known_billed_usage(sample):
    index, hits, response, request = sample
    response["answers"]["page_0"]["noul"] = True
    ranked, report = jev.rerank(index, "Question", hits)
    assert report["status"] == "fallback" and report["model"] == jev.MODEL
    assert report["usage"]["input_tokens"] == 100


def test_context_limit_is_reported_without_echoing_payload(monkeypatch):
    body = io.BytesIO(b'{"detail":{"error_type":"max_tokens_exceeded"},"private":"secret"}')
    error = HTTPError(jev.ENDPOINT, 400, "Bad Request", {}, body)
    monkeypatch.setattr(jev, "urlopen", Mock(side_effect=error))
    with pytest.raises(jev.JevError, match="max_tokens_exceeded") as raised:
        jev._request({}, "private-key")
    assert "secret" not in str(raised.value)
