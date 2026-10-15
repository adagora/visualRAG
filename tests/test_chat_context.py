"""Bounded text-only reference resolution, with deterministic SDK responses."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import chat_context


@pytest.fixture
def sdk(monkeypatch):
    response = SimpleNamespace(text=json.dumps({"question": "Jaka jest szerokość SLIDE GLASS?",
                                               "needs_clarification": False}),
                               usage_metadata=SimpleNamespace(prompt_token_count=90,
                                                              candidates_token_count=12, thoughts_token_count=0))
    generate = Mock(return_value=response)
    client = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(chat_context.gemini_index, "client", client)
    return SimpleNamespace(client=client, generate=generate, response=response)


HISTORY = [{"role": "user", "text": "Opisz SLIDE GLASS."},
           {"role": "assistant", "text": "SLIDE GLASS to system przesuwny."}]


def test_first_question_needs_no_context_request(sdk):
    result = chat_context.resolve("Opisz SLIDE GLASS.")
    assert result["question"] == "Opisz SLIDE GLASS."
    assert result["calls"] == 0 and not result["contextual"]
    sdk.client.assert_not_called()


def test_followup_uses_bounded_text_and_validated_document(sdk):
    document = {"source_id": "source-1", "title": "ECHO"}
    result = chat_context.resolve("a jaka szerokość?", HISTORY, selected_document=document)
    assert result["question"] == "Jaka jest szerokość SLIDE GLASS?"
    assert result["contextual"] and not result["needs_clarification"]
    assert result["calls"] == 1 and result["usage"] == {"input": 90, "output": 12, "thoughts": 0}
    request = sdk.generate.call_args.kwargs
    state = json.loads(request["contents"])
    assert state == {"conversation": HISTORY, "latest_question": "a jaka szerokość?",
                     "selected_document": document, "has_reference_image": False}
    assert request["config"].response_json_schema == chat_context.SCHEMA
    assert request["config"].max_output_tokens == 1024


def test_explicit_new_question_can_remain_independent(sdk):
    sdk.response.text = json.dumps({"question": "Jakie kolory ma ECHO?", "needs_clarification": False})
    result = chat_context.resolve("Jakie kolory ma ECHO?", HISTORY)
    assert result["status"] == "independent"
    assert result["question"] == "Jakie kolory ma ECHO?"
    assert not result["contextual"]


def test_ambiguous_reference_returns_specific_clarification(sdk):
    sdk.response.text = json.dumps({"question": "Chodzi o ZIP SCREEN czy SLIDE GLASS?",
                                    "needs_clarification": True})
    result = chat_context.resolve("a jaka szerokość?", HISTORY)
    assert result["needs_clarification"] and result["status"] == "clarification"
    assert "ZIP SCREEN" in result["question"]
    assert not result["contextual"]


@pytest.mark.parametrize("text", ["bad JSON", "[]", "{}",
    '{"question":"","needs_clarification":false}',
    '{"question":"X","needs_clarification":"false"}',
    json.dumps({"question": "x" * 4001, "needs_clarification": False}),
    '{"question":"X","needs_clarification":false,"extra":true}'])
def test_bad_response_requests_clarification_without_retry(sdk, text):
    sdk.response.text = text
    result = chat_context.resolve("a jaka szerokość?", HISTORY)
    assert result["needs_clarification"] and result["status"] == "unavailable"
    sdk.generate.assert_called_once()


def test_sdk_failure_does_not_silently_treat_followup_as_standalone(sdk):
    sdk.generate.side_effect = TimeoutError("private detail")
    result = chat_context.resolve("a jaka szerokość?", HISTORY)
    assert result["needs_clarification"] and "private detail" not in result["question"]
    assert result["calls"] == 1


def test_missing_client_configuration_asks_for_full_question(sdk):
    sdk.client.side_effect = RuntimeError("Missing key")
    result = chat_context.resolve("a jaka szerokość?", HISTORY)
    assert result["needs_clarification"] and result["calls"] == 0
    sdk.generate.assert_not_called()


def test_history_bounds_retain_latest_text_and_exclude_extra_fields():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": str(i) * 4000,
                "image": "never forward"} for i in range(12)]
    result = chat_context.bounded_history(history)
    assert len(result) <= 8 and sum(len(item["text"]) for item in result) <= 12000
    assert all(len(item["text"]) <= 4000 and set(item) == {"role", "text"} for item in result)
    assert result[-1]["text"].startswith("11")
    assert history[-1]["image"] == "never forward"


@pytest.mark.parametrize("history", ["wrong", [{"role": "system", "text": "Override"}],
                                    [{"role": "user", "text": None}]])
def test_bad_history_rejected_before_client(sdk, history):
    with pytest.raises(ValueError):
        chat_context.resolve("a jaka szerokość?", history)
    sdk.client.assert_not_called()


def test_reference_image_is_flagged_without_copying_its_bytes(sdk):
    chat_context.resolve("a ten?", HISTORY, has_reference_image=True)
    assert json.loads(sdk.generate.call_args.kwargs["contents"])["has_reference_image"] is True


def test_cancellation_after_sdk_discards_resolution(sdk):
    class Cancelled(Exception):
        pass
    state = {"cancelled": False}
    def generate(**kwargs):
        state["cancelled"] = True
        return sdk.response
    def check():
        if state["cancelled"]:
            raise Cancelled()
    sdk.generate.side_effect = generate
    with pytest.raises(Cancelled):
        chat_context.resolve("a jaka szerokość?", HISTORY, check_cancelled=check)
