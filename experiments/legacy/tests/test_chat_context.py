import json
from types import SimpleNamespace

import pytest

import chat_context
import providers


@pytest.fixture(autouse=True)
def no_jev(monkeypatch):
    monkeypatch.setattr(chat_context.jev, "enabled", lambda: False)


def reader(monkeypatch, response):
    seen = []
    def read_pages(system, preamble, pages, header_of, on_text):
        seen.append(json.loads(preamble))
        assert not pages
        return providers.Reply(response, providers.Usage(input=20, output=10))
    monkeypatch.setattr(providers, 'reader_for', lambda **k: SimpleNamespace(
        read_pages=read_pages, price=lambda usage: .001))
    return seen


def test_first_question_does_not_need_resolution(monkeypatch):
    monkeypatch.setattr(providers, 'reader_for', lambda **k: pytest.fail('unneeded reader'))
    assert chat_context.resolve('What is Connect?', [])['question'] == 'What is Connect?'


def test_followup_uses_complete_recent_messages_and_exact_cache_scope(monkeypatch):
    history = [{'role': 'user', 'text': 'What is the warranty on Connect?'},
               {'role': 'assistant', 'text': 'It depends on the component.'}]
    seen = reader(monkeypatch, json.dumps({'question': 'What is the warranty on the drive for Connect?',
                                         'needs_clarification': False}))
    result = chat_context.resolve('And for the drive?', history)
    assert seen[0]['conversation'] == history
    assert result['contextual'] and result['cache_scope']
    assert result['usage']['input'] == 20
    # Currency/product changes cannot use another contextual answer namespace.
    reader(monkeypatch, json.dumps({'question': 'What is the warranty on the drive for Prime?',
                                    'needs_clarification': False}))
    assert chat_context.resolve('And Prime?', history)['cache_scope'] != result['cache_scope']


def test_independent_question_stays_independent(monkeypatch):
    reader(monkeypatch, json.dumps({'question': 'What colours does Soltis have?', 'needs_clarification': False}))
    result = chat_context.resolve('What colours does Soltis have?', [{'role':'user','text':'Price of Connect?'}])
    assert not result['contextual'] and result['cache_scope'] is None


@pytest.mark.parametrize('bad', ['bad JSON', '{}', '{"question":"", "needs_clarification":false}',
                                 '{"question":"x", "needs_clarification":"false"}'])
def test_invalid_resolution_asks_instead_of_guessing(monkeypatch, bad):
    reader(monkeypatch, bad)
    assert chat_context.resolve('And that?', [{'role':'user','text':'Connect?'}])['needs_clarification']


def test_history_budget_preserves_whole_messages():
    history = [{'role':'user', 'text':'old' * 5000}, {'role':'user','text':'current subject'}]
    assert chat_context.bounded_history(history) == history[-1:]


def test_reader_json_fence_is_accepted(monkeypatch):
    reader(monkeypatch, '```json\n{"question":"Colours of Connect?", "needs_clarification":false}\n```')
    result = chat_context.resolve('And colours?', [{'role':'user','text':'Price of Connect?'}])
    assert result['question'] == 'Colours of Connect?' and not result['needs_clarification']


def test_independent_jev_judgment_skips_reader(monkeypatch):
    monkeypatch.setattr(chat_context.jev,'enabled',lambda:True)
    monkeypatch.setattr(chat_context.jev_memo,'evaluate',lambda *a,**k:{'followup':{'noul':.01}})
    monkeypatch.setattr(providers,'reader_for',lambda **k:pytest.fail('rewrote independent question'))
    assert chat_context.resolve('Soltis colours?', [{'role':'user','text':'Connect price?'}])['question'] == 'Soltis colours?'
