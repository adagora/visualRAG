import pytest

import jev_memo

QUESTIONS = {'x':{'type':'noul','instructions':'Does x hold?'}}


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv('PIXELRAG_JEV_MEMO', str(tmp_path/'memo.sqlite'))
    monkeypatch.delenv('PIXELRAG_JEV_MEMO_DISABLED', raising=False)


def test_reuses_raw_answer_and_invalidates_on_prompt_model_state_version(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_memo.jev, 'evaluate', lambda *a, **k: calls.append(a) or {'x':{'noul':.7}})
    def ask(state='state', questions=QUESTIONS, version=1):
        return jev_memo.evaluate(state, questions, version=version, kind='test')
    assert ask() == ask() == {'x':{'noul':.7}}
    assert len(calls) == 1
    ask('changed'); ask(version=2)
    ask(questions={'x':{'type':'noul','instructions':'A different question'}})
    monkeypatch.setenv('TYPESAFE_MODEL','other-model'); ask()
    assert len(calls) == 5


def test_malformed_answers_are_never_cached(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(jev_memo.jev,'evaluate',lambda *a,**k: calls.append(a) or {'x':{'noul':float('nan')}})
    for _ in range(2):
        jev_memo.evaluate('state',QUESTIONS,version=1,kind='test')
    assert len(calls) == 2


def test_expired_judgments_are_evaluated_fresh(cache, monkeypatch):
    clock = [1000]
    monkeypatch.setattr(jev_memo.time,'time',lambda:clock[0])
    calls = []
    monkeypatch.setattr(jev_memo.jev,'evaluate',lambda *a,**k: calls.append(a) or {'x':{'noul':.8}})
    jev_memo.evaluate('state',QUESTIONS,version=1,kind='test')
    clock[0] += jev_memo.TTL + 1
    jev_memo.evaluate('state',QUESTIONS,version=1,kind='test')
    assert len(calls) == 2


def test_unavailable_cache_is_only_a_miss(cache, monkeypatch, tmp_path):
    monkeypatch.setenv('PIXELRAG_JEV_MEMO',str(tmp_path/'missing-dir'/'db'))
    monkeypatch.setattr(jev_memo.jev,'evaluate',lambda *a,**k: {'x':{'noul':.8}})
    assert jev_memo.evaluate('state',QUESTIONS,version=1,kind='test')['x']['noul'] == .8
