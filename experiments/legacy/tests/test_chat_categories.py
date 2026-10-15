import pytest
import requests

import chat_categories as categories
import jev


@pytest.fixture(autouse=True)
def isolated_memo(monkeypatch, tmp_path):
    monkeypatch.setenv('PIXELRAG_JEV_MEMO', str(tmp_path/'memo.sqlite'))


def judgment(category='price', confidence=.9):
    return {'topic': {'choice': category, 'confidence': confidence,
                     'probabilities': {k: 1 if k == category else 0 for k in categories.CRITERIA}}}


@pytest.mark.parametrize('topic', categories.CRITERIA)
def test_choice_categories_are_validated(monkeypatch, topic):
    monkeypatch.setattr(jev, 'enabled', lambda: True)
    def evaluate(state, questions, **kwargs):
        assert state == {'question': 'my question'}
        assert questions['topic']['type'] == 'choice'
        assert kwargs['timeout'] == 3
        return judgment(topic)
    monkeypatch.setattr(jev, 'evaluate', evaluate)
    assert categories.classify('my question') == {'category': topic, 'confidence': .9}


def test_disabled_does_not_call_model(monkeypatch):
    monkeypatch.setattr(jev, 'enabled', lambda: False)
    monkeypatch.setattr(jev, 'evaluate', lambda *a, **k: pytest.fail('called disabled model'))
    assert categories.classify('price?')['category'] == 'unclassified'


@pytest.mark.parametrize('confidence', [.2, float('nan'), True, 1.1])
def test_low_or_invalid_confidence_stays_neutral(monkeypatch, confidence):
    monkeypatch.setattr(jev, 'enabled', lambda: True)
    monkeypatch.setattr(jev, 'evaluate', lambda *a, **k: judgment(confidence=confidence))
    assert categories.classify('price?')['category'] == 'unclassified'


def test_outage_stays_neutral(monkeypatch):
    monkeypatch.setattr(jev, 'enabled', lambda: True)
    def fail(*a, **k):
        raise requests.Timeout()
    monkeypatch.setattr(jev, 'evaluate', fail)
    assert categories.classify('price?') == {'category': 'unclassified', 'confidence': None}
