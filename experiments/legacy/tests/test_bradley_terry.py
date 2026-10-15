"""Ranking checks independent of Jev's semantic accuracy."""
import itertools

import pytest

import jev


def test_recovers_known_strength_ratios():
    strengths = [1, 2, 4, 8]
    pairs = [(i, j, strengths[i] / (strengths[i] + strengths[j]))
             for i, j in itertools.combinations(range(4), 2)]
    fitted = jev.bradley_terry(4, pairs)
    assert fitted == pytest.approx([s / sum(strengths) for s in strengths], abs=0.006)
    assert fitted[3] > fitted[2] > fitted[1] > fitted[0]
    # Renaming/reversing the comparisons cannot change the ranking.
    assert jev.bradley_terry(4, [(j, i, 1 - p) for i, j, p in pairs]) == pytest.approx(fitted)


def test_ties_cycles_and_unanimous_preferences_remain_finite():
    assert jev.bradley_terry(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)]) == pytest.approx([1/3] * 3)
    assert jev.bradley_terry(2, [(0, 1, 0.5)]) == pytest.approx([0.5, 0.5])
    out = jev.bradley_terry(2, [(0, 1, 1)])
    assert 0 < out[1] < out[0] < 1
    assert sum(out) == pytest.approx(1)
    assert jev.bradley_terry(0, []) == []
    assert jev.bradley_terry(1, []) == [1]


def test_batched_page_choices_cover_every_pair_once_and_keep_blank_positions(monkeypatch):
    monkeypatch.setattr(jev, 'PAIR_BATCH', 2)
    calls = []
    weights = {'0': 1, '2': 2, '3': 4}

    def evaluate(state, questions, report, kind):
        calls.append(questions)
        assert state['pages'] == {'0': 'first', '2': 'second', '3': 'third'}
        out = {}
        for key, q in questions.items():
            if q['type'] == 'noul':
                out[key] = {'noul': 0.02}
            else:
                a, b = q['criteria']
                assert f'pages.{a}' in q['criteria'][a]
                p = weights[a] / (weights[a] + weights[b])
                out[key] = {'choice': a if p > 0.5 else b,
                            'probabilities': {a: p, b: 1 - p}}
        return out

    monkeypatch.setattr(jev, 'evaluate', evaluate)
    report = jev.Report()
    scores, found = jev.score_pages('q', ['first', ' ', 'second', 'third'], report)
    assert scores[1] is None and scores[3] > scores[2] > scores[0]
    assert found == 0.02  # no absolute relevance inferred from relative strength
    assert len(calls) == 2
    assert sum(jev.ANSWERABLE in call for call in calls) == 1
    assert report.comparisons == 3 and report.scored == 3 and report.blind == 1
    assert report.as_dict()['ranking_method'] == 'choice-bradley-terry'


@pytest.mark.parametrize('answer', [
    {'choice': '0', 'probabilities': {'0': .7, '1': .7}},
    {'choice': '0', 'probabilities': {'0': 0, '1': 0}},
    {'choice': '2', 'probabilities': {'0': .5, '1': .5}},
    {'choice': '0', 'probabilities': {'0': .5}},
    {'choice': '0', 'probabilities': {'0': True, '1': 0}},
    {'choice': '0', 'probabilities': {'0': float('nan'), '1': .5}},
])
def test_malformed_distribution_cannot_publish_a_ranking(monkeypatch, answer):
    monkeypatch.setattr(jev, 'evaluate', lambda *a: {'pair_0_1': answer})
    report = jev.Report()
    with pytest.raises((ValueError, KeyError)):
        jev.score_pages('q', ['a', 'b'], report)
    assert not report.reranked and report.ranking_method is None


def test_later_batch_failure_is_atomic(monkeypatch):
    monkeypatch.setattr(jev, 'PAIR_BATCH', 1)
    def evaluate(state, questions, report, kind):
        if 'pair_0_2' in questions:
            raise ValueError('missing pair')
        return {'pair_0_1': {'choice': '0', 'probabilities': {'0': .8, '1': .2}}}
    monkeypatch.setattr(jev, 'evaluate', evaluate)
    report = jev.Report()
    with pytest.raises(ValueError):
        jev.score_pages('q', ['a', 'b', 'c'], report)
    assert not report.reranked
