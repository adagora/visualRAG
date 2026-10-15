"""Evaluation failures must not improve metrics; no model or services required."""
import copy
import json
from types import SimpleNamespace

import pytest

import answer_eval
import visual_bench as vb
import ablate


def question():
    return {"id": "q1", "q": "price?", "type": "table_intersection", "visual_dependency": "required",
            "split": "test", "page_family": "prices", "source_id": "s", "answerable": True,
            "gold": [{"source_id": "s", "page": 2, "rectangles": [[0, 0, 1, 1]]},
                     {"source_id": "s", "page": 3, "rectangles": [[0, 0, 1, 1]]}],
            "expected_answer": {"facts": [{"kind": "number", "value": 100, "unit": "PLN"}]},
            "review": {"status": "agent_inspected"}}


def test_any_gold_and_complete_are_distinct():
    q = question()
    result = vb.score(q, [[0, 3], [0, 3], [0, 8]], {"s": 0})
    assert result["hit@1"] and result["hit@4"]
    assert result["mrr@4"] == 1
    assert not result["complete_evidence@4"]
    assert result["covered_pages"] == 1
    assert vb.score(q, [[0, 8], [0, 2], [0, 3]], {"s": 0})["complete_evidence@4"]


def test_errors_stay_in_denominator():
    def broken(*args, **kwargs):
        raise RuntimeError("down")
    q = question()
    failed = vb.measure(q, "visual", 24, retriever=broken)
    row = {**failed, "mode": "visual", "phase": "warm", "question_id": "q1",
           "metrics": vb.score(q, failed["ranking"], {"s": 0})}
    out = vb.summarize([row], [q])["visual"]
    assert out["positive_trials"] == 1 and out["errors"] == 1
    assert out["hit@4"] == out["mrr@4"] == 0


def test_same_page_and_candidate_budget():
    calls = []
    def fake(q, mode, **kw):
        calls.append(kw)
        return [SimpleNamespace(article_id=0, page=2)], [], {}
    for m in vb.MODES:
        assert not vb.measure(question(), m, 24, retriever=fake)["error"]
    assert calls == [{"n_pages": 24, "per_query": 24, "aggregation": "max"}] * 3


def test_shortlist_mrr_retains_below_reader_cutoff():
    result = vb.score(question(), [[0, 8], [0, 9], [0, 10], [0, 11], [0, 2]], {"s": 0})
    assert not result["hit@4"]
    assert result["mrr@4"] == 0
    assert result["mrr@candidate_budget"] == .2


def test_review_and_family_leak_preflight(tmp_path):
    path = tmp_path / "q.json"
    data = {"schema_version": 1, "review_status": "agent_inspected_draft", "questions": [question()]}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="human audit pending"):
        vb.load_dataset(path)
    assert vb.load_dataset(path, allow_draft=True)
    other = copy.deepcopy(question())
    other.update(id="q2", split="dev")
    data["questions"].append(other)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leaks"):
        vb.load_dataset(path, allow_draft=True)


def test_missing_source_is_not_silently_excluded(tmp_path):
    (tmp_path / "articles.json").write_text("[]")
    data = {"sources": [{"source_id": "s", "path": "missing.pdf", "sha256": "x", "page_count": 3}],
            "questions": [question()]}
    with pytest.raises(ValueError, match="source missing from index"):
        vb.preflight(data, tmp_path)


def test_full_source_sha_maps_to_abbreviated_snapshot_source_id(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"placeholder PDF content")
    sha = vb.digest(path)
    (tmp_path / "articles.json").write_text(json.dumps([{"url": str(path), "title": "source"}]))
    manifest = {"schema_version": 1, "index_id": "snapshot", "documents": [
        {"article_id": 0, "source_id": sha[:16], "sha256": sha, "pages": 3}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    q = question()
    q["source_id"] = sha
    for g in q["gold"]:
        g["source_id"] = sha
    data = {"sources": [{"source_id": sha, "path": str(path), "sha256": sha, "page_count": 3}], "questions": [q]}
    assert vb.preflight(data, tmp_path) == {sha: 0}
    manifest["documents"][0]["sha256"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Snapshot source hash"):
        vb.preflight(data, tmp_path)


def test_snapshot_copy_is_authoritative_when_original_changes_or_disappears(tmp_path):
    original = tmp_path / "original.pdf"
    original.write_bytes(b"indexed content version")
    sha = vb.digest(original)
    root = tmp_path / "index"
    snapshot = root / "snapshots" / "version1"
    (snapshot / "sources").mkdir(parents=True)
    copied = snapshot / "sources" / f"{sha}.pdf"
    copied.write_bytes(original.read_bytes())
    (root / "active.json").write_text(json.dumps({"schema_version":1, "index_id": "version1", "path": "snapshots/version1"}))
    (snapshot / "articles.json").write_text(json.dumps([{"url": f"sources/{sha}.pdf"}]))
    manifest = {"schema_version": 1, "index_id": "version1", "documents": [{"article_id": 0,
        "source_id": sha[:16], "sha256": sha, "pages": 3, "path": str(original), "snapshot_path": f"sources/{sha}.pdf"}]}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    q = question()
    q["source_id"] = sha
    for g in q["gold"]:
        g["source_id"] = sha
    data = {"sources": [{"source_id": sha, "path": str(original), "sha256": sha, "page_count": 3}], "questions": [q]}
    original.write_bytes(b"new unrelated version")
    assert vb.preflight(data, root) == {sha: 0}
    original.unlink()
    assert vb.preflight(data, root) == {sha: 0}
    copied.write_bytes(b"corrupt copied content")
    with pytest.raises(ValueError, match="content changed"):
        vb.preflight(data, root)


def test_numeric_unit_and_entity_boundaries():
    expected = {"facts": [{"kind": "number", "value": 469, "unit": "PLN"},
                           {"kind": "entity", "value": "METRO 1000"}]}
    assert answer_eval.check_facts("METRO 1000 kosztuje 469 PLN", expected)["all_expected_facts_present"]
    for answer in ("METRO 1000 kosztuje 1469 PLN", "METRO 1000 kosztuje 469 EUR", "METRO 10000 kosztuje 469 PLN"):
        assert not answer_eval.check_facts(answer, expected)["all_expected_facts_present"]
    assert answer_eval.check_facts("0,48 W/m²K", {"facts": [{"kind": "number", "value": .48, "unit": "W/m2K"}]})["all_expected_facts_present"]
    assert answer_eval.check_facts("426 × 89 mm", {"facts": [{"kind": "dimensions", "values": [426, 89], "unit": "mm"}]})["all_expected_facts_present"]


def test_dataset_answers_match_declared_fact_screen():
    data = vb.load_dataset(vb.DEFAULT_EVAL, allow_draft=True)
    assert len(data["questions"]) == 50
    for q in data["questions"]:
        if q["answerable"]:
            assert answer_eval.check_facts(q["expected_answer"]["text"], q["expected_answer"])["all_expected_facts_present"], q["id"]


def test_currency_aliases_do_not_change_currency_or_number():
    facts = {'facts':[{'kind':'number','value':4349,'unit':'PLN'}]}
    assert answer_eval.check_facts('Cena netto wynosi 4 349 zł.', facts)['all_expected_facts_present']
    assert not answer_eval.check_facts('Cena wynosi 43 490 zł.', facts)['all_expected_facts_present']
    assert not answer_eval.check_facts('Cena wynosi 4349 EUR.', facts)['all_expected_facts_present']


def test_abstention_screen_is_separate_from_positive_facts():
    q = {'answerable':False, 'expected_answer':{'facts':[]}}
    assert answer_eval.screen_answer('W dokumentach nie podano ceny abonamentu.',q)['abstention_screen_pass']
    assert not answer_eval.screen_answer('Nie ma problemu, cena wynosi 99 PLN.',q)['abstention_screen_pass']
    positive = {'answerable':True,'expected_answer':{'facts':[{'kind':'number','value':40,'unit':'mm'}]}}
    result=answer_eval.screen_answer('Nie mogę ustalić grubości na podstawie tych stron.',positive)
    assert result['possible_false_refusal'] and result['manual_review_required']


def test_geometry_requires_matching_encoder(tmp_path):
    indexes = {}
    for g in ("page", "regions", "multiscale"):
        p = tmp_path / g
        p.mkdir()
        indexes[g] = p
        m = {"schema_version": 1, "index_id": g, "encoder": {"model": "same", "revision": "r1"},
             "render": {"dpi": 200}, "chunking": {"mode": g, "width": 875, "height": 1024},
             "documents": [{"source_id": "s", "sha256": "s", "pages": 3}]}
        (p / "manifest.json").write_text(json.dumps(m))
    assert len(ablate.validate(indexes)) == 3
    m["encoder"]["revision"] = "r2"
    (indexes["multiscale"] / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="encoder"):
        ablate.validate(indexes)


def test_scan_fixture_has_no_text_and_keeps_page_map(tmp_path):
    import pymupdf
    from scan_fixture import flatten
    src = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        for text in ("First text page", "Second text page"):
            p = doc.new_page()
            p.insert_text((72, 72), text)
        doc.save(src)
    dest = tmp_path / "scan.pdf"
    manifest = flatten(src, [2, 1], dest, dpi=72)
    assert manifest["textless_verified"]
    assert manifest["page_map"] == [{"fixture_page": 1, "source_page": 2}, {"fixture_page": 2, "source_page": 1}]
    with pymupdf.open(dest) as doc:
        assert all(not page.get_text() for page in doc)
    with pytest.raises(ValueError, match="new file"):
        flatten(src, [1], dest)
