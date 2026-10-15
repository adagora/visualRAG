"""Gold-page evaluation must not score wrong sources or hide failures."""
import json
from types import SimpleNamespace
import pytest
import evaluate

@pytest.fixture
def study(tmp_path):
    digest = "a" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "index_id": "test", "documents": [{
        "sha256": digest, "source_id": "indexed-source", "article_id": 0, "pages": 5}]}))
    dataset = {"sources": [{"sha256": digest, "source_id": "label-source"}], "questions": [
        {"id": "a", "q": "A", "source_id": "label-source", "answerable": True, "gold": [{"page": 2}]},
        {"id": "b", "q": "B", "source_id": "label-source", "answerable": True, "gold": [{"page": 3}]},
        {"id": "n", "q": "Missing", "source_id": "label-source", "answerable": False, "gold": []},
    ]}
    return tmp_path, dataset

def test_source_mismatch_prevents_billed_search(study):
    path, dataset = study
    dataset["sources"][0]["sha256"] = "b" * 64
    index = SimpleNamespace(snapshot_dir=path, search=lambda *a, **k: pytest.fail("Wrong-source call"))
    with pytest.raises(ValueError, match="hash"): evaluate.run(index, dataset)

def test_failures_remain_misses_and_negatives_are_skipped(study):
    path, dataset = study
    called = []
    def search(question, limit, image=None):
        called.append(question)
        if question == "B": raise RuntimeError("API error")
        return [{"source_id": "indexed-source", "page": 2}]
    report = evaluate.run(SimpleNamespace(snapshot_dir=path, search=search), dataset)
    assert called == ["A", "B"]
    assert report["positive_questions"] == 2
    assert report["evidence_first"] == report["evidence_within_four"] == 1
    assert report["errors"] == 1

def test_same_page_from_another_source_is_a_miss(study):
    path, dataset = study
    index = SimpleNamespace(snapshot_dir=path, search=lambda *a, **k: [{"source_id": "wrong", "page": 2}])
    assert evaluate.run(index, dataset)["evidence_within_four"] == 0


@pytest.fixture
def image_study(study, tmp_path):
    from PIL import Image
    path, _ = study
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (1000, 1000), "blue").save(image_path)
    dataset = {"source_sha256": "a" * 64, "limit": "Self-crops only", "image_queries": [
        {"id": "image-1", "page": 2, "rect": [.1, .2, .5, .8], "gold_page": 2}]}
    calls = []
    def search(question, *, image=None, limit=4):
        calls.append((question, image, limit))
        return [{"article_id": 0, "source_id": "indexed-source", "document": "Doc", "page": n, "score": .9}
                for n in range(1, 6)]
    index = SimpleNamespace(snapshot_dir=path, page_path=lambda aid, page: image_path, search=search)
    return index, dataset, calls


def test_image_crop_validation_is_offline_and_execution_embeds_image(image_study):
    import io
    from PIL import Image
    index, dataset, calls = image_study
    assert len(evaluate.prepare(index, dataset)) == 1
    assert calls == []
    report = evaluate.run(index, dataset)
    assert calls[0][0] == "" and calls[0][1].startswith(b"\x89PNG")
    with Image.open(io.BytesIO(calls[0][1])) as image:
        assert image.size == (400, 600)
    assert report["evidence_within_four"] == 1 and report["dataset_limit"] == "Self-crops only"


@pytest.mark.parametrize("rect", [[.5, 0, .1, 1], [0, 0, 2, 1], [0, 0, 1], [0, True, 1, 1],
                                   [0, 0, float("nan"), 1]])
def test_invalid_crops_fail_before_embedding(image_study, rect):
    index, dataset, calls = image_study
    dataset["image_queries"][0]["rect"] = rect
    with pytest.raises(ValueError, match="rectangle"):
        evaluate.run(index, dataset)
    assert calls == []


def test_compare_shares_one_embedding_and_reports_fallbacks(image_study, monkeypatch):
    index, dataset, calls = image_study
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    report = evaluate.run(index, dataset, compare_jev=True)
    assert len(calls) == 1 and calls[0][2] == 8
    assert report["baseline"]["evidence_within_four"] == report["evidence_within_four"] == 1
    assert report["jev"]["fallbacks"] == 1 and report["jev"]["applied"] == 0


def test_partial_evidence_does_not_count_as_complete(study):
    path, dataset = study
    dataset["questions"][0]["gold"].append({"page": 4})
    index = SimpleNamespace(snapshot_dir=path, search=lambda *a, **k: [{"source_id": "indexed-source", "page": 2}])
    report = evaluate.run(index, dataset)
    row = report["rows"][0]
    assert row["hit_within_four"] and not row["all_evidence_within_four"]
    assert row["evidence_recall"] == .5


def test_successful_comparison_preserves_dense_baseline(image_study, monkeypatch):
    import jev
    index, dataset, calls = image_study
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    response = {"model": jev.MODEL, "usage": {"input_tokens": 100, "output_tokens": 5},
                "answers": {f"page_{i}": {"type": "noul", "noul": 1.0 if i == 1 else .1} for i in range(5)}}
    monkeypatch.setattr(jev, "_request", lambda payload, key: response)
    report = evaluate.run(index, dataset, compare_jev=True)
    assert len(calls) == 1
    assert report["evidence_first"] == 1 and report["baseline"]["evidence_first"] == 0
    assert report["baseline"]["rows"][0]["pages"][0]["page"] == 1
    assert report["rows"][0]["pages"][0]["page"] == 2
    assert report["jev"]["applied"] == 1 and report["jev"]["fallbacks"] == 0
