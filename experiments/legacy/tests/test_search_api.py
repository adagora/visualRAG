import json

import faiss
import numpy as np
import pytest
from fastapi import HTTPException

import layout
import search_api


@pytest.fixture
def service(tmp_path):
    idx = layout.IndexLayout(tmp_path)
    vectors = faiss.IndexFlatIP(3)
    vectors.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    faiss.write_index(vectors, str(idx.faiss_index))
    np.savez(tmp_path / "metadata.npz", article_ids=[7, 8], tile_indices=[2, 3], chunk_indices=[4, 5])
    manifest = {"schema_version": 1, "index_id": "test-snapshot", "vector_count": 2,
                "encoder": {"backend": "qwen", "dimensions": 3}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    search_api._state.clear()
    search_api.load(idx)
    yield idx
    search_api._state.clear()


def request(*vectors, n_docs=2):
    return search_api.SearchRequest(queries=[search_api.VectorQuery(embedding=v) for v in vectors], n_docs=n_docs)


def test_search_normalizes_ranks_and_maps_metadata(service):
    result = search_api.search(request([0, 5, 0]))
    hits = result["results"][0]["hits"]
    assert hits[0] == {"score": 1.0, "vector_id": 1, "article_id": 8, "tile_index": 3, "chunk_index": 5}
    assert hits[1]["article_id"] == 7
    assert len(search_api.search(request([1, 0, 0], n_docs=100))["results"][0]["hits"]) == 2


@pytest.mark.parametrize("vectors", [([1, 2],), ([0, 0, 0],), ([float("nan"), 0, 0],),
                                      ([float("inf"), 0, 0],), ([1, 0, 0], [1, 0])])
def test_invalid_embeddings_rejected(service, vectors):
    with pytest.raises(HTTPException) as error:
        search_api.search(request(*vectors))
    assert error.value.status_code == 400


def test_health_reports_exact_snapshot_identity(service):
    health = search_api.health()
    assert health["status"] == "ok"
    assert health["index_id"] == "test-snapshot"
    assert health["dimensions"] == 3


def test_reject_manifest_vector_count_mismatch(service):
    path = service.index_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["vector_count"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        search_api.load(service)


def test_missing_tile_and_negative_identity_return_404(service):
    for identity in [(7, 2, 4), (-1, 0, 0)]:
        with pytest.raises(HTTPException) as error:
            search_api.tile(*identity)
        assert error.value.status_code == 404


def test_existing_tile_returns_snapshot_file(service):
    folder = service.tile_dir(7)
    folder.mkdir(parents=True)
    path = folder / "chunk_0002_04.png"
    path.write_bytes(b"image fixture")
    assert search_api.tile(7, 2, 4).path == path
