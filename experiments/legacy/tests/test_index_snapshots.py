import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import build_index
import chunk_multiscale as chunks
import index_manifest as manifests
import layout
import prepare_hires


def test_publish_and_resolve_safe_pointer(tmp_path, monkeypatch):
    old = tmp_path / "articles.json"
    old.write_text("legacy")
    stage = manifests.begin_snapshot(tmp_path, "snapshot-a")
    manifest = {"schema_version": 1, "index_id": "snapshot-a", "documents": []}
    final = manifests.publish_snapshot(tmp_path, stage, manifest)
    monkeypatch.setenv("PIXELRAG_INDEX_DIR", str(tmp_path))
    assert layout.default().index_dir == final
    assert old.read_text() == "legacy"
    before = manifests.fingerprint(final)
    assert len(before) == 64
    (final / "text.json").write_text("new text sidecar")
    assert manifests.fingerprint(final) != before
    (tmp_path / "active.json").write_text(json.dumps({"schema_version": 1, "index_id": "outside", "path": "../outside"}))
    with pytest.raises(ValueError):
        layout.default()


def test_pointer_replace_failure_retains_previous_active(tmp_path, monkeypatch):
    pointer = tmp_path / "active.json"
    pointer.write_text("previous active")
    stage = manifests.begin_snapshot(tmp_path, "new-snapshot")
    def fail(*args):
        raise OSError("pointer replace failed")
    monkeypatch.setattr(manifests.os, "replace", fail)
    with pytest.raises(OSError, match="pointer replace"):
        manifests.publish_snapshot(tmp_path, stage, {"schema_version": 1, "index_id": "new-snapshot"})
    assert pointer.read_text() == "previous active"
    assert (tmp_path / "snapshots" / "new-snapshot" / "manifest.json").is_file()


def test_same_path_pdf_change_is_stale(tmp_path):
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"before")
    directory = tmp_path / "tiles"
    directory.mkdir()
    import hashlib
    (directory / "tiles.json").write_text(json.dumps({"source": str(pdf), "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}))
    assert not prepare_hires._is_stale(directory, pdf)
    pdf.write_bytes(b"after")
    assert prepare_hires._is_stale(directory, pdf)


def test_render_sources_anchored_to_root_from_other_cwd(tmp_path, monkeypatch):
    import sys
    root = tmp_path / "root"
    (root / "pdfs").mkdir(parents=True)
    (root / "pdfs" / "one.pdf").write_bytes(b"pdf fixture")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(layout, "ROOT", root)
    calls = []
    def render(pdf, output, stem):
        calls.append(pdf)
        target = layout.IndexLayout(Path(output).parent).tile_dir(int(stem))
        target.mkdir(parents=True)
        (target / "tiles.json").write_text(json.dumps({"source": pdf}))
        Image.new("RGB", (100, 100), "black").save(target / "tile_0000.jpg")
    from pathlib import Path
    monkeypatch.setitem(sys.modules, "pixelrag_render.backends.pdf", SimpleNamespace(render_pdf=render))
    docs = prepare_hires.prepare({"source": {"path": "pdfs"}, "output": "new-index"})
    assert calls == [str(root / "pdfs" / "one.pdf")]
    assert docs[0]["pages"] == 1
    assert (root / "new-index" / "articles.json").is_file()
    original = root / "pdfs" / "one.pdf"
    original.write_bytes(b"replacement PDF")
    snapshot = root / "new-index" / docs[0]["snapshot_path"]
    assert snapshot.read_bytes() == b"pdf fixture"
    articles = json.loads((root / "new-index" / "articles.json").read_text())
    assert articles[0]["url"] == docs[0]["snapshot_path"]


@pytest.mark.parametrize("mode", ["page", "regions", "multiscale"])
def test_dry_run_matches_real_filtering_and_removes_old_chunks(tmp_path, mode):
    image = Image.new("RGB", (240, 260), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 100, 100), fill="black")
    image.save(tmp_path / "tile_0000.jpg")
    kwargs = dict(mode=mode, width=120, height=130, stride_x=100, stride_y=100)
    dry = chunks.process_article(tmp_path, 0, "doc", dry_run=True, **kwargs)
    assert not (tmp_path / "chunks.json").exists()
    (tmp_path / "chunk_old.png").write_bytes(b"old")
    real = chunks.process_article(tmp_path, 0, "doc", **kwargs)
    assert dry == real
    assert not (tmp_path / "chunk_old.png").exists()
    assert real["num_chunks"] == len(list(tmp_path.glob("chunk_*.png")))


def test_configured_model_gpu_and_no_resume(tmp_path, monkeypatch):
    calls = []
    def run(command, check):
        calls.append(command)
        if "pixelrag_embed.embed" in command:
            out = command[command.index("--output-dir") + 1]
            np.savez(str(out) + "/shard_000.npz", embeddings=np.ones((2, 3)))
    monkeypatch.setattr(build_index.subprocess, "run", run)
    assert build_index.snapshot_embed(layout.IndexLayout(tmp_path), {"gpu_ids": [1, 2], "batch_size": 4}, "pinned-model") == (2, 3)
    assert calls[0][calls[0].index("--model") + 1] == "pinned-model"
    assert calls[0][calls[0].index("--gpu-ids") + 1] == "1,2"
    assert calls[0][calls[0].index("--instruction") + 1] == "Represent the user's input."
    assert "--resume" not in calls[0]


def test_build_failure_does_not_replace_active(tmp_path, monkeypatch):
    pointer = tmp_path / "active.json"
    pointer.write_text("existing pointer")
    monkeypatch.setattr(build_index, "resolve_model", lambda *args: ("local", "commit"))
    monkeypatch.setattr(prepare_hires, "prepare", lambda cfg: [])
    def fail(*args):
        raise RuntimeError("embedding failed")
    monkeypatch.setattr(build_index, "snapshot_embed", fail)
    with pytest.raises(RuntimeError, match="embedding failed"):
        build_index.build({"output": str(tmp_path)}, {"mode": "page"})
    assert pointer.read_text() == "existing pointer"
    assert not list((tmp_path / "snapshots").iterdir())


@pytest.mark.parametrize("embed", [{"adapter": "lora"}, {"device": "cpu"}, {"backend": "sglang"}])
def test_reject_wrong_config_before_model_download(embed, monkeypatch):
    monkeypatch.setattr(build_index, "resolve_model", lambda *args: pytest.fail("Must reject before model resolution"))
    with pytest.raises(ValueError):
        build_index.build({"embed": embed}, {"mode": "page"})


def test_vector_validation_rejects_removed_chunks(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(read_index=lambda p: SimpleNamespace(ntotal=1, d=3)))
    idx = layout.IndexLayout(tmp_path)
    idx.tile_dir(0).mkdir(parents=True)
    Image.new("RGB", (100, 100), "black").save(idx.page_image(0, 0))
    chunks.process_article(idx.tile_dir(0), 0, "doc", mode="page")
    idx.articles_json.write_text("[{}]")
    np.savez(tmp_path / "metadata.npz", article_ids=[0], tile_indices=[0], chunk_indices=[99])
    with pytest.raises(ValueError, match="Vectors and chunks"):
        build_index.validate_snapshot(idx, [{"article_id": 0, "pages": 1}], 1, 3)
