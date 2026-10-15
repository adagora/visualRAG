import json

import pytest

import encoder_contract as contract
import layout


@pytest.fixture
def identity(tmp_path, monkeypatch):
    for name in ("PIXELRAG_EMBED_MODEL", "PIXELRAG_ADAPTER"):
        monkeypatch.delenv(name, raising=False)
    encoder = {"backend": "qwen", "model": "configured-model", "revision": "pinned-commit",
               "adapter": None, "dimensions": 3, "instruction": contract.DEFAULT_INSTRUCTION}
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "index_id": "index-a", "encoder": encoder}))
    monkeypatch.setattr(layout, "DEFAULT", layout.IndexLayout(tmp_path))
    return contract.identity()


def test_identity_uses_recorded_model_revision_and_dimensions(identity):
    assert identity["model"] == "configured-model"
    assert identity["revision"] == "pinned-commit"
    assert identity["dimensions"] == 3
    assert identity["index_id"] == "index-a"
    contract.validate(dict(identity), identity)


@pytest.mark.parametrize("field,value", [("model", "wrong-model"), ("revision", "wrong-commit"),
                                         ("adapter", "lora"), ("dimensions", 4), ("instruction", "wrong")])
def test_actual_encoder_mismatch_fails_explicitly(identity, field, value):
    actual = {**identity, field: value}
    with pytest.raises(contract.EncoderMismatch, match=field):
        contract.validate(actual, identity)


def test_model_override_cannot_silently_fallback(identity, monkeypatch):
    monkeypatch.setenv("PIXELRAG_EMBED_MODEL", "different-model")
    with pytest.raises(contract.EncoderMismatch, match="MODEL"):
        contract.identity()


def test_adapter_environment_rejected(identity, monkeypatch):
    monkeypatch.setenv("PIXELRAG_ADAPTER", "lora")
    with pytest.raises(contract.EncoderMismatch, match="Adapter"):
        contract.identity()


def test_adapter_manifest_rejected(identity):
    path = layout.DEFAULT.index_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["encoder"]["adapter"] = "lora"
    path.write_text(json.dumps(manifest))
    with pytest.raises(contract.EncoderMismatch, match="Adapter"):
        contract.identity()


def test_missing_revision_is_mismatch_for_pinned_snapshot(identity):
    actual = dict(identity)
    actual.pop("revision")
    with pytest.raises(contract.EncoderMismatch, match="revision"):
        contract.validate(actual, identity)


@pytest.mark.parametrize("previously_available", [None, True])
def test_incompatible_sidecar_cannot_be_cached_as_available(identity, monkeypatch, previously_available):
    import queryembed
    from types import SimpleNamespace
    response = SimpleNamespace(ok=True, json=lambda: {"status": "ok", **identity, "model": "wrong-model"})
    monkeypatch.setattr(queryembed._HTTP, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(queryembed, "_sidecar_ok", previously_available)
    monkeypatch.setattr(queryembed, "_sidecar_checked", 0.0)
    for _ in range(2):
        with pytest.raises(contract.EncoderMismatch, match="model"):
            queryembed._sidecar_available()
