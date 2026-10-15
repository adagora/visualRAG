import json
from types import SimpleNamespace

import pytest

import ablate


class Process:
    def __init__(self):
        self.stopped = False
    def poll(self):
        return 0 if self.stopped else None
    def terminate(self):
        self.stopped = True
    def wait(self, timeout):
        return 0


def test_shared_encoder_reused_without_model_process(monkeypatch):
    identity = {k: None for k in ablate.ENCODER_FIELDS}
    identity.update(model="qwen", dimensions=2048, status="ok")
    monkeypatch.setattr(ablate, "health", lambda url: identity)
    monkeypatch.setattr(ablate.subprocess, "Popen", lambda *a, **k: pytest.fail("must reuse the existing encoder"))
    assert ablate.ensure_encoder("http://127.0.0.1:8001", "unused", {"encoder": identity}, None) is None


def test_encoder_mismatch_is_hard_failure(monkeypatch):
    monkeypatch.setattr(ablate, "health", lambda url: {"status": "ok", "model": "wrong"})
    with pytest.raises(ValueError, match="identity mismatch"):
        ablate.ensure_encoder("http://127.0.0.1:8001", "unused", {"encoder": {"model": "right"}}, None)


def test_execute_owns_each_geometry_service_and_environment(monkeypatch, tmp_path):
    processes, envs = [], []
    indexes = {g: tmp_path / g for g in ("page", "regions")}
    manifests = {g: {"index_id": g} for g in indexes}
    runs = [{"geometry": g, "aggregation": a, "command": [g, a]} for g in indexes for a in ("max", "sum")]
    monkeypatch.setattr(ablate, "ensure_encoder", lambda *args: None)
    monkeypatch.setattr(ablate, "free_port", lambda p: p)
    monkeypatch.setattr(ablate, "wait_ready", lambda *a, **kw: None)
    def popen(command, **kwargs):
        p = Process()
        processes.append(p)
        return p
    monkeypatch.setattr(ablate.subprocess, "Popen", popen)
    monkeypatch.setattr(ablate.subprocess, "run", lambda cmd, **kwargs: envs.append(kwargs["env"]))
    monkeypatch.setattr(ablate, "check_report", lambda *args: {})
    ablate.execute(runs, indexes, manifests, tmp_path)
    assert len(processes) == 2 and all(p.stopped for p in processes)
    assert [e["PIXELRAG_INDEX_DIR"] for e in envs] == [str(indexes["page"])] * 2 + [str(indexes["regions"])] * 2
    assert all(e["PIXELRAG_SEARCH_API"] == "http://127.0.0.1:30002" for e in envs)
    assert all(e["PIXELRAG_ENCODER_URL"] == "http://127.0.0.1:8001" for e in envs)
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "valid"


def test_error_report_marks_execution_invalid_and_cleans_owned(monkeypatch, tmp_path):
    search, encoder = Process(), Process()
    monkeypatch.setattr(ablate, "ensure_encoder", lambda *args: encoder)
    monkeypatch.setattr(ablate, "free_port", lambda p: p)
    monkeypatch.setattr(ablate, "wait_ready", lambda *a, **kw: None)
    monkeypatch.setattr(ablate.subprocess, "Popen", lambda *a, **kw: search)
    monkeypatch.setattr(ablate.subprocess, "run", lambda *a, **kw: None)
    def fail(*args):
        raise RuntimeError("retrieval failed")
    monkeypatch.setattr(ablate, "check_report", fail)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        ablate.execute([{"geometry": "page", "aggregation": "max", "command": []}],
                       {"page": tmp_path}, {"page": {"index_id": "page"}}, tmp_path)
    assert search.stopped and encoder.stopped
    assert json.loads((tmp_path / "execution.json").read_text())["status"] == "invalid"


def test_check_report_rejects_errors_and_wrong_snapshot(tmp_path):
    path = tmp_path / "run.json"
    report = {"provenance": {"index_fingerprint": "same"}, "rows": [{"error": "down"}],
              "summary": {"visual": {}, "lexical": {}, "hybrid": {}}}
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="Invalid geometry"):
        ablate.check_report(path, "same")
    with pytest.raises(ValueError, match="another snapshot"):
        ablate.check_report(path, "other")


def test_report_fingerprint_and_manifest_id_are_separate(tmp_path):
    path = tmp_path / "run.json"
    report = {"provenance": {"index_fingerprint": "content-fingerprint", "index_manifest": {"index_id": "snapshot-id"}},
              "rows": [{"error": None}], "summary": {"visual": {}, "lexical": {}, "hybrid": {}}}
    path.write_text(json.dumps(report))
    assert ablate.check_report(path, "snapshot-id", "content-fingerprint")
    with pytest.raises(ValueError, match="another snapshot"):
        ablate.check_report(path, "snapshot-id", "changed-content")


def test_plan_only_never_starts_services(monkeypatch, tmp_path):
    monkeypatch.setattr(ablate, "validate", lambda indexes: {g: {} for g in indexes})
    monkeypatch.setattr(ablate, "execute", lambda *a: pytest.fail("plan must not execute"))
    monkeypatch.setattr(ablate.sys, "argv", ["ablate", "--page", "a", "--regions", "b", "--multiscale", "c", "--output", str(tmp_path)])
    ablate.main()
    assert len(json.loads((tmp_path / "plan.json").read_text())["runs"]) == 6
