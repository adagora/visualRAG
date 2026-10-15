#!/usr/bin/env python
"""Validate/execute a same-encoder page/regions/multiscale x max/sum study.

Build three immutable snapshots with build_index.py --geometry, keeping model,
revision, corpus, render and crop dimensions/stride fixed. Provide their roots
below. By default only validates; --execute runs local retrieval benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

import index_manifest
from visual_bench import DEFAULT_EVAL, load_dataset, preflight, resolve_index


ENCODER_FIELDS = ("model", "revision", "dimensions", "instruction", "adapter", "backend")


def health(url):
    try:
        response = requests.get(url.rstrip("/") + "/health", timeout=2)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def wait_ready(url, expected, process=None, timeout=180):
    deadline = time.monotonic() + timeout
    announced = 0.0
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owned service exited before readiness: {url}")
        status = health(url)
        if status and status.get("status") == "ok":
            for key, value in expected.items():
                if status.get(key) != value:
                    raise ValueError(f"Service identity mismatch at {url}: {key}")
            return status
        now = time.monotonic()
        if now - announced >= 15:
            print(f"Waiting for owned service readiness: {url}", file=sys.stderr, flush=True)
            announced = now
        time.sleep(.5)
    raise TimeoutError(f"Service did not become ready within {timeout}s: {url}")


def free_port(preferred):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def ensure_encoder(url, snapshot, manifest, log):
    expected = {field: manifest["encoder"].get(field) for field in ENCODER_FIELDS}
    existing = health(url)
    if existing:
        wait_ready(url, expected, timeout=180)
        return None
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path not in ("", "/"):
        raise ValueError("Cannot start an encoder for a non-local URL; start that service separately")
    port = parsed.port or 80
    if free_port(port) != port:
        raise RuntimeError("Encoder port is occupied by an unavailable service; no duplicate model was started")
    env = {**os.environ, "PIXELRAG_INDEX_DIR": str(resolve_index(snapshot)), "PIXELRAG_ENCODER_PORT": str(port)}
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("encoder.py"))], env=env,
                               stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        wait_ready(url, expected, process)
    except BaseException:
        stop_owned(process)
        raise
    return process


def check_report(path, expected_index_id, expected_fingerprint=None):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    actual_id = (provenance.get("index_manifest") or {}).get("index_id", provenance.get("index_fingerprint"))
    if actual_id != expected_index_id or (expected_fingerprint is not None and provenance.get("index_fingerprint") != expected_fingerprint):
        raise ValueError("Benchmark report belongs to another snapshot")
    rows = report.get("rows", [])
    failures = [r for r in rows if r.get("error")]
    failures.extend(r for r in report.get("warmups", {}).values() if r.get("error"))
    if not rows or failures or set(report.get("summary", {})) != {"visual", "lexical", "hybrid"}:
        raise RuntimeError(f"Invalid geometry run: empty/incomplete report or {len(failures)} errors: {path}")
    return report


def execute(runs, indexes, manifests, output, search_port=30002, encoder_url="http://127.0.0.1:8001"):
    """Each geometry owns its FAISS server; an existing compatible encoder is reused."""
    output = Path(output)
    status = {"status": "running", "completed": []}
    encoder = None
    try:
        with (output / "encoder-service.log").open("a", encoding="utf-8") as encoder_log:
            first = next(iter(indexes))
            encoder = ensure_encoder(encoder_url, indexes[first], manifests[first], encoder_log)
            for geometry, index in indexes.items():
                port = free_port(search_port)
                url = f"http://127.0.0.1:{port}"
                snapshot = resolve_index(index)
                env = {**os.environ, "PIXELRAG_INDEX_DIR": str(snapshot), "PIXELRAG_SEARCH_API": url,
                       "PIXELRAG_ENCODER_URL": encoder_url, "PYTHONUTF8": "1"}
                search = None
                with (output / f"{geometry}-search-service.log").open("a", encoding="utf-8") as log:
                    try:
                        search = subprocess.Popen([sys.executable, str(Path(__file__).with_name("search_api.py")),
                            "--index-dir", str(snapshot), "--port", str(port)], env=env, stdout=log, stderr=log,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        wait_ready(url, {"index_id": manifests[geometry]["index_id"]}, search, timeout=60)
                        for run in [r for r in runs if r["geometry"] == geometry]:
                            print(f"Running {geometry}/{run['aggregation']} against owned search service {url}", file=sys.stderr, flush=True)
                            subprocess.run(run["command"], check=True, env=env)
                            check_report(output / f"{geometry}-{run['aggregation']}.json", manifests[geometry]["index_id"],
                                         index_manifest.fingerprint(snapshot))
                            status["completed"].append({"geometry": geometry, "aggregation": run["aggregation"], "search_api": url})
                    finally:
                        stop_owned(search)
            status["status"] = "valid"
    except BaseException as exc:
        status.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_owned(encoder)
        (output / "execution.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def validate(indexes):
    manifests = {}
    for geometry, path in indexes.items():
        root = resolve_index(path)
        m = index_manifest.load(root)
        if not m:
            raise ValueError(f"{geometry}: legacy index has no reproducible manifest; rebuild a snapshot")
        if m["chunking"]["mode"] != geometry:
            raise ValueError(f"{geometry}: snapshot has {m['chunking']['mode']} geometry")
        manifests[geometry] = m
    first = next(iter(manifests.values()))
    docs = lambda m: sorted((d["source_id"], d["sha256"], d["pages"]) for d in m["documents"])
    for name, m in manifests.items():
        if m["encoder"] != first["encoder"] or m["render"] != first["render"] or docs(m) != docs(first):
            raise ValueError(f"{name}: encoder, rendering or source corpus differs")
        if {k: v for k, v in m["chunking"].items() if k != "mode"} != {k: v for k, v in first["chunking"].items() if k != "mode"}:
            raise ValueError(f"{name}: crop dimensions/stride differ")
    return manifests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for g in ("page", "regions", "multiscale"):
        ap.add_argument("--" + g, type=Path, required=True)
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--search-port", type=int, default=30002)
    ap.add_argument("--encoder-url", default="http://127.0.0.1:8001")
    args = ap.parse_args()
    indexes = {g: getattr(args, g) for g in ("page", "regions", "multiscale")}
    manifests = validate(indexes)
    if args.execute:
        data = load_dataset(args.eval, args.allow_draft)
        for index in indexes.values():
            preflight(data, index)
    runs = []
    for g, path in indexes.items():
        for aggregation in ("max", "sum"):
            cmd = [sys.executable, str(Path(__file__).with_name("visual_bench.py")), "--eval", str(args.eval),
                   "--index", str(path), "--aggregation", aggregation, "--candidate-budget", "24",
                   "--rounds", "2", "--seed", "1729", "--output", str(args.output / f"{g}-{aggregation}.json")]
            if args.allow_draft:
                cmd.append("--allow-draft")
            runs.append({"geometry": g, "aggregation": aggregation, "command": cmd})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({"manifests": manifests, "runs": runs,
        "control": "same encoder/revision, corpus hashes, rendering, 4 reader pages, 24 candidates per query per branch",
        "analysis": "report visual-only geometry contrast; lexical is unchanged control; hybrid includes two branches",
        "claims": "no causal benefit claimed until paired retrieval and answer results exist"}, indent=2), encoding="utf-8")
    if args.execute:
        execute(runs, indexes, manifests, args.output, args.search_port, args.encoder_url)
    print(json.dumps({"validated": True, "executed": args.execute, "runs": len(runs)}))


if __name__ == "__main__":
    main()
