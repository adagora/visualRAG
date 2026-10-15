#!/usr/bin/env python
"""Start the three local services and wait until each is actually answering.

The README's start commands are macOS-flavoured (KMP_DUPLICATE_LIB_OK, --device
cpu) and predate the encoder sidecar. On this box — Windows, CUDA — the correct
invocation is different, and the ordering matters: the query encoder must be warm
before the first question or it pays 30s of model load inline.

    .venv/Scripts/python.exe scripts/serve_all.py          # all three, foreground
    .venv/Scripts/python.exe scripts/serve_all.py --stop    # kill whatever is up

Logs go to logs/. Ctrl-C stops everything.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import socket
import uuid
from pathlib import Path

import requests

import layout

ROOT = layout.ROOT
INDEX = layout.DEFAULT
PY = sys.executable
LOGS = ROOT / "logs"
OWNER_FILE = LOGS / "services.json"

SEARCH_PORT = 30001
ENCODER_PORT = 8001
UI_PORT = 8000


def _up(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(url, timeout=timeout).ok
    except requests.RequestException:
        return False


def _wait(name: str, url: str, proc: subprocess.Popen, log: Path,
          limit: float = 300.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            sys.exit(f"{name} exited with code {proc.returncode}.\n"
                     f"  log: {log}\n"
                     f"  tail: {_tail(log)}")
        if _up(url):
            print(f"  {name} ready in {time.perf_counter() - t0:.0f}s", flush=True)
            return
        time.sleep(1.5)
    sys.exit(f"{name} did not become ready in {limit:.0f}s — see {log}")


def _tail(p: Path, n: int = 6) -> str:
    try:
        return "\n        ".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def stop() -> None:
    """Stop only recorded children whose live PID and launch identity agree."""
    if not OWNER_FILE.exists():
        print("no services recorded by this launcher")
        return
    record = json.loads(OWNER_FILE.read_text(encoding="utf-8"))
    for item in record.get("services", []):
        try:
            actual = requests.get(item["url"], timeout=2).json()
            pid = item["pid"]
            if actual.get("owner_token") != record["owner_token"] or actual.get("process_id") != pid:
                continue
            os.kill(pid, signal.SIGTERM)
            print(f"  stopped {item['name']}")
        except (requests.RequestException, ValueError, KeyError, ProcessLookupError, PermissionError):
            continue


def _check_ports(ports):
    for port in ports:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"Port {port} is already occupied; stop its owning launcher or use the running service.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="stop services recorded by this launcher")
    ap.add_argument("--no-ui", action="store_true", help="search + encoder only")
    args = ap.parse_args()

    if args.stop:
        stop()
        return

    os.chdir(ROOT)
    import gemini_index
    is_gemini = gemini_index.enabled()
    ports = ([] if args.no_ui else [UI_PORT]) + ([] if is_gemini else [SEARCH_PORT, ENCODER_PORT])
    _check_ports(ports)
    if not is_gemini and not INDEX.faiss_index.exists():
        sys.exit("No index/index.faiss — build it first:\n"
                 "  .venv/Scripts/python.exe scripts/build_index.py")
    LOGS.mkdir(exist_ok=True)

    owner_token = uuid.uuid4().hex
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "PIXELRAG_INDEX_DIR": str(INDEX.index_dir), "PIXELRAG_OWNER_TOKEN": owner_token}
    procs: list[tuple[str, subprocess.Popen, Path]] = []

    def cleanup():
        for _, process, _ in procs:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in procs:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if OWNER_FILE.exists():
            try:
                if json.loads(OWNER_FILE.read_text(encoding="utf-8")).get("owner_token") == owner_token:
                    OWNER_FILE.unlink()
            except (OSError, ValueError):
                pass

    atexit.register(cleanup)

    def record_services():
        urls = {"search API": f"http://127.0.0.1:{SEARCH_PORT}/health",
                "query encoder (CUDA)": f"http://127.0.0.1:{ENCODER_PORT}/health",
                "web UI": f"http://127.0.0.1:{UI_PORT}/api/docs"}
        OWNER_FILE.write_text(json.dumps({"owner_token": owner_token, "services": [
            {"name": name, "pid": process.pid, "url": urls[name]} for name, process, _ in procs]}), encoding="utf-8")

    def launch(name: str, cmd: list[str], log_name: str) -> subprocess.Popen:
        log = LOGS / log_name
        fh = open(log, "w", encoding="utf-8")
        print(f"starting {name} …", flush=True)
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                             cwd=ROOT)
        procs.append((name, p, log))
        record_services()
        return p

    # 1. faiss search API. OMP_NUM_THREADS=1 is not a tuning choice: faiss and
    #    torch each bundle their own libomp and the process dies without it.
    #    The encoder lives elsewhere precisely so that cap costs nothing.
    search_env_note = {**env, "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    log = LOGS / "serve.log"
    fh = open(log, "w", encoding="utf-8")
    print("starting search API (faiss) …", flush=True)
    if is_gemini:
        if args.no_ui:
            print("Gemini queries need no local model or search service.")
            return
        p_ui = launch("web UI", [PY, str(ROOT / "scripts" / "app.py")], "app.log")
        _wait("web UI", f"http://127.0.0.1:{UI_PORT}/api/docs", p_ui, LOGS / "app.log")
        print(f"\n  open http://127.0.0.1:{UI_PORT}")
        try:
            p_ui.wait()
        except KeyboardInterrupt:
            p_ui.terminate()
        return
    p_search = subprocess.Popen(
        [PY, str(ROOT / "scripts" / "search_api.py"),
         "--index-dir", str(INDEX.index_dir),
         "--port", str(SEARCH_PORT)],
        stdout=fh, stderr=subprocess.STDOUT, env=search_env_note, cwd=ROOT)
    procs.append(("search API", p_search, log))
    record_services()

    # 2. query encoder on CUDA, no faiss in the process, no thread cap.
    p_enc = launch("query encoder (CUDA)",
                   [PY, str(ROOT / "scripts" / "encoder.py")], "encoder.log")

    _wait("search API", f"http://127.0.0.1:{SEARCH_PORT}/health", p_search,
          LOGS / "serve.log")
    _wait("query encoder", f"http://127.0.0.1:{ENCODER_PORT}/health", p_enc,
          LOGS / "encoder.log")

    if not args.no_ui:
        p_ui = launch("web UI", [PY, str(ROOT / "scripts" / "app.py")], "app.log")
        _wait("web UI", f"http://127.0.0.1:{UI_PORT}/api/docs", p_ui, LOGS / "app.log")
        print(f"\n  open http://127.0.0.1:{UI_PORT}")

    print("\nCtrl-C to stop everything.")
    try:
        while True:
            for name, p, lg in procs:
                if p.poll() is not None:
                    print(f"\n{name} died (code {p.returncode}):\n        {_tail(lg)}")
                    raise KeyboardInterrupt
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping …")
        for name, p, _ in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(1.5)
        for name, p, _ in procs:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
