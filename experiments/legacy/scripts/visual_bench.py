#!/usr/bin/env python
"""Matched-budget visual/BM25/hybrid evaluation; labels and failures stay visible.

No service is contacted during --preflight. Agent-inspected draft labels require
--allow-draft; results from them are exploratory, never independent human gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "eval/visual_questions.json"
MODES = ("visual", "lexical", "hybrid")


def resolve_index(index_dir):
    """Resolve an immutable active snapshot without importing orchestration."""
    import layout
    return layout.resolve_index_dir(Path(index_dir))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(path, allow_draft=False):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Expected visual evaluation schema_version=1")
    if data.get("review_status") != "human_audited" and not allow_draft:
        raise ValueError("Dataset is agent-inspected draft; human audit pending. Use --allow-draft for exploratory results.")
    rows = data["questions"]
    if not rows or len({q["id"] for q in rows}) != len(rows):
        raise ValueError("Question ids must be nonempty and unique")
    groups = {}
    evidence_splits = {}
    for q in rows:
        for key in ("id", "q", "type", "visual_dependency", "split", "page_family", "source_id", "answerable", "gold", "expected_answer", "review"):
            if key not in q:
                raise ValueError(f"{q.get('id')}: missing {key}")
        if q["split"] not in ("dev", "test"):
            raise ValueError(f"{q['id']}: split must be dev or test")
        group = (q["source_id"], q["page_family"])
        if group in groups and groups[group] != q["split"]:
            raise ValueError(f"Page family leaks across splits: {group}")
        groups[group] = q["split"]
        if bool(q["gold"]) != q["answerable"]:
            raise ValueError(f"{q['id']}: answerable requires gold; negatives require empty gold")
        for g in q["gold"]:
            page_key = (g.get("source_id", q["source_id"]), g["page"])
            if page_key in evidence_splits and evidence_splits[page_key] != q["split"]:
                raise ValueError(f"Evidence page leaks across splits: {page_key}")
            evidence_splits[page_key] = q["split"]
            if not isinstance(g["page"], int) or g["page"] < 1:
                raise ValueError(f"{q['id']}: pages are 1-based positive integers")
            for rect in g.get("rectangles", []):
                if len(rect) != 4 or not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
                    raise ValueError(f"{q['id']}: rectangles use normalized page coordinates")
        if q["visual_dependency"] == "required" and any(not g.get("rectangles") for g in q["gold"]):
            raise ValueError(f"{q['id']}: visually dependent gold requires evidence rectangles")
        if not q["expected_answer"].get("facts") and q["answerable"]:
            raise ValueError(f"{q['id']}: expected fact labels missing")
        if not allow_draft and q["review"].get("status") != "human_audited":
            raise ValueError(f"{q['id']}: individual human audit pending")
    return data


def preflight(data, index_dir):
    """Missing labels/sources are hard errors, never a smaller denominator."""
    index_dir = resolve_index(index_dir)
    articles = json.loads((index_dir / "articles.json").read_text(encoding="utf-8"))
    def canonical(value, base=ROOT):
        path = Path(value.replace("\\", "/"))
        return (path if path.is_absolute() else base / path).resolve()
    import index_manifest
    manifest = index_manifest.load(index_dir)
    mapping = {}
    urls = {canonical(a["url"], index_dir if manifest else ROOT): i for i, a in enumerate(articles)}
    if len(urls) != len(articles):
        raise ValueError("Duplicate source identities in index")
    recorded = {}
    if manifest:
        for doc in manifest.get("documents", []):
            sha = doc.get("sha256")
            if sha in recorded:
                raise ValueError("Duplicate full source hashes in snapshot manifest")
            recorded[sha] = doc
    for source in data["sources"]:
        sid = source["source_id"]
        if manifest:
            # Match the label's content version before considering any original
            # path. Immutable copied sources remain valid if originals move/change.
            doc = recorded.get(source["sha256"])
            if not doc or doc.get("pages") != source.get("page_count"):
                raise ValueError(f"Snapshot source hash/page count disagrees with gold: {sid}")
            aid = doc["article_id"]
            if not isinstance(aid, int) or not 0 <= aid < len(articles):
                raise ValueError("Invalid snapshot article identity")
            path = canonical(doc.get("snapshot_path") or articles[aid]["url"], index_dir)
            if doc.get("snapshot_path") and not path.is_relative_to(index_dir):
                raise ValueError("Snapshot copied source escapes the snapshot directory")
            if canonical(articles[aid]["url"], index_dir) != path:
                raise ValueError("Snapshot article URL disagrees with its copied source")
            mapping[sid] = aid
        else:
            path = canonical(source["path"])
            if path not in urls:
                raise ValueError(f"Gold source missing from index: {sid}")
            mapping[sid] = urls[path]
        if not path.is_file() or digest(path) != source["sha256"]:
            raise ValueError(f"Gold source missing or content changed: {sid}")
        if not source.get("page_count"):
            raise ValueError(f"Missing page count for {sid}")
    sources = {s["source_id"]: s for s in data["sources"]}
    for q in data["questions"]:
        if q["source_id"] not in sources:
            raise ValueError(f"{q['id']}: unlabeled source")
        for g in q["gold"]:
            sid = g.get("source_id", q["source_id"])
            if sid not in sources or sid not in mapping or g["page"] > sources[sid]["page_count"]:
                raise ValueError(f"{q['id']}: invalid gold source/page")
    return mapping


def score(q, ranking, mapping, k=4):
    gold = {(mapping[g.get("source_id", q["source_id"])], g["page"]) for g in q["gold"]}
    pool = list(dict.fromkeys(tuple(p) for p in ranking))
    got = pool[:k]
    ranks = [i + 1 for i, p in enumerate(got) if p in gold]
    all_ranks = [i + 1 for i, p in enumerate(pool) if p in gold]
    return {"hit@1": bool(gold and got and got[0] in gold),
            "hit@4": bool(ranks), "mrr@4": 1 / min(ranks) if ranks else 0.0,
            "mrr@candidate_budget": 1 / min(all_ranks) if all_ranks else 0.0,
            "complete_evidence@4": bool(gold and gold <= set(got)),
            "gold_pages": len(gold), "covered_pages": len(gold & set(got)),
            "complete_possible@4": len(gold) <= k}


def measure(q, mode, budget, aggregation="max", retriever=None):
    started = time.perf_counter()
    try:
        if retriever is None:
            import rag
            retriever = rag.retrieve_for
        pages, _, stats = retriever(q["q"], mode, n_pages=budget,
                                   per_query=budget, aggregation=aggregation)
        ranking = [[p.article_id, p.page] for p in pages][:budget]
        return {"ranking": ranking, "stats": stats, "error": None,
                "latency_ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:
        return {"ranking": [], "stats": {}, "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000}


def summarize(rows, questions):
    positives = {q["id"] for q in questions if q["answerable"]}
    result = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode and r["phase"] == "warm"]
        positive_rows = [r for r in selected if r["question_id"] in positives]
        n = len(positive_rows)
        result[mode] = {"positive_trials": n, "question_count": len(positives),
                        "errors": sum(bool(r["error"]) for r in selected),
                        "negative_trials": len(selected) - n,
                        "negative_policy": "retrieval cannot establish refusal; answer_eval scores abstention",
                        **{key: sum(r["metrics"][key] for r in positive_rows) / n if n else None
                           for key in ("hit@1", "hit@4", "mrr@4", "mrr@candidate_budget", "complete_evidence@4")},
                        "median_warm_ms": statistics.median(r["latency_ms"] for r in selected) if selected else None}
    return result


def provenance(index_dir, dataset):
    import index_manifest
    manifest = index_manifest.load(Path(index_dir))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        gpu_info = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu_info = None
    return {"index_manifest": manifest, "index_fingerprint": index_manifest.fingerprint(Path(index_dir)),
            "dataset_sha256": digest(dataset), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "gpu_inventory": gpu_info, "manifest_status": "recorded" if manifest else "legacy_unversioned_index",
            "configuration": {key: value for key, value in os.environ.items()
                              if key in ("PIXELRAG_INDEX_DIR", "PIXELRAG_ENCODER_DEVICE", "PIXELRAG_ENCODER_DTYPE", "PIXELRAG_SEARCH_API", "PIXELRAG_ENCODER_URL", "PIXELRAG_EMBED_MODEL", "PIXELRAG_AGGREGATION", "PIXELRAG_QUERY_TOKENS", "PIXELRAG_COMPILE")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--index", type=Path, default=ROOT / "index-px1")
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="test")
    ap.add_argument("--candidate-budget", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--aggregation", choices=("max", "sum"), default="max")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        payload = json.loads(args.worker)
        print(json.dumps(measure(**payload), ensure_ascii=False))
        return
    if args.candidate_budget < 4 or args.rounds < 1:
        ap.error("candidate-budget must be >=4; rounds must be positive")
    # layout binds its default at import. Select the CLI index before the
    # resolver imports it, so corpus/lexical/queryembed all bind the same data.
    os.environ["PIXELRAG_INDEX_DIR"] = str(args.index.resolve())
    args.index = resolve_index(args.index)
    os.environ["PIXELRAG_INDEX_DIR"] = str(args.index)
    data = load_dataset(args.eval, args.allow_draft)
    mapping = preflight(data, args.index)
    qs = [q for q in data["questions"] if args.split == "all" or q["split"] == args.split]
    if not qs:
        ap.error("selected split is empty")
    report = {"schema_version": 1, "review_status": data["review_status"],
              "claim_status": "exploratory" if data["review_status"] != "human_audited" else "audited",
              "provenance": provenance(args.index, args.eval), "config": vars(args).copy(),
              "budget": {"reader_pages": 4, "ranked_shortlist_pages": args.candidate_budget,
                         "mrr_rank_cutoff": args.candidate_budget, "candidates_per_query_per_retriever": args.candidate_budget,
                         "hybrid_note": "two retrievers; per-branch budget matched, total candidate work is larger"},
              "cold_definition": "fresh Python process; OS and remote service caches may remain warm",
              "questions": qs,
              "rows": []}
    if not args.preflight:
        rng = random.Random(args.seed)
        modes = list(MODES)
        rng.shuffle(modes)
        for order, mode in enumerate(modes):
            payload = {"q": qs[0], "mode": mode, "budget": args.candidate_budget, "aggregation": args.aggregation}
            t = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(payload)],
                                   capture_output=True, text=True, encoding="utf-8", timeout=300, env={**os.environ, "PYTHONUTF8": "1"})
                r = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"ranking": [], "stats": {}, "error": p.stderr[-2000:], "latency_ms": (time.perf_counter() - t) * 1000}
            except Exception as exc:
                r = {"ranking": [], "stats": {}, "error": str(exc), "latency_ms": (time.perf_counter() - t) * 1000}
            report["rows"].append({**r, "mode": mode, "phase": "cold", "order": order, "question_id": qs[0]["id"],
                                   "process_wall_ms": (time.perf_counter() - t) * 1000, "metrics": score(qs[0], r["ranking"], mapping)})
        # Explicit unmeasured warmups; failures are retained in the report.
        report["warmups"] = {m: measure(qs[0], m, args.candidate_budget, args.aggregation) for m in MODES}
        for repetition in range(args.rounds):
            order_qs = list(qs)
            rng.shuffle(order_qs)
            for q in order_qs:
                rng.shuffle(modes)
                for order, mode in enumerate(modes):
                    r = measure(q, mode, args.candidate_budget, args.aggregation)
                    report["rows"].append({**r, "mode": mode, "phase": "warm", "round": repetition,
                                           "order": order, "question_id": q["id"], "type": q["type"],
                                           "visual_dependency": q["visual_dependency"], "metrics": score(q, r["ranking"], mapping)})
        report["summary"] = summarize(report["rows"], qs)
        report["summary_by_type"] = {kind: summarize([r for r in report["rows"] if r.get("type") == kind], [q for q in qs if q["type"] == kind])
                                      for kind in sorted({q["type"] for q in qs})}
        report["summary_by_visual_dependency"] = {kind: summarize([r for r in report["rows"] if r.get("visual_dependency") == kind], [q for q in qs if q["visual_dependency"] == kind])
                                                  for kind in sorted({q["visual_dependency"] for q in qs})}
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
