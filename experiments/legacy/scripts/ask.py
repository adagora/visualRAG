#!/usr/bin/env python
"""Ask a question; the agent searches and browses tiles until it can answer.

Retrieval, tools, and prompting live in rag.py, shared with the web UI
(scripts/app.py) so the two cannot drift.

Requires the search API and an Anthropic key:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/pixelrag serve \
      --index-dir ./index --tiles-dir ./index/tiles \
      --articles-json ./index/articles.json --port 30001 --device cpu
    export ANTHROPIC_API_KEY=...        # or: ant auth login

Usage:
    .venv/bin/python scripts/ask.py "Ile kosztuje brama Connect H 2400x3000?"
    .venv/bin/python scripts/ask.py "..." --retrieve-only   # no model call
"""

import argparse
import sys
import json
from pathlib import Path

import citeparse
import corpus
import rag
import requests
import visual_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--image", type=Path, help="PNG/JPEG/WebP reference screenshot or crop")
    ap.add_argument("--pages", type=int, choices=range(1,13), help="Reader page budget (default 4)")
    ap.add_argument("--json", action="store_true", help="JSON retrieval report with --retrieve-only")
    ap.add_argument("--k", type=int, default=5,
                    help="hits for --retrieve-only (the agent picks its own otherwise)")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="show what search returns; skip the billed agent loop")
    ap.add_argument("--max-steps", type=int, default=rag.MAX_STEPS)
    ap.add_argument("--mode", choices=["oneshot", "agent"],
                    help="default: PIXELRAG_ASK or oneshot")
    ap.add_argument("--retrieval", choices=["auto", *rag.RETRIEVAL_MODES],
                    default="auto",
                    help="which retriever answers; default: auto "
                         "(PIXELRAG_JEV / PIXELRAG_HYBRID). "
                         "scripts/compare.py runs them against each other")
    ap.add_argument("--provider", choices=["gemini", "anthropic"],
                    help="default: whichever API key is set")
    ap.add_argument("--list-models", action="store_true",
                    help="list models the configured key can reach, then exit")
    ap.add_argument("--log", metavar="LEVEL",
                    help="debug|info|warning|error (or set PIXELRAG_LOG)")
    args = ap.parse_args()
    # To stderr, alongside the trace: stdout carries the answer, and a warning
    # in the middle of a streamed price is worse than no warning.
    rag.configure_logging(args.log, stream=sys.stderr)

    if args.list_models:
        provider = args.provider or rag.detect_provider()
        print(f"# {provider}")
        for m in rag.list_models(provider):
            print(" ", m)
        return
    image = None
    if args.image:
        if args.image.stat().st_size > visual_query.MAX_BYTES:
            ap.error("Image must be at most 5 MB")
        image = visual_query.decode_image(visual_query.data_url(args.image.read_bytes()))
    if not args.question and image is not None:
        args.question = "Znajdź ten element w dokumentacji i opisz go na podstawie źródeł."
    if not args.question:
        ap.error("a question is required (or use --list-models)")

    if args.retrieve_only:
        try:
            pages, debug, stats = rag.retrieve_for(args.question, args.retrieval,
                                                   n_pages=args.k, query_image=image)
        except requests.RequestException as e:
            sys.exit(f"Search API unreachable on {rag.SEARCH_API}: {e}")
        if args.json:
            print(json.dumps({"pages": [rag._hit_row(page) for page in pages], "variants": debug, "stats": stats}, ensure_ascii=False))
        else:
            for page in pages:
                print(f"  {corpus.doc_title(page.article_id)} p{page.page} region {page.focus} score {page.ranking_score:.3f}")
        return

    # The answer streams to stdout as it arrives; the citation block is stripped
    # from `result["answer"]` afterwards, so what was streamed is reprinted only
    # when nothing streamed at all (a cache hit, or a provider without deltas).
    streamed = False
    nonlocal_state = {"buf": "", "shown": 0, "done": False}

    def show(ev):
        nonlocal streamed
        if ev["type"] == "search":
            s = ev.get("stats") or {}
            detail = (f"{s['unique_chunks']} chunks → {s['pages']} pages · "
                      f"{s['reranker']} · {s['ms']:.0f}ms" if s
                      else f"{len(ev['hits'])} hits")
            print(f"  ⌕ searched {ev['query']!r} — {detail}", file=sys.stderr)
            for note in s.get("notes", []):
                print(f"  ! {note}", file=sys.stderr)
        elif ev["type"] == "tile":
            print(f"  ▣ read {ev['document']} p{ev['page']} region {ev['chunk_index']}",
                  file=sys.stderr)
        elif ev["type"] == "tile_error":
            print(f"  ! no region {ev['tile_index']}:{ev['chunk_index']}", file=sys.stderr)
        elif ev["type"] == "answer_delta":
            # The reader ends with a ---CYTATY--- JSONL block that split_citations
            # strips before anyone sees it. Streaming bypasses that strip, so the
            # cut has to happen here too — otherwise raw citation JSON scrolls
            # past, which is exactly what the marker exists to prevent.
            nonlocal_state["buf"] += ev["text"]
            if nonlocal_state["done"]:
                return
            buf = nonlocal_state["buf"]
            if '---ZOOM---' in buf:
                nonlocal_state['done'] = True
                return
            if citeparse.CITE_MARK in buf:
                nonlocal_state["done"] = True
                tail = buf.split(citeparse.CITE_MARK)[0]
                text = tail[nonlocal_state["shown"]:]
            else:
                # Hold back a marker's worth so a split delta cannot print a
                # partial "---CYTATY---" before we recognise it.
                keep = max(0, len(buf) - len(citeparse.CITE_MARK))
                text = buf[nonlocal_state["shown"]:keep]
            if text:
                if not streamed:
                    print(file=sys.stderr)
                    streamed = True
                nonlocal_state["shown"] += len(text)
                print(text, end="", flush=True)
        elif ev["type"] == "answer_reset":
            if nonlocal_state['shown']:
                print()
            nonlocal_state.update(buf='', shown=0, done=False)

    try:
        result = rag.run_agent(args.question, on_event=show,
                               max_steps=args.max_steps, provider=args.provider,
                               mode=args.mode, retrieval=args.retrieval,
                               query_image=image, n_pages=args.pages)
    except requests.RequestException as e:
        sys.exit(f"Search API unreachable on {rag.SEARCH_API}: {e}")

    if streamed:
        prefix = nonlocal_state['buf'][:nonlocal_state['shown']]
        if result['answer'].startswith(prefix):
            print(result['answer'][nonlocal_state['shown']:])
        else:
            print('\n' + result['answer'])
    else:
        print()
        print(result["answer"])

    u = result["usage"]
    cost = f" — about ${u['cost_usd']:.3f}" if u["cost_usd"] is not None else ""
    thoughts = u.get("thoughts") or 0
    think = f" · {thoughts} think" if thoughts else ""
    cached = ""
    if result.get("cached"):
        cached = f" · CACHED (cos {result['cache_similarity']:.3f})"
    cache_tok = ""
    if u.get("cache_read"):
        cache_tok = f" · {u['cache_read']} cache-read"
    t = result.get("timings") or {}
    speed = ""
    if t.get("total_ms") is not None:
        parts = [f"retrieval {t['retrieval_ms']:.0f}ms"] if t.get("retrieval_ms") is not None else []
        if t.get("ttft_ms") is not None:
            parts.append(f"first token {t['ttft_ms']:.0f}ms")
        parts.append(f"total {t['total_ms']:.0f}ms")
        speed = " · " + " / ".join(parts)
    print(f"\n[{result['provider']}/{result['model']} · {result.get('retrieval', 'auto')} · "
          f"{result['steps']} round trips · "
          f"{u['input']} in / {u['output']} out{think}{cache_tok}{cost}{cached}{speed}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
