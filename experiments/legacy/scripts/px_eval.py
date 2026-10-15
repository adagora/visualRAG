#!/usr/bin/env python
"""Page-image retrieval vs project_x's text RAG, on project_x's own gold set.

project_x (Node + LanceDB, text chunks, vector + BM25 fused by RRF) ships a
gold set of questions about one price list — `jevGolden.ts`, every case with
the pages that answer it. Two exports from that repo make the comparison:

    npx ts-node src/custom/exportGolden.ts            -> reports/golden-export.json
    npm run eval:retrieval -- --label=pages ...       -> reports/retrieval-pages-*.json

The second carries, per question, the rank of the first gold page among the
distinct pages of project_x's RRF pool: `pageRankInDoc` counts only the gold
price list's pages, `pageRank` counts every document. This script ranks the
same questions with this repo's retrieval modes and scores both the same way —
the first gold page's rank, and hit@k over it. bench.py is not reused because
its top-1 is "the one primary page"; several of these cases accept 3-14 pages,
and project_x counts any of them.

    PIXELRAG_INDEX_DIR=index-px1 python scripts/px_eval.py \
        --golden ../project_x/reports/golden-export.json \
        --baseline ../project_x/reports/retrieval-pages-2026-10-13.json

Retrieval only, no reader. Paid modes (anything that calls TypeSafe) refuse to
run without --paid, so the default invocation costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

KS = (1, 4, 6)
PAID = ("jev-expand", "jev", "jev+hybrid", "jev-page", "xray")


def first_gold_rank(got: list[tuple[int, int]], aid: int, gold: set[int]) -> int | None:
    for i, (a, p) in enumerate(got, 1):
        if a == aid and p in gold:
            return i
    return None


def group_of(case_id: str) -> str:
    # U* are verbatim FAQ questions; the rest are prices, options and facts
    # written against the price matrices.
    return "faq" if case_id.startswith("U") else "price"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--modes", default="visual,hybrid")
    ap.add_argument("--depth", type=int, default=20, help="pages ranked per question")
    ap.add_argument("--paid", action="store_true", help="allow TypeSafe modes")
    ap.add_argument("--out", type=Path, default=None, help="per-question JSON")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    paid = [m for m in modes if m in PAID]
    if paid and not args.paid:
        sys.exit(f"{', '.join(paid)} call TypeSafe and cost money — pass --paid to run them.")
    if not paid:
        os.environ["PIXELRAG_JEV"] = "0"  # kill switch: nothing below may bill

    import corpus
    import rag

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    baseline = {r["id"]: r for r in
                json.loads(args.baseline.read_text(encoding="utf-8"))["results"]}
    source = re.compile(golden["source"])
    aids = [i for i, a in enumerate(corpus.articles()) if source.search(a["title"])]
    if len(aids) != 1:
        sys.exit(f"expected one article matching /{golden['source']}/, found {len(aids)}")
    aid = aids[0]

    # Same population as project_x's retrieval eval: first turns that have
    # answer pages and are not the "not in the documents" class.
    cases = [(c["id"], c["turns"][0]) for c in golden["cases"]
             if c["id"] in baseline and c["turns"][0]["answerPages"]
             and not c["turns"][0]["notFound"]]

    rows = []
    for cid, turn in cases:
        gold = set(turn["answerPages"])
        b = baseline[cid]
        row = {"id": cid, "group": group_of(cid), "q": turn["q"], "gold": sorted(gold),
               "px_doc": b.get("pageRankInDoc"), "px_all": b.get("pageRank")}
        for mode in modes:
            t0 = time.perf_counter()
            pages, _, stats = rag.retrieve_for(turn["q"], mode, n_pages=args.depth)
            got = [(p.article_id, p.page) for p in pages]
            row[mode] = first_gold_rank(got, aid, gold)
            row[f"{mode}_top"] = [p for a, p in got[:6] if a == aid]
            row[f"{mode}_ms"] = round((time.perf_counter() - t0) * 1000)
            row[f"{mode}_usd"] = stats.get("cost_usd") or 0.0
        rows.append(row)
        ranks = "  ".join(f"{m} {str(row[m]):>4}" for m in modes)
        print(f"{cid:4} px {str(row['px_doc']):>4}  {ranks}  gold {row['gold'][:6]}  {turn['q'][:60]}",
              flush=True)

    cols = [("project_x (in doc)", "px_doc"), ("project_x (all docs)", "px_all")] + \
           [(m, m) for m in modes]
    print()
    for group in ("price", "faq", None):
        sel = [r for r in rows if group is None or r["group"] == group]
        print(f"== {group or 'all'} (n={len(sel)})")
        for label, key in cols:
            hits = "  ".join(
                f"@{k} {sum(1 for r in sel if r[key] is not None and r[key] <= k):>2}/{len(sel)}"
                for k in KS)
            print(f"  {label:22} {hits}")
    for m in modes:
        ms = sorted(r[f"{m}_ms"] for r in rows)
        usd = sum(r[f"{m}_usd"] for r in rows)
        print(f"  {m}: median {ms[len(ms) // 2]} ms/question, ${usd:.4f} total")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
