"""Parse reader citations and resolve evidence to attached original pages.

Source identity takes precedence. Legacy page-only citations resolve only when
there is one attached document for that page. Ambiguous or unattached evidence
is retained as unsupported without linking it to a guessed document.

Visual model regions work on scans; phrase location supplies text geometry
when available. Neither method verifies whether the cited evidence supports
an answer. Status records model-region, text-located, unlocated or unsupported.
"""

from __future__ import annotations

import json
import logging
import re

import citations as C
import corpus
import evidence

log = logging.getLogger("pixelrag")

CITE_MARK = "---CYTATY---"


def split_citations(text: str) -> tuple[str, list[dict]]:
    """Separate the prose answer from the trailing ---CYTATY--- JSONL block.

    Tolerant by design: a reader that omits the block, fences it in ```, or
    writes one malformed line should still produce a usable answer. Anything
    unparseable is dropped rather than shown.
    """
    if CITE_MARK not in text:
        return text.strip(), []
    body, _, tail = text.partition(CITE_MARK)
    cites = []
    for line in tail.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(c, dict) and (c.get("quote") or "rects" in c):
            cites.append(c)
    return body.strip(), cites


def resolve_citations(answer: str, cites: list[dict],
                      pages: list[dict]) -> list[dict]:
    """Resolve document identity before drawing visual or text rectangles."""
    by_key = {(p["article_id"], p["page"]): p for p in pages}
    by_source: dict[tuple, list[dict]] = {}
    by_page: dict[int, list[dict]] = {}
    for p in pages:
        by_page.setdefault(p["page"], []).append(p)
        by_source.setdefault((evidence.source_id(p), p["page"]), []).append(p)

    out = []
    for c in cites:
        pno = evidence.page_number(c.get("page"))
        if pno is None:
            continue
        aid = c.get("article_id")
        sid = c.get("source_id")
        candidates = by_page.get(pno, [])
        if sid is not None:
            matches = by_source.get((sid, pno), []) if isinstance(sid, str) else []
            target = matches[0] if len(matches) == 1 else None
        elif aid is not None:
            target = by_key.get((aid, pno)) if isinstance(aid, int) and not isinstance(aid, bool) else None
        else:
            target = candidates[0] if len(candidates) == 1 else None
        if target is not None and aid is not None and target["article_id"] != aid:
            target = None
        if target is None:
            out.append({"source_id": sid, "page": pno, "quote": c.get("quote", ""),
                        "supports": c.get("supports") or c.get("claim") or "",
                        "rects": [], "status": "unsupported", "verified": False,
                        "reason": "ambiguous-page" if sid is None and aid is None and len(candidates) > 1
                        else "unattached-source", "text_layer": False, "coverage": 0.0})
            continue
        pdf = corpus.source_pdf(target["article_id"])
        quote = str(c.get("quote") or "")
        try:
            found = C.locate_quote(pdf, pno, quote) if pdf and quote else []
            text_layer = bool(pdf and C._page_words(str(pdf), pno))
        except Exception:
            found, text_layer = [], False
        raw_rects = c.get("rects", [])
        rects = [evidence.validate_rect(r) for r in raw_rects] if isinstance(raw_rects, list) else [None]
        invalid = any(r is None for r in rects)
        visual = rects if not invalid else []
        out.append({
            "document": target["document"],
            "article_id": target["article_id"],
            "source_id": evidence.source_id(target),
            "page": pno,
            "quote": quote,
            "supports": c.get("supports") or c.get("claim") or "",
            # Invalid model geometry does not invalidate a known source or
            # independently located text. Never guess a different coordinate unit.
            "rects": visual or (found[0]["rects"] if found else []),
            "status": "model-region" if visual else "text-located" if found else "unlocated",
            "reason": "invalid-rectangles" if invalid else "",
            "coverage": found[0]["coverage"] if found else 0.0,
            # Geometry describes location; it does not verify claim support.
            "text_layer": text_layer,
            "verified": False,
        })

    # Numbers are per (document, page), not per citation — attach to the first
    # citation of each so the UI draws each box once. Grouping by page alone
    # would pin one document's figures onto another's page of the same number.
    for aid, pno in {(c["article_id"], c["page"]) for c in out if "article_id" in c and c["status"] != "unsupported"}:
        pdf = corpus.source_pdf(aid)
        if not pdf:
            continue
        try:
            pinned, repeated = C.locate_numbers(pdf, pno, answer)
        except Exception:
            continue
        for c in out:
            if c.get("article_id") == aid and c["page"] == pno:
                c["numbers"] = pinned
                c["repeated"] = repeated
                break
    return out


_NUM = re.compile(r"\d[\d   .,]{1,12}\d|\d{2,}")


def locate_values(article_id: int, page: int, answer: str) -> list[dict]:
    """Find where the numbers the model quoted actually sit on the page.

    The rendered tiles are pixels, but these PDFs still carry a text layer, so
    every figure in the answer can be located exactly rather than approximated
    by the region the model happened to open. Returns boxes as percentages of
    the page, the same convention rag.box_pct uses for chunk overlays.

    This is a *check on the model*, not a source of truth: a quoted number that
    cannot be found on the cited page is worth seeing.
    """
    src = corpus.source_pdf(article_id)
    if src is None:
        return []

    # "3 555 zł" / "3.555" / "3555" should all find the cell reading 3555.
    # Prices and millimetre dimensions are 3-6 digits. Anything longer is a
    # date or a document code (answers cite "CBG_PL-PLN_10.10.2024"); anything
    # shorter matches half the table.
    stem = src.stem
    wanted: set[str] = set()
    for m in _NUM.finditer(answer):
        raw = m.group(0).strip(" .,\u00a0\u202f")
        if raw in stem:                       # part of the document identifier
            continue
        digits = re.sub(r"[\s.,\u00a0\u202f]", "", raw)
        if not digits.isdigit() or not 3 <= len(digits) <= 6:
            continue
        wanted.add(digits)
        if raw != digits and len(raw) - len(digits) == 1:
            wanted.add(raw)                   # "3 555" as typeset in the page
    if not wanted:
        return []

    import fitz

    out: list[dict] = []
    with fitz.open(src) as doc:
        if not 0 <= page - 1 < doc.page_count:
            return []
        pg = doc[page - 1]
        pw, ph = pg.rect.width, pg.rect.height
        for term in sorted(wanted, key=len, reverse=True)[:12]:
            try:
                rects = pg.search_for(term)
            except Exception:
                continue
            for r in rects[:6]:
                out.append({
                    "value": term,
                    "left": 100 * r.x0 / pw, "top": 100 * r.y0 / ph,
                    "width": 100 * (r.x1 - r.x0) / pw,
                    "height": 100 * (r.y1 - r.y0) / ph,
                    "ambiguous": len(rects) > 1,
                })
    return _confirm_intersections(out)


def _confirm_intersections(pins: list[dict], tol: float = 1.5) -> list[dict]:
    """Resolve which copy of a repeated figure is the one actually cited.

    In a price matrix the answer names the row and column thresholds as well as
    the price, so the right cell is the one sitting under a quoted column header
    and level with a quoted row header. A value that repeats elsewhere in the
    table will not satisfy both. This turns two dashed guesses into one
    confirmed cell without asking the model where it looked.
    """
    for p in pins:
        has_col = any(q is not p and abs(q["left"] - p["left"]) < tol
                      and q["top"] < p["top"] - tol for q in pins)
        has_row = any(q is not p and abs(q["top"] - p["top"]) < tol
                      and q["left"] < p["left"] - tol for q in pins)
        p["confirmed"] = bool(has_col and has_row)
    # A confirmed cell is no longer an open question, whatever its twin does.
    for p in pins:
        if p["confirmed"]:
            p["ambiguous"] = False
    return pins


