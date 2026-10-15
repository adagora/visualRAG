"""Locate reader citations on visible PDF pages without another model call."""
from __future__ import annotations

import math
import re
import unicodedata

import pymupdf


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())


def _quote_rects(page, quote):
    matches = page.search_for(quote, quads=True)
    if matches:
        return [match.rect for match in matches[:32]]
    # Some PDFs keep the line-end hyphen even with TEXT_DEHYPHENATE. Match
    # complete token sequences, joining only hyphens at a real line boundary.
    words = page.get_text("words", sort=True)
    tokens, locations = [], []
    i = 0
    while i < len(words):
        word = words[i]
        text, origins = word[4], [word]
        while (text.endswith(("-", "\u00ad")) and i + 1 < len(words)
               and text[:-1].isalpha() and words[i + 1][4][0].isalpha()
               and words[i + 1][5] == word[5] and words[i + 1][6] == word[6] + 1
               and words[i + 1][7] == 0):
            i += 1
            word = words[i]
            text = text[:-1] + word[4]
            origins.append(word)
        for token in _tokens(text):
            tokens.append(token)
            locations.append(origins)
        i += 1
    wanted = _tokens(re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", quote))
    if not wanted:
        return []
    lines = {}
    for start in range(len(tokens) - len(wanted) + 1):
        if tokens[start:start + len(wanted)] != wanted:
            continue
        for origins in locations[start:start + len(wanted)]:
            for word in origins:
                key = (word[5], word[6])
                rect = pymupdf.Rect(word[:4])
                lines[key] = lines[key] | rect if key in lines else rect
        if len(lines) >= 32:
            break
    return list(lines.values())[:32]


def _region(box):
    if not isinstance(box, dict) or set(box) != {"left", "top", "right", "bottom"}:
        return []
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in box.values()):
        return []
    left, top, right, bottom = (box[key] for key in ("left", "top", "right", "bottom"))
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return []
    return [{"left": left, "top": top, "width": right - left, "height": bottom - top}]


def locate(pdf, page, quote, box=None):
    """Prefer a copied quote; otherwise mark a valid reader box as approximate.

    Rectangles are fractions of the upright, cropped page, like the indexed
    JPEG and PDF.js canvas. PDF text search coordinates need the page rotation
    matrix before normalization. No PDF text is sent to a model or retrieval.
    """
    quote = quote.strip()[:600] if isinstance(quote, str) else ""
    result = {"quote": quote, "rects": [], "highlight_status": "unavailable"}
    if quote and pdf is not None:
        try:
            with pymupdf.open(pdf) as document:
                source = document[page - 1]
                matches = _quote_rects(source, quote)
                visible = source.rect
                rects = []
                for match in matches:
                    rect = (match * source.rotation_matrix) & visible
                    if not rect.is_empty:
                        rects.append({
                            "left": (rect.x0 - visible.x0) / visible.width,
                            "top": (rect.y0 - visible.y0) / visible.height,
                            "width": rect.width / visible.width,
                            "height": rect.height / visible.height,
                        })
                if rects:
                    result.update(rects=rects, highlight_status="matched")
                    return result
        except (OSError, ValueError, IndexError, RuntimeError):
            # A missing text layer or unreadable PDF must not lose the answer.
            pass
    rects = _region(box)
    if rects:
        result.update(rects=rects, highlight_status="approximate")
    return result
