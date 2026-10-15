"""Validate visual evidence and bounded requests in original-page percentages."""
from __future__ import annotations

import hashlib
import io
import json
import math

ZOOM_MARK = "---ZOOM---"


def source_id(page: dict) -> str:
    if page.get("source_id"):
        return str(page["source_id"])
    # Compatibility for old callers. Identity never depends on ranked position.
    name = str(page.get("document", page.get("article_id", "unknown")))
    return "legacy-" + hashlib.sha256(name.encode()).hexdigest()[:16]


def page_number(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
        if number < 1 or float(value) != number:
            return None
        return number
    except (TypeError, ValueError, OverflowError):
        return None


def validate_rect(rect) -> dict | None:
    if not isinstance(rect, dict):
        return None
    keys = ("left", "top", "width", "height")
    aliases = ("x", "y", "w", "h")
    try:
        values = [rect[k] if k in rect else rect[a] for k, a in zip(keys, aliases)]
        if any(isinstance(v, bool) for v in values):
            return None
        x, y, w, h = map(float, values)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 100 or y + h > 100:
        return None
    return dict(zip(keys, (x, y, w, h)))


def parse_zoom(text: str) -> dict | None:
    if text.count(ZOOM_MARK) != 1:
        return None
    tail = text.partition(ZOOM_MARK)[2].strip().strip("`").strip()
    try:
        request = json.loads(tail)
    except (ValueError, TypeError):
        return None
    if not isinstance(request, dict) or not isinstance(request.get("source_id"), str):
        return None
    page = page_number(request.get("page"))
    rect = validate_rect(request.get("rect"))
    if page is None or rect is None:
        return None
    return {"source_id": request["source_id"], "page": page, "rect": rect}


def resolve_zoom(request: dict | None, pages: list[dict]):
    if not request:
        return None
    rect = validate_rect(request.get("rect"))
    matches = [p for p in pages if source_id(p) == request.get("source_id")
               and p["page"] == page_number(request.get("page"))]
    if rect is None or len(matches) != 1:
        return None
    return matches[0], rect


def context_rect(rect: dict, margin: float = 8) -> dict:
    rect = validate_rect(rect)
    if rect is None:
        raise ValueError("Invalid original-page rectangle")
    x, y = max(0, rect["left"] - margin), max(0, rect["top"] - margin)
    right = min(100, rect["left"] + rect["width"] + margin)
    bottom = min(100, rect["top"] + rect["height"] + margin)
    return {"left": x, "top": y, "width": right - x, "height": bottom - y}


def crop_with_context(image_bytes: bytes, rect: dict) -> bytes:
    from PIL import Image
    r = context_rect(rect)
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        bounds = (math.floor(width * r["left"] / 100),
                  math.floor(height * r["top"] / 100),
                  math.ceil(width * (r["left"] + r["width"]) / 100),
                  math.ceil(height * (r["top"] + r["height"]) / 100))
        out = io.BytesIO()
        image.crop(bounds).save(out, format="PNG")
        return out.getvalue()
