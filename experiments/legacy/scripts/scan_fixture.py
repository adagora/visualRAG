#!/usr/bin/env python
"""Flatten a small, explicit source-page selection to a textless PDF.

Derived fixtures need their own indexes; source page map is emitted alongside.
This script uses raster rendering, never OCR or an inference API.
"""
import argparse
import json
from pathlib import Path

from visual_bench import digest


def flatten(source, pages, output, dpi=120):
    import pymupdf
    if not pages or len(pages) > 12 or len(set(pages)) != len(pages):
        raise ValueError("Choose 1-12 unique source pages; avoid duplicating the whole corpus")
    if not 72 <= dpi <= 200:
        raise ValueError("Fixture dpi must be 72-200")
    output = Path(output)
    if output.resolve() == Path(source).resolve() or output.exists():
        raise ValueError("Fixture destination must be a new file distinct from the source")
    with pymupdf.open(source) as src, pymupdf.open() as dst:
        if any(p < 1 or p > len(src) for p in pages):
            raise ValueError("Invalid 1-based source page selection")
        for number in pages:
            page = src[number - 1]
            out = dst.new_page(width=page.rect.width, height=page.rect.height)
            out.insert_image(out.rect, stream=page.get_pixmap(dpi=dpi).tobytes("png"))
        output.parent.mkdir(parents=True, exist_ok=True)
        dst.save(output, deflate=True)
    with pymupdf.open(output) as check:
        if any(p.get_text().strip() for p in check):
            raise RuntimeError("Flattened fixture unexpectedly has extractable text")
    manifest = {"source": str(source), "source_sha256": digest(source), "fixture_sha256": digest(output),
                "dpi": dpi, "textless_verified": True,
                "page_map": [{"fixture_page": i + 1, "source_page": p} for i, p in enumerate(pages)],
                "review_status": "derived_fixture_requires_separate_index_and_gold_mapping"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--pages", required=True, help="comma-separated 1-based source pages")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()
    print(json.dumps(flatten(args.source, [int(x) for x in args.pages.split(",")], args.output, args.dpi), indent=2))


if __name__ == "__main__":
    main()
