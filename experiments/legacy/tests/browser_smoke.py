"""Browser checks against a running UI; model responses are deterministic.

Run: python tests/browser_smoke.py --url http://127.0.0.1:8000
The source PDF/viewer are real. No reader API or embedding model is called.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright, expect


def frame(kind, payload):
    return "data: " + json.dumps({"kind": kind, "payload": payload}) + "\n\n"


def run(url, screenshot):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator("#mode option")).to_have_count(4)
        docs = page.request.get(url + "/api/docs").json()["docs"]
        assert docs and docs[0]["pages"] >= 22, "Use PX1 for this source-viewer check."
        document = docs[0]["title"]

        def reply(route):
            body = route.request.post_data_json
            requests.append(body)
            number = 17 if len(requests) == 1 else 22
            quote = "First evidence" if number == 17 else "Second evidence"
            row = {"article_id": 0, "source_id": "source-0", "page": number,
                   "tile_index": number - 1, "chunk_index": 1, "document": document,
                   "score": .7, "box": {"left": 10, "top": 20, "width": 30, "height": 40}}
            citation = {"article_id": 0, "source_id": "source-0", "document": document,
                        "page": number, "quote": quote, "supports": quote,
                        "status": "model-region", "rects": [{"left": 20, "top": 30, "width": 25, "height": 20}]}
            cites = [citation]
            if number == 22:
                cites.append({"article_id": 0, "page": number, "quote": "Unsupported evidence", "supports": "invalid region",
                              "status": "unsupported", "rects": []})
            result = {"answer": quote, "citations": cites, "pins": [], "provider": "fake",
                      "model": "browser-check", "retrieval": "visual", "usage": {}, "timings": {}}
            response = frame("event", {"type": "search", "query": body["question"], "hits": [row]})
            response += frame("event", {"type": "tile", **row})
            response += frame("event", {"type": "zoom", "article_id": 0, "page": number, "rect": row["box"]})
            response += frame("event", {"type": "answer_reset"})
            response += frame("event", {"type": "answer_delta", "text": quote})
            response += frame("done", result)
            route.fulfill(status=200, content_type="text/event-stream", body=response)

        page.route("**/api/ask", reply)
        page.locator("#q").fill("First question")
        page.locator("#send").click()
        first = page.get_by_role("button", name=re.compile("First evidence")).first
        expect(first).to_be_visible()
        expect(page.locator("#pgNum")).to_have_value("17", timeout=30000)
        page.locator("#q").fill("Second question")
        page.locator("#send").click()
        expect(page.get_by_role("button", name=re.compile("Second evidence")).first).to_be_visible()
        expect(page.locator("#pgNum")).to_have_value("22", timeout=30000)
        expect(page.get_by_role("button", name=re.compile("Unsupported evidence"))).to_be_disabled()
        expect(page.locator('#marks .hl')).to_have_count(1)
        first.click()
        expect(page.locator("#pgNum")).to_have_value("17", timeout=30000)
        expect(page.locator("#vfoot")).to_contain_text("First evidence")
        assert "history" not in requests[0]

        image = io.BytesIO()
        Image.new("RGB", (120, 80), "white").save(image, format="PNG")
        page.locator("#queryImage").set_input_files({"name": "reference.png", "mimeType": "image/png", "buffer": image.getvalue()})
        expect(page.locator("#clearImage")).to_be_visible()
        page.locator("#q").fill("Find this drawing")
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled()
        assert requests[-1]["query_image"].startswith("data:image/png;base64,")
        page.locator("#clearImage").click()
        expect(page.locator("#clearImage")).to_be_hidden()
        expect(page.locator("#cancelAsk")).to_be_hidden()
        page.screenshot(path=str(screenshot), full_page=True)
        assert not errors, errors
        browser.close()
        return {"browser_errors": errors, "checks": ["default modes", "real PDF navigation",
                "older-answer citation state", "unsupported citation", "bounded zoom events",
                "screenshot attachment", "no automatic conversation preprocessing"], "screenshot": str(screenshot)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshot", type=Path, default=Path("eval/results/ui-smoke.png"))
    args = parser.parse_args()
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.url, args.screenshot)))
