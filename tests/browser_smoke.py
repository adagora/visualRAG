"""Real PDF viewer with deterministic replies; no model calls."""
import argparse
import io
import json
from pathlib import Path
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from browser_citation_cache import run_cache_check

def run(url, screenshot):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1450, "height": 1000})
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        docs = page.request.get(url + "/api/docs").json()["docs"]
        assert docs and docs[0]["pages"] >= 4
        doc = docs[0]
        def reply(route):
            body = route.request.post_data_json
            requests.append(body)
            number = 2 if len(requests) == 1 else 4
            hit = {"article_id": doc["article_id"], "source_id": doc["source_id"], "document": doc["title"], "page": number, "score": .9}
            decision = "applied" if len(requests) == 1 else "fallback"
            result = {"answer": f"Evidence on page {number}.", "citations": [hit], "pages": [hit], "model": "fake", "usage": {}, "reader_calls": 1,
                      "retrieval": {"jev": {"status": decision}}}
            event = {"kind": "event", "payload": {"type": "retrieved", "pages": [hit]}}
            done = {"kind": "done", "payload": result}
            route.fulfill(status=200, content_type="text/event-stream", body="".join("data: " + json.dumps(item) + "\n\n" for item in (event, done)))
        page.route("**/api/ask", reply)
        expect(page.locator("#pdfCanvas")).to_be_visible(timeout=30000)
        assert page.locator("#mode").count() == page.locator("#cmp").count() == 0
        for text in ("First question", "Second question"):
            page.locator("#q").fill(text)
            page.locator("#send").click()
            expect(page.locator("#send")).to_be_enabled()
        expect(page.locator(".status").first).to_contain_text("strony wybrane z Jev")
        expect(page.locator(".status").last).to_contain_text("Jev niedostępny")
        expect(page.locator(".citation")).to_have_count(2)
        expect(page.locator(".evidence")).to_have_count(2)
        page.locator(".citation").last.click()
        expect(page.locator("#pgNum")).to_have_value("4")
        expect(page.locator("#pdfCanvas")).to_be_visible(timeout=30000)
        page.locator(".citation").first.click()
        expect(page.locator("#pgNum")).to_have_value("2")
        expect(page.locator("#pdfCanvas")).to_be_visible(timeout=30000)
        image = io.BytesIO()
        Image.new("RGB", (100, 60), "white").save(image, format="PNG")
        page.locator("#queryImage").set_input_files({"name": "reference.png", "mimeType": "image/png", "buffer": image.getvalue()})
        expect(page.locator("#clearImage")).to_be_visible()
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled()
        assert requests[-1]["query_image"].startswith("data:image/png;base64,")
        assert set(requests[-1]) == {"question", "query_image"}
        expect(page.locator("#clearImage")).to_be_hidden()
        expect(page.locator("#cancelAsk")).to_be_hidden()
        page.screenshot(path=str(screenshot), full_page=True)
        assert not errors, errors
        browser.close()
        return {"browser_errors": errors, "checks": ["single retrieval flow", "real PDF viewer", "retrieved previews", "older answer citations", "image-only question", "Jev success and fallback status"], "screenshot": str(screenshot)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshot", type=Path, default=Path("eval/results/ui-mvp-smoke.png"))
    args = parser.parse_args()
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.url, args.screenshot)
    result["citation_cache"] = run_cache_check()
    print(json.dumps(result))
