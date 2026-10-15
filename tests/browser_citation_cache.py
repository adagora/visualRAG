"""Citation navigation with a warm browser cache and changing source PDFs."""
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pymupdf
import uvicorn
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app


def run_cache_check():
    out = Path(__file__).resolve().parents[1] / "eval/results/citation-cache"
    out.mkdir(parents=True, exist_ok=True)
    for source in ("OLD", "NEW"):
        with pymupdf.open() as document:
            for number in (1, 2):
                page = document.new_page(width=400, height=500)
                page.insert_text((40, 100), f"{source} SOURCE PAGE {number}", fontsize=24)
                page.insert_text((40, 460), "Warranty coverage: five years.", fontsize=16)
                color = (1, 0, 0) if source == "OLD" else (0, 0, 1)
                page.draw_rect((40, 150, 300, 300), color=color, fill=color)
            document.save(out / f"{source}.pdf")
    state = {"source": "OLD"}

    class Index:
        @property
        def documents(self):
            return [{"article_id": 0, "source_id": state["source"], "title": state["source"], "pages": 2}]

        def source_pdf(self, aid):
            return out / (state["source"] + ".pdf")

        def page_path(self, aid, page):
            path = out / f"{state['source']}-{page}.jpg"
            with pymupdf.open(self.source_pdf(aid)) as pdf:
                pdf[page - 1].get_pixmap().save(path)
            return path

    def answer(index, question, **kwargs):
        row = {"article_id": 0, "source_id": state["source"], "document": state["source"], "page": 2}
        kwargs["on_event"]({"type": "retrieved", "pages": [row]})
        classification = {"category": "warranty", "confidence": .95}
        kwargs["on_event"]({"type": "classification", **classification})
        # Derive trustworthy normalized coordinates from the real, rendered PDF.
        with pymupdf.open(index.source_pdf(0)) as pdf:
            page = pdf[1]
            quote = f"{state['source']} SOURCE PAGE 2"
            rect = page.search_for(quote)[0]
            location = {"left": rect.x0 / page.rect.width, "top": rect.y0 / page.rect.height,
                        "width": rect.width / page.rect.width, "height": rect.height / page.rect.height}
        # Exercise the actual reader parser and PDF locator before the UI.
        raw = {"source_id": row["source_id"], "page": row["page"],
               "quote": ("" if question == "Approximate" else
                         "Warranty coverage: five years." if question == "Lower" else quote),
               "box": {"left": location["left"], "top": location["top"],
                       "right": location["left"] + location["width"],
                       "bottom": location["top"] + location["height"]}}
        response = SimpleNamespace(text=json.dumps({"answer": "Evidence on page 2.", "citations": [raw]}))
        answer_text, citations = app.rag._parse_response(response, [row], index)
        citation = citations[0]
        if question == "Invalid":
            citation.update(quote="<img src=x onerror=alert(1)>", rects=[
                {**location, "left": -.1}, {**location, "width": 2},
                {**location, "top": "0.1"}], highlight_status="matched")
        if question == "Unclassified":
            classification = {"category": "unclassified", "confidence": None}
        return {"answer": answer_text, "citations": [citation], "pages": [row],
                "classification": classification}

    original_answer, original_overrides = app.rag.answer, dict(app.app.dependency_overrides)
    app.rag.answer = answer
    app.app.dependency_overrides[app.get_index] = lambda: Index()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app.app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(.02)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1300, "height": 900}, device_scale_factor=2)
            errors, renders, links = [], [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            url = f"http://127.0.0.1:{port}"
            # No request routing: Playwright routing would disable HTTP cache.
            page.goto(url)
            for source in ("OLD", "NEW"):
                if source == "NEW":
                    state["source"] = source
                    page.reload()
                expect(page.locator("#viewerStatus")).to_contain_text(source)
                expect(page.locator("#pdfCanvas")).to_be_visible(timeout=10000)
                page.locator("#q").fill("Question")
                page.locator("#send").click()
                expect(page.locator(".citation")).to_have_count(1)
                expect(page.locator(".topic-badge")).to_have_text("Temat: Gwarancje")
                citation = page.locator(".citation")
                assert citation.evaluate("element => element.tagName") == "A"
                assert f"source={source}" in citation.get_attribute("href")
                citation.click()
                expect(page.locator("#viewerStatus")).to_contain_text(f"{source} · strona 2")
                expect(page.locator("#pdfCanvas")).to_be_visible()
                highlight = page.locator(".pdf-highlight.matched")
                expect(highlight).to_have_count(1)
                expect(highlight).to_be_visible()
                assert highlight.evaluate("node => getComputedStyle(node).borderTopStyle") == "solid"
                expect(page.locator("#highlightQuote")).to_have_text(f"{source} SOURCE PAGE 2")
                def assert_alignment():
                    canvas = page.locator("#pdfCanvas").bounding_box()
                    overlay = page.locator("#pdfHighlights").bounding_box()
                    for key in ("x", "y", "width", "height"):
                        assert abs(canvas[key] - overlay[key]) < 1, (canvas, overlay)
                    box = highlight.bounding_box()
                    assert canvas["x"] <= box["x"] < box["x"] + box["width"] <= canvas["x"] + canvas["width"]
                    assert canvas["y"] <= box["y"] < box["y"] + box["height"] <= canvas["y"] + canvas["height"]
                    with pymupdf.open(out / f"{source}.pdf") as pdf:
                        expected = pdf[1].search_for(f"{source} SOURCE PAGE 2")[0]
                        assert abs(box["x"] - canvas["x"] - expected.x0 / 400 * canvas["width"]) < 1
                        assert abs(box["y"] - canvas["y"] - expected.y0 / 500 * canvas["height"]) < 1
                        assert abs(box["width"] - expected.width / 400 * canvas["width"]) < 1
                        assert abs(box["height"] - expected.height / 500 * canvas["height"]) < 1
                assert_alignment()
                page.set_viewport_size({"width": 1100, "height": 850})
                page.wait_for_timeout(500)
                expect(highlight).to_be_visible()
                assert_alignment()
                renders.append(page.locator("#pdfCanvas").evaluate("canvas => canvas.toDataURL()"))
                links.append(page.locator("#openPdf").get_attribute("href"))
                page.locator("#prevPage").click()
                expect(page.locator("#viewerStatus")).to_contain_text("strona 1")
                expect(page.locator(".pdf-highlight")).to_have_count(0)
                for question in ("Approximate", "Invalid", "Unclassified"):
                    page.locator("#q").fill(question)
                    page.locator("#send").click()
                    expect(page.locator("#send")).to_be_enabled()
                    page.locator(".citation").last.click()
                    if question == "Approximate":
                        expect(page.locator(".pdf-highlight.approximate")).to_be_visible()
                        assert page.locator(".pdf-highlight").evaluate("node => getComputedStyle(node).borderTopStyle") == "dashed"
                    elif question == "Invalid":
                        expect(page.locator(".pdf-highlight")).to_have_count(0)
                        expect(page.locator("#viewerStatus")).to_contain_text("położenie fragmentu niedostępne")
                        expect(page.locator("#highlightQuote")).to_have_text("<img src=x onerror=alert(1)>")
                        expect(page.locator("#highlightQuote img")).to_have_count(0)
                    else:
                        expect(page.locator(".topic-badge").last).to_be_hidden()
                # A citation in an older answer retains its own location metadata.
                page.locator(".citation").first.click()
                expect(highlight).to_be_visible()
                assert_alignment()
                page.locator("#pgNum").fill("1")
                page.locator("#pgNum").press("Enter")
                expect(page.locator("#viewerStatus")).to_contain_text("strona 1")
                expect(page.locator(".pdf-highlight")).to_have_count(0)
                page.set_viewport_size({"width": 1100, "height": 500})
                page.locator("#q").fill("Lower")
                page.locator("#send").click()
                expect(page.locator("#send")).to_be_enabled()
                page.locator(".citation").last.click()
                expect(page.locator("#highlightQuote")).to_have_text("Warranty coverage: five years.")
                expect(highlight).to_be_visible()
                area = page.locator("#pdfArea").bounding_box()
                box = highlight.bounding_box()
                assert area["y"] <= box["y"] <= box["y"] + box["height"] <= area["y"] + area["height"]
                assert page.locator("#pdfArea").evaluate("node => node.scrollTop") > 0
                if source == "NEW":
                    page.screenshot(path=str(out / "citation-highlight-topic.png"), full_page=True)
                page.set_viewport_size({"width": 1100, "height": 850})
            assert links[0] != links[1]
            assert renders[0] != renders[1], "Citation displayed the previous source's cached PDF"
            assert not errors, errors
            page.screenshot(path=str(out / "after-corpus-change.png"))
            browser.close()
            return {"warm_cache_source_change": "passed", "highlights_resize_history_bounds": "passed",
                    "topic_badges": "passed", "source_links": links, "browser_errors": errors}
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        app.rag.answer = original_answer
        app.app.dependency_overrides.clear()
        app.app.dependency_overrides.update(original_overrides)
        sock.close()


if __name__ == "__main__":
    print(json.dumps(run_cache_check()))
