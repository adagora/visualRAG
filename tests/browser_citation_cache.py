"""Citation navigation with a warm browser cache and changing source PDFs."""
import json
from pathlib import Path
import socket
import sys
import threading
import time

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
        return {"answer": "Evidence on page 2.", "citations": [row], "pages": [row]}

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
            page = browser.new_page(viewport={"width": 1300, "height": 900})
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
                citation = page.locator(".citation")
                assert citation.evaluate("element => element.tagName") == "A"
                assert f"source={source}" in citation.get_attribute("href")
                citation.click()
                expect(page.locator("#viewerStatus")).to_contain_text(f"{source} · strona 2")
                expect(page.locator("#pdfCanvas")).to_be_visible()
                renders.append(page.locator("#pdfCanvas").evaluate("canvas => canvas.toDataURL()"))
                links.append(page.locator("#openPdf").get_attribute("href"))
            assert links[0] != links[1]
            assert renders[0] != renders[1], "Citation displayed the previous source's cached PDF"
            assert not errors, errors
            page.screenshot(path=str(out / "after-corpus-change.png"))
            browser.close()
            return {"warm_cache_source_change": "passed", "source_links": links, "browser_errors": errors}
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        app.rag.answer = original_answer
        app.app.dependency_overrides.clear()
        app.app.dependency_overrides.update(original_overrides)
        sock.close()


if __name__ == "__main__":
    print(json.dumps(run_cache_check()))
