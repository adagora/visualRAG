"""Deterministic real-app conversation browser checks, without model calls."""
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import pymupdf
from PIL import Image
from playwright.sync_api import expect, sync_playwright
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app


def run_conversation_check():
    captures = []
    cancelled_started = threading.Event()
    cancelled_finished = threading.Event()
    resolved = 'Warranty for <img src=x onerror="window.injected=true">'
    with tempfile.TemporaryDirectory(prefix="pixelrag-conversation-") as temporary:
        pdf_path = Path(temporary) / "source.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((40, 80), "Conversation test document")
            document.save(pdf_path)

        class Index:
            documents = [{"article_id": 0, "source_id": "conversation-source", "title": "Test", "pages": 1}]

            def source_pdf(self, article_id):
                return pdf_path

        def answer(index, question, *, image=None, on_event=None, cancelled=None,
                   history=None, selected_document=None):
            captures.append({"question": question, "history": history,
                             "selected_document": selected_document, "image": image is not None})
            if question == "Cancel":
                cancelled_started.set()
                try:
                    for _ in range(500):
                        if cancelled():
                            raise app.rag.CancelledError()
                        time.sleep(.01)
                    raise RuntimeError("Cancellation did not reach worker")
                finally:
                    cancelled_finished.set()
            if question == "Fail":
                raise RuntimeError("Expected failure")
            if history:
                on_event({"type": "resolving_context"})
            effective = resolved if question == "Followup" else question
            on_event({"type": "context", "question": effective, "contextual": bool(history)})
            return {"answer": "Which product?" if question == "Clarify" else "Answer " + question,
                    "resolved_question": effective, "needs_clarification": question == "Clarify",
                    "pages": [], "citations": []}

        original_answer = app.rag.answer
        original_overrides = dict(app.app.dependency_overrides)
        app.rag.answer = answer
        app.app.dependency_overrides[app.get_index] = lambda: Index()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
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
                page = browser.new_page()
                errors, requests = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: requests.append(request.post_data_json)
                        if request.url.endswith("/api/ask") else None)
                page.goto(f"http://127.0.0.1:{sock.getsockname()[1]}")
                expect(page.locator("#pdfCanvas")).to_be_visible(timeout=15000)

                def ask(question):
                    page.locator("#q").fill(question)
                    page.locator("#send").click()
                    expect(page.locator("#send")).to_be_enabled()

                ask("First")
                assert captures[-1]["history"] == []
                assert captures[-1]["selected_document"] == {"source_id": "conversation-source", "title": "Test"}
                ask("Followup")
                assert captures[-1]["history"] == [
                    {"role": "user", "text": "First"}, {"role": "assistant", "text": "Answer First"}]
                expect(page.locator(".response").last).to_contain_text(resolved)
                expect(page.locator(".response img")).to_have_count(0)
                assert page.evaluate("window.injected") is None
                ask("Clarify")
                assert captures[-1]["history"][-2]["text"] == resolved
                expect(page.locator(".status").last).to_have_text("Doprecyzuj pytanie")
                expect(page.locator(".answer").last).to_have_text("Which product?")
                expected_history = requests[-1]["history"] + [
                    {"role": "user", "text": "Clarify"}, {"role": "assistant", "text": "Which product?"}]
                ask("Fail")
                expect(page.locator(".status").last).to_contain_text("Expected failure")
                page.locator("#q").fill("Cancel")
                page.locator("#send").click()
                expect(page.locator("#newChat")).to_be_disabled()
                assert cancelled_started.wait(5)
                page.locator("#cancelAsk").click()
                expect(page.locator("#send")).to_be_enabled()
                expect(page.locator(".status").last).to_contain_text("Anulowano")
                assert cancelled_finished.wait(5)
                ask("After errors")
                assert captures[-1]["history"] == expected_history
                before = page.locator("#openPdf").get_attribute("href")
                page.locator("#newChat").click()
                expect(page.locator(".turn")).to_have_count(0)
                assert page.locator("#openPdf").get_attribute("href") == before
                ask("Fresh")
                assert captures[-1]["history"] == []
                image = io.BytesIO()
                Image.new("RGB", (100, 60)).save(image, format="PNG")
                page.locator("#queryImage").set_input_files({"name": "test.png", "mimeType": "image/png", "buffer": image.getvalue()})
                expect(page.locator("#clearImage")).to_be_visible()
                page.locator("#send").click()
                expect(page.locator("#send")).to_be_enabled()
                expect(page.locator("#clearImage")).to_be_hidden()
                assert requests[-1]["query_image"].startswith("data:image/png;base64,")
                ask("After image")
                assert "data:image" not in json.dumps(requests[-1]["history"])
                assert all(set(message) == {"role", "text"} for message in requests[-1]["history"])
                for number in range(6):
                    ask(f"Bound {number} " + "x" * 3900)
                for body in requests:
                    assert len(body["history"]) <= 8
                    assert sum(len(message["text"]) for message in body["history"]) <= 12000
                    assert all(len(message["text"]) <= 4000 for message in body["history"])
                assert requests[-1]["history"][-2]["text"].startswith("Bound 4"), [(message["role"], message["text"][:50]) for message in requests[-1]["history"]]
                page.reload()
                expect(page.locator("#pdfCanvas")).to_be_visible()
                ask("After reload")
                assert captures[-1]["history"] == []
                assert not errors, errors
                browser.close()
                return {"history_bounds": "passed", "resolved_and_clarification": "passed",
                        "cancel_failure_exclusion": "passed", "reset_and_reload": "passed",
                        "selected_source": "passed", "browser_errors": errors}
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            app.rag.answer = original_answer
            app.app.dependency_overrides.clear()
            app.app.dependency_overrides.update(original_overrides)
            sock.close()


if __name__ == "__main__":
    print(json.dumps(run_conversation_check()))
