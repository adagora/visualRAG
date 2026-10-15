"""Real-app Markdown and topic presentation checks; no model or external calls."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import pymupdf
from playwright.sync_api import expect, sync_playwright
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app


MARKDOWN = """## Dobór produktu

Polecam **pergola aluminiowa ECHO** oraz *montaż zgodny z instrukcją*.

- Aluminiowa konstrukcja
- Ochrona przed deszczem

1. Sprawdź wymiary
2. Zamów produkt

Ustaw `tryb = bezpieczny`.

```python
print("<script>window.injected = true</script>")
```

| Produkt | Szczegóły techniczne | Cena | Szerokość | Wysokość |
| --- | --- | --- | --- | --- |
| ECHO | """ + "długi_parametr_" * 6 + """ | 1000 zł | 300 cm | 250 cm |

[Dokumentacja](https://example.com/documentation)
[Niebezpieczny](javascript:alert(1))
[Dane](data:text/html,evil)
![Obraz zewnętrzny](https://example.com/forbidden-markdown-image.png)

<script>window.injected = true</script>
<img src="https://example.com/forbidden-html-image.png" onerror="window.injected=true">
"""


def run_presentation_check():
    captures = []
    release_reset = threading.Event()
    labels = {"warranty": "Gwarancje", "claim": "Reklamacje", "service": "Serwis",
              "manual": "Instrukcje", "price": "Ceny", "other": "Inne"}
    with tempfile.TemporaryDirectory(prefix="pixelrag-presentation-") as temporary:
        pdf_path = Path(temporary) / "source.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((40, 80), "Presentation test document")
            document.save(pdf_path)

        class Index:
            documents = [{"article_id": 0, "source_id": "presentation-source",
                          "title": "ECHO", "pages": 1}]

            def source_pdf(self, article_id):
                return pdf_path

        def answer(index, question, *, image=None, on_event=None, cancelled=None,
                   history=None, selected_document=None):
            captures.append({"question": question, "history": history})
            category = question if question in labels else "unclassified"
            if question == "Unknown category":
                category = "not-a-topic"
            if question == "Reset theme":
                on_event({"type": "classification", "category": "claim"})
                if not release_reset.wait(10):
                    raise RuntimeError("Browser did not release classification fixture")
            else:
                on_event({"type": "classification", "category": category})
            return {"answer": MARKDOWN if question == "Markdown" else "**Odpowiedź** dla " + question,
                    "classification": {"category": category}, "pages": [],
                    "citations": [{"article_id": 0, "source_id": "presentation-source",
                                   "page": 1, "document": "ECHO"}]}

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
            assert server.started, "Presentation fixture server failed to start"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1800})
                errors, external_requests, ask_requests = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                origin = f"http://127.0.0.1:{sock.getsockname()[1]}"

                def record_request(request):
                    if request.url.endswith("/api/ask"):
                        ask_requests.append(request.post_data_json)
                    if request.url.startswith(("http:", "https:")) and not request.url.startswith(origin + "/"):
                        external_requests.append(request.url)

                page.on("request", record_request)
                page.route("https://example.com/**", lambda route: route.abort())
                page.goto(origin)
                expect(page.locator("#pdfCanvas")).to_be_visible(timeout=15000)

                def ask(question):
                    page.locator("#q").fill(question)
                    page.locator("#send").click()
                    expect(page.locator("#send")).to_be_enabled(timeout=15000)
                    return page.locator(".response").last

                card = ask("Markdown")
                body = card.locator(".answer-body.markdown")
                expect(body.locator("strong")).to_have_text("pergola aluminiowa ECHO")
                expect(body.locator("em")).to_have_text("montaż zgodny z instrukcją")
                expect(body.locator("h2")).to_have_text("Dobór produktu")
                expect(body.locator("ul > li")).to_have_count(2)
                expect(body.locator("ol > li")).to_have_count(2)
                expect(body.locator("p > code")).to_have_text("tryb = bezpieczny")
                expect(body.locator("pre > code")).to_contain_text('<script>window.injected = true</script>')
                expect(body.locator("table th")).to_have_count(5)
                expect(body.locator("table tbody tr")).to_have_count(1)
                expect(body.locator("script, img")).to_have_count(0)
                expect(body).to_contain_text('<img src="https://example.com/forbidden-html-image.png"')
                assert page.evaluate("window.injected") is None
                expect(body.locator('a[href^="javascript:"], a[href^="data:"]')).to_have_count(0)
                safe_link = body.get_by_role("link", name="Dokumentacja", exact=True)
                expect(safe_link).to_have_attribute("href", "https://example.com/documentation")
                expect(safe_link).to_have_attribute("target", "_blank")
                assert {"noopener", "noreferrer"} <= set(safe_link.get_attribute("rel").split())
                citation = card.locator("a.citation")
                expect(citation).to_have_count(1)
                assert "/api/pdf/0?source=presentation-source#page=1" in citation.get_attribute("href")
                citation.click()
                expect(page.locator("#pgNum")).to_have_value("1")
                output = Path(__file__).resolve().parents[1] / "eval/results/presentation"
                output.mkdir(parents=True, exist_ok=True)
                page.locator("#log").evaluate("el => el.scrollTop = 0")
                page.screenshot(path=str(output / "markdown-response.png"), full_page=True)
                ask("warranty")
                assert ask_requests[-1]["history"][-1] == {"role": "assistant", "text": MARKDOWN}
                # The server normalizes outer whitespace; the browser must send raw Markdown.
                assert captures[-1]["history"][-1] == {"role": "assistant", "text": MARKDOWN.strip()}
                page.locator("#newChat").click()

                styles = {}
                for category, label in labels.items():
                    card = ask(category)
                    expect(card).to_have_attribute("data-topic", category)
                    expect(card.locator(".response-header .status")).to_have_count(1)
                    badge = card.locator(".response-header .topic-badge")
                    expect(badge).to_be_visible()
                    expect(badge).to_contain_text(label)
                    styles[category] = card.evaluate("""card => {
                        const style = getComputedStyle(card), badge = getComputedStyle(card.querySelector('.topic-badge'));
                        const rgb = color => color.match(/[\\d.]+/g).slice(0, 3).map(Number).map(v => {
                            v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
                        });
                        const lum = color => rgb(color).reduce((sum, v, i) => sum + v * [.2126,.7152,.0722][i], 0);
                        const a = lum(badge.color), b = lum(badge.backgroundColor);
                        return {background:style.background, border:style.borderTopColor,
                                contrast:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};
                    }""")
                    assert styles[category]["contrast"] >= 4.5, (category, styles[category])
                assert len({style["background"] for style in styles.values()}) == len(labels), styles
                assert len({style["border"] for style in styles.values()}) == len(labels), styles
                page.locator("#log").evaluate("el => el.scrollTop = 0")
                page.screenshot(path=str(output / "topic-cards.png"), full_page=True)

                neutral = ask("Unknown category").evaluate("el => getComputedStyle(el).background")
                assert neutral not in {style["background"] for style in styles.values()}
                assert page.locator(".response").last.get_attribute("data-topic") is None
                expect(page.locator(".topic-badge").last).to_be_hidden()
                page.locator("#q").fill("Reset theme")
                page.locator("#send").click()
                card = page.locator(".response").last
                expect(card).to_have_attribute("data-topic", "claim")
                expect(page.locator("#send")).to_be_disabled()
                release_reset.set()
                expect(page.locator("#send")).to_be_enabled()
                assert card.get_attribute("data-topic") is None
                expect(card.locator(".topic-badge")).to_be_hidden()
                assert card.evaluate("el => getComputedStyle(el).background") == neutral

                page.set_viewport_size({"width": 390, "height": 844})
                page.locator("#newChat").click()
                card = ask("Markdown")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Page overflows on mobile"
                assert card.evaluate("el => el.getBoundingClientRect().right <= innerWidth")
                assert card.locator("table").evaluate("""table => {
                    for (let el = table; el && !el.classList.contains('response'); el = el.parentElement) {
                        if (['auto','scroll'].includes(getComputedStyle(el).overflowX) && el.scrollWidth > el.clientWidth) return true;
                    }
                    return false;
                }"""), "Wide Markdown table needs a horizontal scroll container"
                assert not external_requests, external_requests
                assert not errors, errors
                browser.close()
                return {"markdown": "passed", "raw_history": "passed", "topic_themes": "passed",
                        "streaming_reset": "passed", "mobile_overflow": "passed",
                        "inert_content": "passed", "screenshots": str(output)}
        finally:
            release_reset.set()
            server.should_exit = True
            thread.join(timeout=5)
            app.rag.answer = original_answer
            app.app.dependency_overrides.clear()
            app.app.dependency_overrides.update(original_overrides)
            sock.close()


if __name__ == "__main__":
    print(json.dumps(run_presentation_check()))
