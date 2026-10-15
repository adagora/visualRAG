"""One-process Gemini PixelRAG web app."""
from __future__ import annotations

import argparse
import asyncio
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Literal
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

import gemini_index
import chat_context
import rag
import visual_query

load_dotenv(Path.cwd() / ".env", override=False)
STATIC = Path(__file__).parent / "static"
if not STATIC.is_dir():
    STATIC = Path(sys.prefix) / "share/pixelrag/static"
app = FastAPI(title="PixelRAG")
app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")


@lru_cache(maxsize=1)
def get_index() -> gemini_index.Index:
    try:
        return gemini_index.Index(Path(os.environ.get("PIXELRAG_INDEX_DIR", "index")))
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"Cannot open Gemini index: {exc}") from exc


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=chat_context.MAX_MESSAGE_CHARS)


class Ask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(default="", max_length=4000)
    query_image: str | None = Field(default=None, max_length=7_000_000)
    history: list[Message] = Field(default_factory=list, max_length=chat_context.MAX_MESSAGES)
    selected_source_id: str | None = Field(default=None, min_length=1, max_length=128)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/docs")
def documents(index: gemini_index.Index = Depends(get_index)):
    return {"docs": index.documents}


def _check_source(index, article_id, source):
    if source is None:
        return
    document = next((doc for doc in index.documents if doc["article_id"] == article_id), None)
    if document is None:
        raise HTTPException(404, "No such document")
    if document["source_id"] != source:
        raise HTTPException(409, "The document changed. Refresh the app to load its current sources.")


@app.get("/api/page/{article_id}/{page}")
def page_image(article_id: int, page: int, source: str | None = None,
               index: gemini_index.Index = Depends(get_index)):
    _check_source(index, article_id, source)
    try:
        path = index.page_path(article_id, page)
    except (IndexError, KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "No such page") from exc
    if not path.is_file():
        raise HTTPException(404, "No such page")
    return FileResponse(path, media_type="image/jpeg", headers={
        "Cache-Control": "private, max-age=86400" if source else "no-store",
    })


@app.get("/api/pdf/{article_id}")
def source_pdf(article_id: int, source: str | None = None,
               index: gemini_index.Index = Depends(get_index)):
    _check_source(index, article_id, source)
    try:
        path = index.source_pdf(article_id)
    except (IndexError, KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "No such document") from exc
    if not path.is_file():
        raise HTTPException(404, "No such document")
    ascii_name = path.name.encode("ascii", "replace").decode().replace('"', "")
    disposition = f'inline; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(path.name, safe="")}'
    return FileResponse(path, media_type="application/pdf", headers={
        "Content-Disposition": disposition,
        "Cache-Control": "private, max-age=86400" if source else "no-store",
    })


@app.post("/api/ask")
def ask(req: Ask, request: Request, index: gemini_index.Index = Depends(get_index)):
    selected_document = None
    if req.selected_source_id is not None:
        selected = next((doc for doc in index.documents if doc["source_id"] == req.selected_source_id), None)
        if selected is None:
            raise HTTPException(409, "The document changed. Refresh the app to load its current sources.")
        selected_document = {"source_id": selected["source_id"], "title": selected["title"]}
    history = chat_context.bounded_history([message.model_dump() for message in req.history])
    try:
        image = visual_query.decode_image(req.query_image)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    question = req.question.strip()
    if not question:
        if image is None:
            raise HTTPException(400, "Enter a question or attach an image")
        question = "Znajdź element pokazany na obrazie i opisz go na podstawie źródeł."

    messages: queue.Queue = queue.Queue(maxsize=8)
    cancelled = threading.Event()

    def post(kind, payload):
        while not cancelled.is_set():
            try:
                messages.put((kind, payload), timeout=.2)
                return
            except queue.Full:
                continue

    def worker():
        try:
            result = rag.answer(index, question, image=image,
                                on_event=lambda event: post("event", event),
                                cancelled=cancelled.is_set, history=history,
                                selected_document=selected_document)
            post("done", result)
        except rag.CancelledError:
            pass
        except Exception as exc:
            post("error", f"{type(exc).__name__}: {exc}")
        finally:
            post(None, None)

    async def stream():
        threading.Thread(target=worker, daemon=True).start()
        try:
            while not await request.is_disconnected():
                try:
                    kind, payload = await asyncio.to_thread(messages.get, True, .2)
                except queue.Empty:
                    continue
                if kind is None:
                    break
                yield f"data: {json.dumps({'kind': kind, 'payload': payload}, ensure_ascii=False)}\n\n"
        finally:
            cancelled.set()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(os.environ.get("PIXELRAG_INDEX_DIR", "index")))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jev", action=argparse.BooleanOptionalAction, default=None,
                        help="Use experimental Jev image decisions for every question")
    args = parser.parse_args()
    if args.jev is not None:
        os.environ["PIXELRAG_JEV"] = "rerank" if args.jev else "off"
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    os.environ["PIXELRAG_INDEX_DIR"] = str(args.index.resolve())
    get_index.cache_clear()
    try:
        get_index()
    except HTTPException as exc:
        parser.exit(1, f"{exc.detail}\nBuild with pixelrag-index PDF --output INDEX.\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
