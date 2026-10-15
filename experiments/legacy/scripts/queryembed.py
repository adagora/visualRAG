"""Encoding a query into the vector the index was built with.

WHY THIS IS NOT IN `pixelrag serve`, which has the index and would be the
obvious home: faiss and torch each ship their own libomp, and two OpenMP
runtimes in one process are unsafe, so that process runs under
OMP_NUM_THREADS=1. That single thread is nearly all of search latency
(server-side encode p95 1673ms) and profiling puts 98% of it in the forward.
Nothing requires the encoder to live there — /search accepts a precomputed
`embedding` — so encoding happens here instead, at p95 ~450ms on CPU threads
and ~24ms on a CUDA GPU in fp16, graphed.

WHY IT IS NOT IN rag.py EITHER, which is where it used to be: rag.py is the
orchestration, and this is a model, a process boundary, two caches and a
fallback ladder. It has its own lifecycle (a sidecar that may or may not be
up, a 28s in-process load, a graph that only one thread may replay) and its
own reason to be tested, and none of that is about answering a question.

Two ways to get a vector, in preference order:

  1. the sidecar (scripts/encoder.py) — one model load serves every client
  2. in-process — correct, and pays ~30s on first use

If both fail, the query fails explicitly. The vector service never loads an
encoder or substitutes a different query template. PIXELRAG_LOG=info reports
when the in-process fallback is used.

THE INSTRUCTION STRING IS LOAD-BEARING. `DEFAULT_INSTRUCTION` below must match
every other copy of it byte for byte — scripts/encoder.py, profile_search.py
and pixelrag_serve's own api.py all keep one. A query encoded through a
different chat template lands somewhere else in the space and returns
plausible-looking wrong tiles with no error raised anywhere; it measured as
1/10 identical rankings until the template was replicated exactly.
`scripts/check_parity.py` compares the copies at source level and lists this
file in INSTRUCTION_OWNERS, so moving the constant out of here means updating
that list or silently losing the check.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import requests

import encoder_device
import encoder_contract
import visual_query

log = logging.getLogger("pixelrag")

# Own session, own host. The sidecar is a different service from the faiss one
# and pooling to it is the same win: one handshake rather than one per query.
_HTTP = requests.Session()


LOCAL_ENCODE = os.environ.get("PIXELRAG_LOCAL_ENCODE", "1") != "0"
EMBED_MODEL = os.environ.get("PIXELRAG_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
# Prefer the sidecar (scripts/encoder.py) so the ~30s model load is paid once
# at its startup rather than per process, and one copy of the weights serves
# every client. Falls back to loading in-process if it isn't running.
ENCODER_URL = os.environ.get("PIXELRAG_ENCODER_URL", "http://127.0.0.1:8001")
_encoder: dict | None = None
_sidecar_ok: bool | None = None
_sidecar_checked = 0.0


def _sidecar_available() -> bool:
    global _sidecar_ok, _sidecar_checked
    if _sidecar_ok is not False or time.monotonic() - _sidecar_checked > 1:
        _sidecar_checked = time.monotonic()
        try:
            r = _HTTP.get(f"{ENCODER_URL}/health", timeout=2)
            _sidecar_ok = None
            available = r.ok and r.json().get("status") == "ok"
            if available:
                encoder_contract.validate(r.json())
            _sidecar_ok = available
        except requests.RequestException as e:
            _sidecar_ok = False
            log.info("encoder sidecar not reachable at %s (%s) — loading the "
                     "model in-process instead; first query pays ~30s",
                     ENCODER_URL, e)
    return _sidecar_ok


def _get_encoder() -> dict:
    """Load the query encoder once. ~28s here — weights, then graph capture.

    Never in the faiss process.
    """
    global _encoder
    if _encoder is None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dev, dtype = encoder_device.resolve(torch)
        identity = encoder_contract.identity()
        revision = {"revision": identity["revision"]} if identity.get("revision") else {}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            identity["model"], dtype=dtype, **revision).eval().to(dev)
        processor = AutoProcessor.from_pretrained(identity["model"], trust_remote_code=True, **revision)
        encoder_device.use_fast_tokenise(processor, identity["model"], revision=identity.get("revision"))
        encoder_device.graph_forward(torch, model, dev)
        _encoder = {"processor": processor, "model": model, "device": dev,
                    "torch": torch, "identity": identity}
        # Record a graph per padding width before returning. Whoever asks first
        # would otherwise pay ~6s for whichever shape their query happens to
        # have, and this path is already the slow fallback — its costs belong at
        # load. Assigning _encoder first is what keeps this from recursing, and
        # calling _forward_pass rather than embed_query keeps it from queueing
        # onto the thread it is already running on.
        for width in encoder_device.warm_widths():
            _forward_pass("warmup", None, width)
    return _encoder


# pixelrag_serve wraps every query in this chat template before tokenising.
# Passing raw text instead changes the token sequence entirely and moves the
# embedding — measured as 1/10 identical rankings until this was replicated.
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."


@lru_cache(maxsize=256)
def _embed_cached(text: str, instruction: str | None) -> tuple[float, ...]:
    """Memoised encode. Tuple so the cache cannot hand out a mutable list.

    The same text gets encoded more than once per question: retrieve.py searches
    the question and its noun phrase. The encoder is deterministic for a given
    text and pinned snapshot. Exact-repeat answer caching needs no embedding.
    """
    if _sidecar_available():
        try:
            r = _HTTP.post(f"{ENCODER_URL}/embed",
                           json={"text": text, "instruction": instruction}, timeout=60)
            r.raise_for_status()
            return tuple(r.json()["embedding"])
        except requests.RequestException:
            global _sidecar_ok
            _sidecar_ok = False
            log.warning("encoder sidecar failed; encoding locally", exc_info=True)
    return tuple(_encode_local(text, instruction))


def embed_query(text: str, instruction: str | None = None, *, image: bytes | None = None) -> list[float]:
    """Encode as pixelrag_serve._encode_queries does.

    Same chat template and instruction, same base-model forward, same
    last-token pooling over last_hidden_state, same L2 normalisation. Any
    *structural* divergence puts the query in a different space from the indexed
    documents and silently degrades retrieval. The arithmetic does differ — fp16,
    padding and the CUDA graph together move an index score by up to 7.9e-04 at
    cos 0.999994, which check_parity.py exists to keep honest.
    """
    import gemini_index
    if gemini_index.enabled():
        return gemini_index.embed_query(text, image=image)
    if instruction is not None and instruction != encoder_contract.identity()["instruction"]:
        raise encoder_contract.EncoderMismatch("Query instruction differs from the indexed encoder contract.")
    if image is not None:
        if _sidecar_available():
            try:
                r = _HTTP.post(f"{ENCODER_URL}/embed", json={"text": text,
                    "instruction": instruction, "image": visual_query.data_url(image)}, timeout=120)
                r.raise_for_status()
                return r.json()["embedding"]
            except requests.RequestException:
                global _sidecar_ok
                _sidecar_ok = False
                log.warning("image encoder sidecar failed; encoding locally", exc_info=True)
        def local_image():
            state = _get_encoder()
            return visual_query.encode(state, text, image,
                                       instruction or state["identity"]["instruction"])
        return encoder_device.on_model_thread(local_image)
    return list(_embed_cached(text, instruction))


def _encode_local(text: str, instruction: str | None = None) -> list[float]:
    """embed_query's in-process case, queued onto the thread that owns the model.

    Which thread runs the forward is not a detail here: it is the thread that
    recorded the CUDA graphs, and only that thread may replay them. See
    encoder_device.on_model_thread.
    """
    return encoder_device.on_model_thread(_forward_pass, text, instruction, 0)


def _forward_pass(text: str, instruction: str | None,
                  pad_width: int) -> list[float]:
    """The encode itself — same steps, same order as pixelrag_serve does them.

    Runs on the model thread. Every numerical decision it makes comes from
    encoder_device, so this path and the sidecar's cannot drift apart; pad_width
    is how _get_encoder records a graph for a length no query it has happens to
    produce.
    """
    enc = _get_encoder()
    torch, dev = enc["torch"], enc["device"]
    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": enc["identity"]["instruction"] if instruction is None else instruction}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    prompt = enc["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = enc["processor"](text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs, forward = encoder_device.prepare_forward(torch, enc["model"],
                                                     enc["processor"], inputs,
                                                     pad_width)
    with torch.no_grad():
        out = forward(**inputs)
    h = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()[0].tolist()


def warm_encoder() -> None:
    """Pay the model load at startup rather than on a user's first question."""
    if LOCAL_ENCODE:
        try:
            embed_query("warmup")
        except Exception:
            log.warning("query encoder failed to warm up; visual queries will "
                        "fail unless the encoder recovers", exc_info=True)

