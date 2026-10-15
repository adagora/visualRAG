# Visual RAG — PixelRAG MVP

Find evidence in technical PDFs by searching rendered page images, then answer from those images. Text retrieval is a control and optional enhancement.

The main interface has three modes: **visual**, **BM25**, and **hybrid**. Ask a question or attach a screenshot/crop. Retrieved pages appear before the answer. The reader may request one high-resolution crop from an attached source; the answer links back to its evidence.

## Run

Python 3.12 or newer. From the repository root:

```powershell
uv pip install -e ".[gemini,dev]"
uv pip install "pixelrag[index,serve,pdf]"
```

Use a PyTorch build appropriate to your accelerator. The Qwen index builder currently requires CUDA and the direct GPU backend. Gemini page embeddings provide a separate API-based path without a local embedding model; see [the public Attention example](examples/attention/README.md).

Set a reader key in the git-ignored `.env`: `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`. Install `.[anthropic]` for the Anthropic reader. Exported environment variables take precedence.

Choose PDFs through `source.path` in your index configuration. The existing PX1 configuration points at the local 236-page garage-door catalogue:

```powershell
.venv/Scripts/python.exe scripts/build_index.py --config pixelrag.px1.yaml --geometry regions
$env:PIXELRAG_INDEX_DIR = "index-px1"
.venv/Scripts/python.exe scripts/serve_all.py
```

Open **http://127.0.0.1:8000**. The launcher starts vector search, the query encoder, and the UI. Stop that foreground launcher with Ctrl-C. Every service binds to localhost.

The builder also accepts `--output`, `--revision`, `--width`, `--height`, `--stride-x`, and `--stride-y`. `--stock` is an alias for page geometry. Adapter builds and `--skip-render` are rejected because they cannot satisfy the current query/source contract.

## What happens to a question

```text
PDF → 200-DPI page JPEGs → page/crop embeddings → FAISS
question + optional reference image → matching query encoder
FAISS hits → page ranking → selected whole-page images → vision reader
                                      ↳ optional single original-resolution crop
answer → source ID + page + evidence rectangles → PDF viewer
```

BM25 searches the PDF text layer. Hybrid combines visual and BM25 page rankings using reciprocal-rank fusion. The reader receives images in every mode. A supplied reference image helps interpret the question; it cannot be cited as retrieved evidence.

| Mode | Purpose |
|---|---|
| `visual` | Test whether image embeddings retrieve useful evidence |
| `lexical` | BM25-only control; no image/query encoder |
| `hybrid` | Practical combined retrieval |

Visual is the default. `PIXELRAG_HYBRID=1` explicitly defaults to hybrid. Merely adding a TypeSafe key never changes the retriever or adds planning/gating calls.

CLI examples:

```powershell
.venv/Scripts/python.exe scripts/ask.py "Pokaż przekrój panelu UniTherm" --retrieval visual
.venv/Scripts/python.exe scripts/ask.py "Znajdź ten profil" --image reference.png
.venv/Scripts/python.exe scripts/ask.py "Wymiary montażowe" --retrieve-only --retrieval lexical --json
.venv/Scripts/python.exe scripts/compare.py "Wymiary montażowe"
```

Screenshots support PNG/JPEG/WebP up to 5 MB. Reader budgets are explicit: four pages by default, one to twelve with `ask.py --pages` or API `pages`. Main questions are independent; automatic conversation rewriting and category/task planning are experimental opt-ins.

## Reproducible indexes

Every build uses a new staging directory and publishes a completed immutable snapshot:

```text
index-px1/
  active.json
  snapshots/<index-id>/
    manifest.json
    sources/<document-sha256>.pdf
    articles.json, text.json, index.faiss, metadata.npz
    tiles/, embeddings/
```

The manifest records original document hashes, immutable source copies, page counts, encoder model/revision/dimensions, document and query instructions, rendering, crop geometry, and vector count. Vector identities, page geometry, source copies, and counts are checked before publication. A failed build leaves the previous active pointer unchanged. Existing legacy files remain available.

The query encoder and vector service must match the snapshot. A mismatch fails explicitly. Restart the services after selecting a new snapshot; a running service remains bound to its original snapshot.

The query role uses `Retrieve images or text relevant to the user's query.`; Qwen document embeddings use the upstream native `Represent the user's input.`. These roles are recorded separately. Hugging Face weights resolve to a pinned commit before indexing.

Exact-repeat answer caching preserves numbers, product codes, units, and negation. Cache identity includes the source/index fingerprint, prompt version, retrieval configuration, budget, reader, and reference-image hash. Similar embedding vectors do not authorize reuse of an answer.

## Evidence and the second look

Citation objects identify `source_id` and 1-based `page`. Rectangles use original-page percentages: `left`, `top`, `width`, `height`. Quotes are optional for visual evidence.

A reader can return one `---ZOOM---` request naming an attached source/page and a bounded rectangle. The next call includes the whole page and an original-resolution crop with context. No additional crop request is executed. Both calls' usage is counted.

Evidence status is explicit:

- `text-located`: quoted text was found; this does not verify the answer.
- `model-region`: the reader identified a visual region.
- `unlocated`: the source is known but precise evidence is missing.
- `unsupported`: source identity is invalid or ambiguous; no document link is fabricated.

Invalid model rectangles are discarded without losing a known source link or independently located quote. Coordinates are never rescaled by guessing their units.

A text-only Jev judgment never vetoes visual/BM25/hybrid reading. Textless drawings and scans can still be answered and cited with visual rectangles.

## Evaluate the claim

[visual_questions.json](eval/visual_questions.json) has 50 directly inspected draft questions with expected facts, source hashes, evidence pages, visual-dependency labels, and page-family development/test splits. They were authored without Jev labels or retrieval rankings. **Independent human review is pending; results are exploratory.** This single-document set cannot establish performance on unseen document families.

The [scan fixture](eval/fixtures/px1-textless.pdf) contains four rendered source pages with no text layer; [its remapped questions](eval/visual_questions_scan.json) test the actual pixel path.

Local exploratory runs on the 21 positive test questions retrieved an evidence page within four pages for 21/21 questions with regions/max, versus 19/21 for BM25. Regions/max ranked evidence first for 16/21; regions/sum did so for 14/21. These are draft labels on one catalogue, not an independently audited accuracy claim. Two repeated runs do not double the number of independent questions.

On the four-page textless fixture, visual retrieval ranked evidence first for 23/25 questions; BM25 had no hits. Hit@4 is trivial with four source pages, so use this fixture to check the image path, not retrieval at scale.

The public [27-page ECHO pergola brochure](https://www.wisniowski.pl/images/foldery-online/pl/k75-pl-ECHO-wisniowski/docs/k75-pl-ECHO-wisniowski.pdf) provides a second, image-heavy comparison. Both encoders indexed identical 200-DPI page JPEGs, one vector per page. The [ten positive and two negative draft questions](eval/echo_questions.json) were authored from source inspection before inspecting retrieval rankings; the positives cover eight evidence pages.

| Retriever | Evidence first | Evidence within four |
|---|---:|---:|
| Gemini Embedding 2, 3072 dimensions | 9/10 | 10/10 |
| Local Qwen3-VL-Embedding-2B, 2048 dimensions | 7/10 | 10/10 |
| BM25 control | 5/10 | 9/10 |

For [three reference-image crops](eval/echo_image_queries.json), each encoder ranked the source page first twice and found all three within four results. These are crops of the indexed PDF, not independent user photos. With the same Gemini reader, both pipelines read the table value **726** and declined to invent a missing price. Both also added an unverified unit and failed to locate a precise citation rectangle. Labels and absence judgments still need independent human review; this small, correlated sample does not establish a general encoder winner. Cached benchmark timings are not uncached encoding latency.

Save the linked PDF at `corpora/echo/k75-pl-ECHO-wisniowski.pdf`. Build Qwen with `build_index.py --config pixelrag.echo.yaml --geometry page`, and Gemini with `gemini_index.py corpora/echo/k75-pl-ECHO-wisniowski.pdf --output index-echo-gemini --dimensions 3072` (27 billed page-image embedding requests). Run `visual_bench.py --allow-draft --eval eval/echo_questions.json --split all --rounds 1 --index <index>` against each matching backend. For Qwen, point `PIXELRAG_SEARCH_API` at a separate `search_api.py --index-dir index-echo-qwen --port 30004` service and keep the matching query encoder running. Set `PIXELRAG_EMBED_BACKEND` to `qwen` or `gemini` for the corresponding run. Gemini query embeddings and reader calls are billed; measured dollar costs are unavailable. Local run records are in `eval/results/echo-*.json`.

```powershell
.venv/Scripts/python.exe scripts/visual_bench.py --preflight --allow-draft --index index-px1
.venv/Scripts/python.exe scripts/visual_bench.py --allow-draft --index index-px1 --output eval/results/three-way.json
.venv/Scripts/python.exe scripts/answer_eval.py --allow-draft --index index-px1 --retrieval eval/results/three-way.json --output eval/results/answers-review.json
```

The retrieval benchmark uses visual/BM25/hybrid on the same corpus, four reader pages, and a matched candidate budget per branch. It reports hit@1, hit@4, MRR@24, complete evidence, errors, cold/warm timing, and per-question results. Failed queries remain in the denominator. Hybrid does more total work because it has two branches.

The answer evaluator prepares paired retrieved-page versus gold-page reads. Add `--execute` to make billed reader calls. It reports expected-fact checks and review records; passing numeric/string checks is not a general factual-correctness guarantee.

To compare geometry with one encoder:

```powershell
.venv/Scripts/python.exe scripts/build_index.py --config pixelrag.px1.yaml --geometry page --output index-px1-page
.venv/Scripts/python.exe scripts/build_index.py --config pixelrag.px1.yaml --geometry regions --output index-px1-regions
.venv/Scripts/python.exe scripts/build_index.py --config pixelrag.px1.yaml --geometry multiscale --output index-px1-multiscale
.venv/Scripts/python.exe scripts/ablate.py --page index-px1-page --regions index-px1-regions --multiscale index-px1-multiscale --allow-draft --output eval/results/geometry --execute
```

The study checks matching source hashes, encoder revision, rendering, and crop settings, then compares max versus sum aggregation. It starts the correct vector service for each snapshot and rejects runs with retrieval errors. Max aggregation is the MVP default; `PIXELRAG_AGGREGATION=sum` retains the overlap-sensitive agreement experiment.

Build the textless fixture's separate index with `build_index.py --config pixelrag.scan.yaml --geometry regions`. Launch matching services with `PIXELRAG_INDEX_DIR=index-px1-scan`, then pass `--eval eval/visual_questions_scan.json --index index-px1-scan --split all` to `visual_bench.py`. The fixture's page numbers differ from the original catalogue; its manifest records the mapping.

## Optional experiments

The [historical experiment log](experiments/README.md) preserves earlier Jev/x-ray/transcription/model studies. Its historical scores and latency are not measurements of the current code.

Explicit CLI retrieval modes retain `jev-expand`, `jev`, `jev+hybrid`, `jev-page`, and `xray`. They may incur TypeSafe usage and judge extracted text. `PIXELRAG_JEV=0` disables TypeSafe. The full mode inventory is available at `/api/modes?experiments=true`.

An image-aware reranker is an opt-in experiment: install `.[rerank]`, set `PIXELRAG_VISUAL_RERANK=1` and `PIXELRAG_RERANK_MODEL` to a local Qwen3-VL-Reranker directory, then choose `visual-rerank`. It scores at most twelve candidates, one at a time, and does not download weights automatically. Its benefit must be measured.

## Validate

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m playwright install chromium
.venv/Scripts/python.exe tests/browser_smoke.py
.venv/Scripts/python.exe tests/live_smoke.py
```

Unit tests use fake readers and small local fixtures; no billed API calls. Browser checks require the UI running on the PX1 corpus and use deterministic responses while exercising the real PDF viewer.

The live smoke check requires services on `index-px1-regions`. It encodes a real screenshot, checks its source page is retrieved, and verifies subsequent text embeddings stay unchanged. Its optional `--reader` flag makes at most three billed Gemini reader calls with catalogue page 36 and a crop. The approved live run read 4,349 PLN after one zoom and correctly reported the 6000 × 2500 mm price cell as blank. It exposed invalid reader rectangles; replaying the saved response locally verified the citation fix without another API call. `--reader --resume-reader` preserves the saved call budget when continuing an interrupted check. The optional reranker has contract tests but has not been measured with local reranker weights.

The main modules are `build_index`/`index_manifest` (snapshots), `queryembed`/`encoder`/`search_api` (matched vector search), `retrieve`/`lexical` (ranking), `rag`/`providers` (reading), `evidence`/`citeparse` (grounding), and `app`/`static/index.html` (interface).
