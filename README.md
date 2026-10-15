# PixelRAG MVP

Search technical PDFs by their page images, then let Gemini answer from the retrieved pages.

```text
PDF → page JPEGs → Gemini Embedding 2 → local NumPy vectors
question + optional screenshot → one embedding → top 4 pages
top pages → one Gemini read → answer + source/page links
```

One app process. The PDF viewer, screenshot input, page previews and citations are built in.

## Run

Python 3.12+, [uv](https://docs.astral.sh/uv/), and a Gemini API key:

```powershell
uv sync --extra dev
```

Put `GEMINI_API_KEY=...` in the git-ignored `.env`. Exported environment variables take precedence. `GOOGLE_API_KEY` is also accepted; `GEMINI_MODEL` overrides the answer model.

Build your index, then start the app:

```powershell
uv run pixelrag-index path/to/document.pdf --output index
uv run pixelrag-serve --index index
```

Open **http://127.0.0.1:8000**. Ctrl-C stops the app. No GPU, Qwen, FAISS, model server or Poppler installation is required.

Indexing sends each rendered page to the Gemini embedding API, one request per page. A question sends its text/reference image for embedding, then at most four selected page images to the reader. Both steps are billed; the app does not estimate dollar costs.

To reuse the existing public ECHO index without rebuilding it:

```powershell
uv run pixelrag-serve --index index-echo-gemini
```

The [27-page ECHO brochure](https://www.wisniowski.pl/images/foldery-online/pl/k75-pl-ECHO-wisniowski/docs/k75-pl-ECHO-wisniowski.pdf) can also be saved at `corpora/echo/k75-pl-ECHO-wisniowski.pdf` and indexed with the same build command. Multiple PDF arguments produce one shared index. `--title` can be repeated once per PDF.

## Questions and evidence

Type a question or attach a PNG/JPEG/WebP screenshot up to 5 MB. Found pages appear before the answer. Click a preview or citation to open the corresponding original PDF page; citations from earlier answers continue to work.

Citation links include the source identity, so changing indexes cannot reuse another document's cached PDF. They also support opening the original source in a new tab.

Each question is independent. The reader gets one call and at most four pages. Missing prices, unreadable values and unstated units should produce an explicit limitation instead of a guess. Citations are accepted only for attached source/page pairs; this locates evidence, not a verification of every claim.

```powershell
uv run pixelrag-ask "Jakie wymiary ma ten profil?" --index index-echo-gemini
uv run pixelrag-ask "Znajdź ten element" --image reference.png --index index-echo-gemini
uv run pixelrag-ask "Przekrój słupa" --retrieve-only --index index-echo-gemini
```

`--retrieve-only` makes a billed query embedding and, when enabled, a Jev decision request, with no answer-reader call. `--json` prints the complete answer record. `PIXELRAG_INDEX_DIR` supplies a default index path for the app and CLI.

## Optional Jev image decisions

The base pixel pipeline is fully integrated: page images are embedded directly and the selected original page images go to Gemini. The [PixelRAG paper](https://arxiv.org/abs/2606.28344) and [official implementation](https://github.com/StarTrail-org/PixelRAG) motivate this architecture; their Qwen/FAISS runtime is not required for this Gemini implementation.

Jev is available as an experimental evidence-page selector in the same CLI and web flow. Add `TYPESAFE_API_KEY=...` to `.env`, then opt in:

```powershell
uv run pixelrag-serve --index index-echo-gemini --jev
uv run pixelrag-ask "Jakie wymiary ma ten profil?" --index index-echo-gemini --jev --json
uv run pixelrag-ask --image reference.png --index index-echo-gemini --jev --retrieve-only
```

`PIXELRAG_JEV=rerank` enables it through the environment or `.env`; the default is `off`. `--no-jev` overrides that setting. Having a TypeSafe key alone does not enable extra requests.

When enabled, the existing embedding retrieves eight unique source/page candidates. One request to `https://api.typesafe.ai/v1/systemone` asks one Noul relevance question per candidate, then code stably selects four pages. Each candidate image appears in its own question's instructions; only the query and optional reference image are shared state. There is no pairwise tournament. Jev sees baseline JPEG thumbnails capped at 1024 pixels, padded when needed to meet its minimum size. Gemini still reads the original indexed JPEGs. No OCR or PDF text extraction is added.

The default model is pinned to **`jev-1.14.0`**, released October 14, 2026. `TYPESAFE_MODEL` can override it. At review time, `jev-preview` resolves to 1.14 while `jev-latest` still resolves to 1.13; simply retaining the old alias would not update the model. See [TypeSafe models](https://docs.typesafe.ai/models).

The integration sends the required `TypeSafe-Danger-Zone-Images: true` header. [Images](https://docs.typesafe.ai/images) work experimentally on both 1.13 and 1.14, including in instructions and criteria; 1.14 additionally supports [Box localization](https://docs.typesafe.ai/primitives/box). TypeSafe describes these image features as prototypes. Enabling Jev sends the query, reference image and up to eight candidate thumbnails to TypeSafe, with additional billed input tokens and latency. A timeout, API error, missing key or invalid decision preserves the original dense ranking. The UI and JSON report fallback explicitly. Cancellation still stops before the next operation.

Feature decisions after reviewing the two posts and current official docs:

| Capability | Decision | Reason |
| --- | --- | --- |
| Images in questions, typed Noul decisions | Integrated, opt-in | Select evidence directly from pixels in one bounded request. |
| Pairwise Choice / Bradley–Terry ranking | Defer | More comparisons; no matched evidence here that the extra work improves this pipeline. |
| Box localization | Defer | Full-page evidence and PDF links already work. Box always returns a box, including for an absent target, so a future crop flow also needs a presence check. |
| Experimental 262k long context | Defer | The bounded thumbnail request fits normal context; larger context adds documented latency and accuracy costs. |
| Hard answerability/confidence gate | Defer | Relevance probabilities have not been calibrated for these documents and cannot certify answer grounding. |

The [exrhizo announcement](https://x.com/exrhizo/status/2110546960767992105) highlights images *inside* questions. The [kabgaspard reply](https://x.com/kabgaspard/status/2109649433923088598), responding to a [pairwise-ranking suggestion](https://x.com/dotpem/status/2109474543249264878), asks for tests that justify quadratic comparisons. Both informed the bounded, independent-question implementation; neither post supplies a benchmark for this corpus.

A matched live run on **October 15, 2026** used the existing ECHO snapshot `0223d15397024cdeaf54219ea4ef252b` and the unchanged datasets below. The PDF SHA256 matched the manufacturer's public download before execution. Each comparison reused exactly one query embedding and its eight candidates:

| Dataset | Dense evidence first | Jev evidence first | Evidence within four, both |
| --- | --- | --- | --- |
| 10 positive text questions | 9/10 | 10/10 | 10/10 |
| 3 source-crop image queries | 2/3 | 3/3 | 3/3 |

All 13 Jev requests applied successfully with no fallback. Jev preparation plus request time averaged about 1.4 seconds per case, with 228,111 recorded Jev input tokens in total. Full 1568-pixel thumbnails exceeded the API context limit in this batch; 1024-pixel thumbnails resolved that without enabling long context. These small, agent-inspected draft questions and self-crops do not establish general gains or improved answer accuracy. **Keep Jev optional:** the observed rank-one improvement did not improve the four-page evidence coverage already achieved by the base retriever. Local reports are saved under `eval/results/jev-1.14-{text,image}-comparison.json`.

Three additional real web/SSE checks exercised the complete Jev-to-Gemini path: the post dimensions cited page 4, a missing-price question abstained from giving a price, and a Slide Glass profile screenshot cited page 24. Each made one reader call and citations stayed within attached pages. These are smoke checks, not a complete factual-grounding evaluation; the records are in `eval/results/jev-1.14-end-to-end.json`.

## Storage and checks

An index contains page JPEGs, a NumPy vector matrix, page/source metadata and immutable source PDF copies. New builds publish a completed snapshot through `active.json`; failed builds retain the previous active snapshot. Source hashes, dimensions and page identities are checked when opening it. Restart the app to select a newly built snapshot.

New builds render directly with PyMuPDF at 200 DPI/JPEG85 and use 3072-dimensional Gemini embeddings. Existing 768/1536/3072-dimensional Gemini snapshots remain readable.

The fixed [ECHO questions](eval/echo_questions.json) include ten positive cases across eight evidence pages and two missing-information cases. They are agent-inspected drafts, with independent human review pending. The previous matched page-image comparison ranked evidence first for Gemini on 9/10 questions, Qwen on 7/10, and BM25 on 5/10. These exploratory results preceded this cleanup and do not establish a universal winner. The three [image cases](eval/echo_image_queries.json) are source crops, not independent user photos.

```powershell
uv run pixelrag-eval --index index-echo-gemini
uv run pixelrag-eval --index index-echo-gemini --execute --output eval/results/retrieval.json
uv run pixelrag-eval --index index-echo-gemini --eval eval/echo_image_queries.json
uv run pixelrag-eval --index index-echo-gemini --execute --compare-jev --output eval/results/jev-text.json
uv run pixelrag-eval --index index-echo-gemini --eval eval/echo_image_queries.json --execute --compare-jev --output eval/results/jev-images.json
uv run pytest
uv run playwright install chromium
uv run python tests/browser_smoke.py
```

Evaluation without `--execute` checks source hashes, gold pages and image crops without API calls. Execution makes one billed embedding per positive question; `--jev` or `--compare-jev` additionally makes a Jev request per case. Evaluation defaults to dense retrieval regardless of `PIXELRAG_JEV`. Compare mode shares the exact embedding and candidates between both rankings and reports fallback separately. An execution with embedding errors or Jev fallbacks exits nonzero after writing the report.

Metrics include evidence first, any evidence within four, all required evidence within four, and mean gold-page recall. Entries in a question's `gold` list represent collectively required source/page pairs. Errors count as misses. Negative cases remain excluded because retrieval rankings cannot establish reader abstention. Image cases generate real PNG crops from the indexed JPEGs and follow the reference-image normalization path. Browser checks use deterministic answers and real PDFs, including a warm-cache document change and Jev success/fallback messages, without model calls.

## Code

Eight modules under `scripts/`: `gemini_index` (images, embeddings and search), `index_manifest` (safe publication), `visual_query` (reference images), `jev` (optional visual decisions), `rag` (shared retrieval and one reader call), `app`, `ask`, and `evaluate`. Jev uses the documented HTTP API directly, so there is no TypeSafe SDK dependency to upgrade.

The earlier Qwen/CUDA/FAISS stack, hybrid retrieval, text-only Jev/x-ray, pairwise rerankers, planners, caches, browsing agents and larger evaluation suite are preserved under [experiments/legacy](experiments/legacy/ARCHIVE.md). They are outside the main package and tests. The [historical experiment log](experiments/README.md) remains available.
