# Attention Is All You Need + Gemini Embedding 2

A 15-page visual RAG example you can chat with: Vaswani et al.,
[Attention Is All You Need, arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762v7).
Each page is rendered to a JPEG by upstream PixelRAG's `render_pdf`, embedded as an
image with `gemini-embedding-2` (768 dimensions), and searched by cosine similarity.
Queries use the same model. The reader, BM25 fusion, citations and web UI are the
ones the rest of this repo uses.

The only code this example adds is `scripts/gemini_index.py` (the embedding step,
which upstream has no equivalent for) and a short hook in `rag.py` and `queryembed.py`.
Everything else is a command that already exists.

## Run it

From the repository root, with the setup from the main README done and
`GEMINI_API_KEY` in `.env` (`TYPESAFE_API_KEY` is optional and only enables Jev features):

```bash
uv pip install -e '.[gemini,dev]'
brew install poppler          # upstream's PDF renderer shells out to pdftoppm

# 1. Download the pinned version and check it is the file the example was written for.
mkdir -p examples/attention/data
curl -L -o examples/attention/data/1706.03762v7.pdf https://arxiv.org/pdf/1706.03762v7
(cd examples/attention/data && shasum -a 256 -c ../SHA256SUMS)

# 2. Build the index: 15 page images, 15 embedding requests (about 20 seconds).
.venv/bin/python scripts/gemini_index.py examples/attention/data/1706.03762v7.pdf \
  --output examples/attention/index --title "Attention Is All You Need"

# 3. Chat with it at http://127.0.0.1:8000
PIXELRAG_INDEX_DIR=examples/attention/index PIXELRAG_EMBED_BACKEND=gemini \
PIXELRAG_LOCAL_ENCODE=0 PIXELRAG_PROVIDER=gemini PIXELRAG_RETRIEVAL=hybrid \
  .venv/bin/python -m uvicorn app:app --app-dir scripts --port 8000
```

The build refuses to overwrite an existing index; delete `examples/attention/index`
or pass a new `--output` to rebuild. `GEMINI_MODEL` picks the answer model
independently of the embedding model.

Questions to try:

1. Explain scaled dot-product attention and the square-root scaling factor.
2. How does multi-head attention extend scaled dot-product attention?
3. Compare the base and big Transformer in Table 2.
4. What optimizer and warmup schedule did the authors use?
5. What does Figure 1 show about the encoder and decoder?

## Check retrieval

[`questions.yaml`](questions.yaml) holds five questions with their gold pages, in the
format the repo's own benchmark reads. The pages were found by searching the PDF text,
not by a model:

```bash
PIXELRAG_INDEX_DIR=examples/attention/index PIXELRAG_EMBED_BACKEND=gemini \
PIXELRAG_LOCAL_ENCODE=0 .venv/bin/python scripts/bench.py \
  --eval examples/attention/questions.yaml --modes visual,hybrid --no-gate -k 3
```

On the build above: 4 of 5 questions have the right page at rank 1 and all 5 within
the top 3, for both `visual` and `hybrid`. With five questions one question is 20
points and the 95% interval on 80% is 38-96%, so this shows the index is wired up
correctly. It does not show how good Gemini Embedding 2 is at retrieval.

## Why the search is NumPy

Upstream `pixelrag serve` loads the Qwen3-VL model at startup even when every query
arrives pre-embedded, so it cannot serve a Gemini index without downloading that
model and installing torch. `gemini_index.py` therefore keeps the vectors in a
`.npy` file and takes one matrix product per query. That is exact and instant for a
handful of documents; for a large corpus use the FAISS path in the main README.

Gemini Embedding 2 aggregates several inputs in one request into a single vector, so
the build sends one image per request. That keeps every hit tied to one citable page.

## Licence

The repository contains the source link, checksum and questions, not the paper. The
PDF, page images and embeddings are produced locally and are git-ignored. arXiv records
the paper under its [non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/),
which grants arXiv distribution rights; it is not a licence for this repository to
redistribute the paper.

## Your own documents

```bash
.venv/bin/python scripts/gemini_index.py path/to/a.pdf path/to/b.pdf \
  --output examples/my-index --title "Doc A" --title "Doc B"
```

Then serve with `PIXELRAG_INDEX_DIR=examples/my-index` and the same environment as step 3.
Embedding follows Google's [Gemini Embedding 2 documentation](https://ai.google.dev/gemini-api/docs/embeddings).
