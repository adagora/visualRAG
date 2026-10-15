# Attention Is All You Need

An optional public-document example using the same Gemini-only app. Follow the root [setup instructions](../../README.md) first.

Download [arXiv:1706.03762v7](https://arxiv.org/pdf/1706.03762v7) to `examples/attention/data/1706.03762v7.pdf` and check it against [SHA256SUMS](SHA256SUMS).

```powershell
uv run pixelrag-index examples/attention/data/1706.03762v7.pdf --output examples/attention/index --title "Attention Is All You Need"
uv run pixelrag-serve --index examples/attention/index
```

Indexing sends one page image per billed Gemini embedding request. Questions use one query embedding and one reader call over at most four pages. Rendering uses PyMuPDF, and vector search runs locally with NumPy.

Questions to try:

1. What does Figure 1 show about the encoder and decoder?
2. Explain scaled dot-product attention and the square-root scaling factor.
3. How does multi-head attention extend scaled dot-product attention?
4. Compare the base and big Transformer in Table 2.
5. What optimizer and warmup schedule did the authors use?

The source PDF, generated images and vectors are local, git-ignored artifacts. The old five-question benchmark and its historical results belong to the archived research app; they are not measurements of the current pipeline.
