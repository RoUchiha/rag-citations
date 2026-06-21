# DECISIONS

Assumptions and deviations made during autonomous execution, dated.

## 2026-06-20

- **Tokens approximated by whitespace words** for chunking — provider-agnostic, no
  tokenizer dependency. Chunk ids are `{doc_id}:{start_index}`, so re-ingesting a
  document is idempotent (same ids overwrite).
- **Default store is in-memory numpy cosine**; ChromaDB is an optional backend behind
  the same `VectorStore` protocol (the `chroma` extra). Keeps tests offline and light.
- **Default embedder is HashingEmbedder** (bag-of-words); real semantic retrieval uses
  sentence-transformers (the `embeddings`/`demo` extra). Tests use a scripted
  `MockEmbedder` for precise control of dense scores.
- **No-answer is judged by dense cosine**, not the RRF fused score. RRF scores are tiny
  (`~1/k`) and not meaningful as absolute confidence, so relevance uses the best dense
  cosine (or any BM25 lexical hit).
- **Offline generation is extractive.** Since the offline default has no LLM, the
  `HeuristicProvider` builds the answer from the retrieved spans that best overlap the
  question and cites them — genuinely grounded and verifiable, not scripted. Real models
  plug in via the provider abstraction.
- **CLI `ask` ingests inline via `--path`** rather than requiring a persisted collection,
  because the default store is in-process. Persisted collections work with the Chroma
  backend.
- **Verification support = max cosine** between the claim and its cited spans; a claim is
  verified only if every surviving cited id was actually retrieved AND support ≥ threshold
  (default 0.55). Under `strip`, unsupported claims are removed entirely.
