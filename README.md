# ragv — Hybrid RAG with Verified Citations

**⚡ [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/rosingh/ai-ml-portfolio-demos)** (RAG tab) — ask a grounded question and see every citation verified.

## 🧠 In plain English

**The problem:** "RAG" means giving an AI your documents so it answers from them instead of from memory (which it makes up). Two things go wrong: it can't find the right document, or it writes a confident answer with a **fake citation** — pointing to a source that doesn't actually say what it claims.

**The fix (analogy):** a research assistant who (a) searches both by *topic* and by *exact keywords*, and (b) shows you the highlighted sentence backing every claim — and crosses out any claim they can't back up.

**How it works:** documents are split into chunks → a question runs a **meaning search** (embeddings) and a **keyword search** (BM25) *at the same time* → the two result lists are **fused** into one ranking → the AI answers, tagging each sentence with the chunk it used → a **verifier** strips any citation that isn't really supported → if nothing relevant was found, it says *"insufficient evidence"* instead of inventing.

**Why both halves matter:** meaning-search and keyword-search miss in *opposite* ways (embeddings miss exact terms like error codes; keywords miss paraphrases). Combining them catches both — and verification makes the answer *actually* sourced, not just sourced-looking.

A retrieval-augmented generation pipeline that (a) retrieves with **hybrid search**
(dense embeddings + sparse BM25, fused with **Reciprocal Rank Fusion**), (b) generates
grounded answers with inline `[chunk_id]` citations, and (c) **verifies every citation** —
each claim must map to a real retrieved span that actually supports it. **No hallucinated
citations ship.**

## Why hybrid + verification

- **Dense** retrieval captures meaning but misses exact keywords (IDs, error codes, rare
  terms); **BM25** nails lexical matches but misses paraphrase. **RRF** fuses them without
  reconciling incompatible score scales.
- **Citation verification** is the integrity gate: a cited `[chunk_id]` must be in the
  retrieved set *and* semantically support the claim. Hallucinated or unsupported
  citations are **stripped** (or flagged) before the answer is returned.
- **No-answer path**: if nothing relevant is retrieved, it returns "insufficient evidence"
  rather than inventing one.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -e ".[dev]"

# Ingest a folder and ask (offline: hashing embeddings + extractive generator)
ragv ask --path ./docs --q "what causes a disk failure?" --policy strip
ragv eval --qa-file qa.yaml
```

Tests and the CLI run **offline** with a deterministic hashing embedder and an
extractive generator. The live demo uses real sentence-transformer embeddings
(`pip install -e ".[embeddings]"`); ChromaDB is an optional store (`".[chroma]"`).

## Tests

```bash
pytest                 # all gates, offline
pytest --cov=ragv
```

Includes the integrity test: a **planted hallucinated citation is stripped**, and a
fully-supported answer keeps all citations.

## Layout

```
src/ragv/
  config.py        # chunking, retrieval, fusion, verification thresholds
  models.py        # Chunk, Document, RetrievedSpan, Citation, Answer
  ingest/          # loader (txt/md/pdf), token-aware chunker (stable ids), embedder
  store/           # VectorStore protocol: in-memory (numpy) + optional Chroma
  retrieve/        # dense, sparse (BM25), RRF fusion, hybrid orchestrator
  generate/        # grounded prompt, provider (mock/heuristic/real), citation parser
  verify/          # citation verifier (claim<->span support)
  pipeline.py      # ingest -> retrieve -> generate -> verify, with no-answer path
  cli.py           # ingest / ask / eval
```

See [DECISIONS.md](DECISIONS.md) for tuning (RRF k, thresholds) and trade-offs.
