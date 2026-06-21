# ragv — Hybrid RAG with Verified Citations

**⚡ [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/rosingh/ai-ml-portfolio-demos)** (RAG tab) — ask a grounded question and see every citation verified.

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
