"""Sparse retrieval via BM25 (rank_bm25)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ragv._text import tokenize as _tok
from ragv.models import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks]) if chunks else None

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in ranked[:k]]
