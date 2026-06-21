"""In-memory cosine vector store (numpy). Idempotent on chunk_id."""

from __future__ import annotations

import numpy as np

from ragv.models import Chunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._emb: dict[str, list[float]] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.chunk_id] = chunk      # overwrite -> idempotent
            self._emb[chunk.chunk_id] = emb

    def search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        if not self._emb:
            return []
        ids = list(self._emb)
        matrix = np.asarray([self._emb[i] for i in ids], dtype=float)
        q = np.asarray(query_embedding, dtype=float)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * qn + 1e-12)
        order = np.argsort(-sims)[:k]
        return [(self._chunks[ids[i]], float(sims[i])) for i in order]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def clear(self) -> None:
        self._chunks.clear()
        self._emb.clear()

    def __len__(self) -> int:
        return len(self._chunks)
