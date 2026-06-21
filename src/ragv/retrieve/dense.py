"""Dense retrieval over the vector store."""

from __future__ import annotations

from ragv.ingest.embedder import Embedder
from ragv.models import Chunk
from ragv.store.base import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def rank(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed(query), k)
