"""End-to-end RAG pipeline: ingest -> hybrid retrieve -> generate -> verify.

The no-answer path short-circuits before generation when nothing relevant is
retrieved, so the model is never asked to answer from thin air.
"""

from __future__ import annotations

from loguru import logger

from ragv.config import Config
from ragv.generate.generator import Generator
from ragv.generate.provider import GenProvider
from ragv.ingest.chunker import chunk_document
from ragv.ingest.embedder import Embedder
from ragv.models import Answer, Document
from ragv.retrieve.dense import DenseRetriever
from ragv.retrieve.retriever import HybridRetriever
from ragv.retrieve.sparse import BM25Retriever
from ragv.store.base import VectorStore
from ragv.store.memory import InMemoryVectorStore
from ragv.verify.citation_verifier import verify_answer


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        provider: GenProvider,
        config: Config | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.config = config or Config()
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()
        self.generator = Generator(provider)
        self._bm25 = BM25Retriever([])

    def ingest(self, docs: list[Document]) -> int:
        chunks = []
        for doc in docs:
            chunks += chunk_document(doc, self.config.chunk_tokens, self.config.chunk_overlap)
        if chunks:
            self.store.add(chunks, self.embedder.embed_batch([c.text for c in chunks]))
        self._bm25 = BM25Retriever(self.store.chunks())  # rebuild over all chunks
        logger.info("ingested {} chunks (store size {})", len(chunks), len(self.store))
        return len(chunks)

    async def ask(self, query: str) -> Answer:
        hybrid = HybridRetriever(
            DenseRetriever(self.store, self.embedder), self._bm25, self.config
        )
        spans = hybrid.retrieve(query)

        max_dense = max((s.dense_score for s in spans), default=0.0)
        max_sparse = max((s.sparse_score for s in spans), default=0.0)
        if not spans or (max_dense < self.config.min_retrieval_score and max_sparse <= 0):
            logger.info("no relevant evidence (dense={:.3f}, sparse={:.3f})", max_dense, max_sparse)
            return Answer(text="Insufficient evidence to answer.", no_answer=True, confidence=0.0)

        _raw, claims = await self.generator.generate(query, spans)
        answer = verify_answer(claims, spans, self.embedder, self.config)

        # If the strip policy removed every claim, there is no supported answer.
        if self.config.unsupported_policy == "strip" and not answer.citations:
            answer.no_answer = True
            answer.text = "Insufficient supported evidence to answer."
        return answer
