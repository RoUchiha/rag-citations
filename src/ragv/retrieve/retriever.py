"""Hybrid retriever: fuse dense + BM25 rankings with RRF, return top-k spans."""

from __future__ import annotations

from ragv.config import Config
from ragv.models import Chunk, RetrievedSpan
from ragv.retrieve.dense import DenseRetriever
from ragv.retrieve.fusion import reciprocal_rank_fusion
from ragv.retrieve.sparse import BM25Retriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, config: Config) -> None:
        self.dense = dense
        self.sparse = sparse
        self.config = config

    def retrieve(self, query: str) -> list[RetrievedSpan]:
        pool = max(self.config.top_k * 4, 20)
        dense_hits = self.dense.rank(query, pool)
        sparse_hits = self.sparse.rank(query, pool)

        chunk_map: dict[str, Chunk] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        for c, s in dense_hits:
            chunk_map[c.chunk_id] = c
            dense_scores[c.chunk_id] = s
        for c, s in sparse_hits:
            chunk_map[c.chunk_id] = c
            sparse_scores[c.chunk_id] = s

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in dense_hits], [c.chunk_id for c, _ in sparse_hits]],
            k=self.config.rrf_k,
        )
        ordered = sorted(fused, key=lambda cid: -fused[cid])[: self.config.top_k]
        return [
            RetrievedSpan(
                chunk_id=cid,
                text=chunk_map[cid].text,
                dense_score=dense_scores.get(cid, 0.0),
                sparse_score=sparse_scores.get(cid, 0.0),
                fused_score=fused[cid],
            )
            for cid in ordered
        ]
