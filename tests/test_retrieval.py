"""Gates 3-4: dense ordering, RRF math, and hybrid > pure-dense on a keyword query."""

from __future__ import annotations

from ragv.config import Config
from ragv.ingest.embedder import MockEmbedder
from ragv.models import Chunk
from ragv.retrieve.dense import DenseRetriever
from ragv.retrieve.fusion import reciprocal_rank_fusion
from ragv.retrieve.retriever import HybridRetriever
from ragv.retrieve.sparse import BM25Retriever
from ragv.store.memory import InMemoryVectorStore

# K: keyword match but semantically far. D: semantically close, no keyword. N: unrelated.
K = Chunk(chunk_id="k", doc_id="x", text="Error code XZ9 indicates a hard drive failure.",
          token_count=8)
D = Chunk(chunk_id="d", doc_id="x", text="A hard drive can stop working and lose data.",
          token_count=9)
N = Chunk(chunk_id="n", doc_id="x", text="Bananas are a good source of potassium.",
          token_count=7)
QUERY = "what does XZ9 mean for my hard drive"

VECTORS = {
    QUERY: [1.0, 0.0, 0.0],
    D.text: [0.95, 0.05, 0.0],   # close to query
    K.text: [0.0, 1.0, 0.0],     # far from query (keyword chunk)
    N.text: [0.0, 0.0, 1.0],     # unrelated
}


def _build():
    emb = MockEmbedder(vectors=VECTORS, dim=8)
    store = InMemoryVectorStore()
    chunks = [K, D, N]
    store.add(chunks, emb.embed_batch([c.text for c in chunks]))
    return emb, store, chunks


def test_rrf_math():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert fused["a"] == fused["b"]  # symmetric
    fused2 = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert fused2["a"] > fused2["b"] > fused2["c"]


def test_dense_orders_by_cosine():
    emb, store, _ = _build()
    ranked = DenseRetriever(store, emb).rank(QUERY, k=3)
    assert ranked[0][0].chunk_id == "d"   # semantically closest
    assert ranked[-1][0].chunk_id in {"k", "n"}  # keyword chunk is NOT dense-top


def test_hybrid_surfaces_keyword_chunk():
    emb, store, chunks = _build()
    dense = DenseRetriever(store, emb)
    sparse = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, sparse, Config(top_k=2, rrf_k=60))
    spans = hybrid.retrieve(QUERY)
    ids = [s.chunk_id for s in spans]
    # Pure dense would rank K last; hybrid surfaces it via BM25 keyword match.
    assert "k" in ids
    kspan = next(s for s in spans if s.chunk_id == "k")
    assert kspan.sparse_score > 0
