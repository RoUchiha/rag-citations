"""Embedding abstraction: hashing (default/tests), mock (scripted), and
sentence-transformers (real, used by the demo via the `embeddings` extra)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ragv._text import tokenize


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim)
        for tok in tokenize(text):
            vec[hash(tok) % self.dim] += 1.0
        return _l2(vec).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class MockEmbedder:
    """Scripted vectors for precise tests; hashing fallback otherwise."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, dim: int = 8) -> None:
        self.vectors = vectors or {}
        self._fallback = HashingEmbedder(dim=dim)

    def embed(self, text: str) -> list[float]:
        if text in self.vectors:
            return _l2(np.array(self.vectors[text], dtype=float)).tolist()
        return self._fallback.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedder:  # pragma: no cover - heavy optional dep
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        return np.asarray(self._m.encode(text, normalize_embeddings=True), dtype=float).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        arr = self._m.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=float).tolist() for v in arr]
