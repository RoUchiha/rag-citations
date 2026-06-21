"""Config: chunking, retrieval, fusion, verification (YAML -> Pydantic)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # Chunking (token-aware; tokens approximated by whitespace words).
    chunk_tokens: int = 120
    chunk_overlap: int = 20

    # Retrieval / fusion.
    top_k: int = 5
    rrf_k: int = 60                      # RRF damping constant

    # Verification.
    support_threshold: float = 0.55     # cosine claim<->span to count as supported
    unsupported_policy: str = "strip"   # "strip" | "flag"

    # No-answer path. Relevance is judged by the best *dense cosine* (or any BM25
    # lexical hit), not the tiny RRF fused score.
    min_retrieval_score: float = 0.15   # below this dense cosine (and no BM25 hit) -> no answer

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)
