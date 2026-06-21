"""Pydantic v2 models for the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    metadata: dict = Field(default_factory=dict)


class RetrievedSpan(BaseModel):
    chunk_id: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    claim_text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    verified: bool = False


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    no_answer: bool = False
