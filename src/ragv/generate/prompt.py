"""Grounded-answer prompt that enforces inline [chunk_id] citations."""

from __future__ import annotations

from ragv.models import RetrievedSpan


def build_prompt(query: str, spans: list[RetrievedSpan]) -> str:
    context = "\n".join(f"[{s.chunk_id}] {s.text}" for s in spans)
    return (
        "Answer the question using ONLY the context below. After each sentence, cite the "
        "chunk id(s) you used in square brackets, e.g. [doc:0]. If the context does not "
        "support an answer, say you don't have enough information.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
