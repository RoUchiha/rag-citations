"""Token-aware chunking with overlap and stable, deterministic chunk ids.

Tokens are approximated by whitespace words (provider-agnostic, no tokenizer
dep). Chunk ids are derived from the doc_id + start index, so re-ingesting the
same document yields identical ids — ingestion is idempotent.
"""

from __future__ import annotations

from ragv.models import Chunk, Document


def chunk_document(doc: Document, chunk_tokens: int = 120, overlap: int = 20) -> list[Chunk]:
    words = doc.text.split()
    if not words:
        return []
    step = max(1, chunk_tokens - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_tokens]
        if not window:
            break
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{start}",
                doc_id=doc.doc_id,
                text=text,
                token_count=len(window),
            )
        )
        if start + chunk_tokens >= len(words):
            break
    return chunks
