"""Gate 2: loader + token-aware chunker with stable, idempotent ids."""

from __future__ import annotations

from ragv.ingest.chunker import chunk_document
from ragv.ingest.loader import load_file, load_text

TEXT = " ".join(f"word{i}" for i in range(300))


def test_load_text_stable_doc_id():
    a = load_text(TEXT, source="s")
    b = load_text(TEXT, source="s")
    assert a.doc_id == b.doc_id  # deterministic


def test_chunking_overlap_and_ids():
    doc = load_text(TEXT, source="s")
    chunks = chunk_document(doc, chunk_tokens=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].token_count == 100
    # stable, deterministic ids
    assert chunks[0].chunk_id == f"{doc.doc_id}:0"
    assert chunks[1].chunk_id == f"{doc.doc_id}:80"  # step = 100 - 20


def test_reingest_idempotent():
    doc = load_text(TEXT, source="s")
    ids1 = [c.chunk_id for c in chunk_document(doc, 100, 20)]
    ids2 = [c.chunk_id for c in chunk_document(doc, 100, 20)]
    assert ids1 == ids2


def test_load_file(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nsome content here", encoding="utf-8")
    doc = load_file(p)
    assert "content" in doc.text and doc.source.endswith("doc.md")
