"""Gate 6: citation verification — hallucinated/unsupported citations removed."""

from __future__ import annotations

from ragv.config import Config
from ragv.ingest.embedder import HashingEmbedder
from ragv.models import Citation, RetrievedSpan
from ragv.verify.citation_verifier import verify_answer

SPANS = [RetrievedSpan(chunk_id="d:0", text="the mitochondria is the powerhouse of the cell")]
EMB = HashingEmbedder()


def test_supported_citation_verified():
    claims = [Citation(claim_text="the mitochondria is the powerhouse of the cell",
                       cited_chunk_ids=["d:0"])]
    ans = verify_answer(claims, SPANS, EMB, Config(unsupported_policy="strip"))
    assert len(ans.citations) == 1 and ans.citations[0].verified
    assert not ans.unsupported_claims


def test_hallucinated_citation_stripped():
    # cites a chunk_id that was never retrieved
    claims = [Citation(claim_text="dogs are mammals", cited_chunk_ids=["d:99"])]
    ans = verify_answer(claims, SPANS, EMB, Config(unsupported_policy="strip"))
    assert ans.citations == []
    assert ans.unsupported_claims == ["dogs are mammals"]
    assert "dogs" not in ans.text  # stripped from output


def test_unsupported_claim_stripped():
    # cited id IS retrieved but the claim is semantically unrelated to the span
    claims = [
        Citation(claim_text="the mitochondria is the powerhouse of the cell",
                 cited_chunk_ids=["d:0"]),
        Citation(claim_text="the ocean is full of salt water", cited_chunk_ids=["d:0"]),
    ]
    ans = verify_answer(claims, SPANS, EMB, Config(unsupported_policy="strip"))
    assert len(ans.citations) == 1
    assert "ocean" not in ans.text
    assert "the ocean is full of salt water" in ans.unsupported_claims


def test_flag_policy_keeps_but_marks():
    claims = [Citation(claim_text="the ocean is full of salt water", cited_chunk_ids=["d:0"])]
    ans = verify_answer(claims, SPANS, EMB, Config(unsupported_policy="flag"))
    assert "UNVERIFIED" in ans.text
