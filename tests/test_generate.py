"""Gate 5: grounded generation parses into claims with cited chunk ids."""

from __future__ import annotations

import pytest

from ragv.generate.generator import Generator, parse_claims
from ragv.generate.provider import MockProvider
from ragv.models import RetrievedSpan


def test_parse_claims_extracts_citations():
    answer = "The sky is blue [d:0]. Water is wet [d:1, d:2]. No citation here."
    claims = parse_claims(answer)
    assert claims[0].claim_text == "The sky is blue" and claims[0].cited_chunk_ids == ["d:0"]
    assert claims[1].cited_chunk_ids == ["d:1", "d:2"]
    assert claims[2].cited_chunk_ids == []  # uncited sentence handled gracefully


@pytest.mark.asyncio
async def test_generator_returns_parsed_claims():
    provider = MockProvider(response="Cats are mammals [c:0]. They purr [c:0].")
    spans = [RetrievedSpan(chunk_id="c:0", text="Cats are mammals that purr.")]
    raw, claims = await Generator(provider).generate("tell me about cats", spans)
    assert "[c:0]" in raw and len(claims) == 2
    assert all(c.cited_chunk_ids == ["c:0"] for c in claims)
