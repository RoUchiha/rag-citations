"""Gate 7: end-to-end ingest -> ask -> verified citations; no-answer path."""

from __future__ import annotations

import pytest

from ragv.config import Config
from ragv.generate.provider import HeuristicProvider
from ragv.ingest.embedder import HashingEmbedder
from ragv.ingest.loader import load_text
from ragv.pipeline import RagPipeline

DOCS = [
    load_text("The mitochondria is the powerhouse of the cell. It produces ATP.", "bio"),
    load_text("Photosynthesis occurs in chloroplasts and converts light to energy.", "bio2"),
    load_text("Mount Everest is the tallest mountain above sea level.", "geo"),
]


def _pipe(policy="strip"):
    return RagPipeline(HashingEmbedder(), HeuristicProvider(), Config(unsupported_policy=policy))


@pytest.mark.asyncio
async def test_e2e_relevant_question_verified_citations():
    pipe = _pipe()
    pipe.ingest(DOCS)
    ans = await pipe.ask("what is the powerhouse of the cell")
    assert not ans.no_answer
    assert ans.citations and all(c.verified for c in ans.citations)
    assert "[" in ans.text  # carries inline citation(s)


@pytest.mark.asyncio
async def test_e2e_irrelevant_question_no_answer():
    pipe = _pipe()
    pipe.ingest(DOCS)
    ans = await pipe.ask("how do rocket engines work in orbit")
    assert ans.no_answer is True


@pytest.mark.asyncio
async def test_empty_corpus_no_answer():
    pipe = _pipe()
    pipe.ingest([])
    ans = await pipe.ask("anything")
    assert ans.no_answer is True
