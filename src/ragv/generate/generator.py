"""Generate an answer and parse it into claims with cited chunk ids."""

from __future__ import annotations

import re

from ragv.generate.prompt import build_prompt
from ragv.generate.provider import GenProvider
from ragv.models import Citation, RetrievedSpan

_CITE = re.compile(r"\[([^\]]+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def parse_claims(answer: str) -> list[Citation]:
    """Split an answer into sentence-level claims with their cited chunk ids."""
    claims: list[Citation] = []
    for sentence in _SENT.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        cited = []
        for group in _CITE.findall(sentence):
            cited.extend(cid.strip() for cid in group.split(",") if cid.strip())
        claim_text = re.sub(r"\s+", " ", _CITE.sub("", sentence)).strip().rstrip(".").strip()
        claims.append(Citation(claim_text=claim_text, cited_chunk_ids=cited))
    return claims


class Generator:
    def __init__(self, provider: GenProvider) -> None:
        self.provider = provider

    async def generate(self, query: str, spans: list[RetrievedSpan]) -> tuple[str, list[Citation]]:
        raw = await self.provider.generate(build_prompt(query, spans))
        return raw, parse_claims(raw)
