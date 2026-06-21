"""Verify each claim's citations against the retrieved spans.

A citation is verified iff (a) every cited chunk_id is actually in the retrieved
set AND (b) the cited span semantically supports the claim (cosine >= threshold).
Hallucinated chunk_ids are dropped; unsupported claims are stripped or flagged.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ragv.config import Config
from ragv.ingest.embedder import Embedder
from ragv.models import Answer, Citation, RetrievedSpan


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def verify_answer(
    claims: list[Citation],
    spans: list[RetrievedSpan],
    embedder: Embedder,
    config: Config,
) -> Answer:
    span_text = {s.chunk_id: s.text for s in spans}
    retrieved_ids = set(span_text)

    verified: list[Citation] = []
    unsupported: list[str] = []

    for claim in claims:
        # Drop any cited id that wasn't actually retrieved (hallucinated citation).
        valid_ids = [cid for cid in claim.cited_chunk_ids if cid in retrieved_ids]
        if claim.cited_chunk_ids and not valid_ids:
            logger.warning("hallucinated citation(s) {} dropped", claim.cited_chunk_ids)

        support = 0.0
        if valid_ids:
            claim_emb = embedder.embed(claim.claim_text)
            support = max(_cosine(claim_emb, embedder.embed(span_text[cid])) for cid in valid_ids)

        is_verified = bool(valid_ids) and support >= config.support_threshold
        claim.cited_chunk_ids = valid_ids
        claim.support_score = round(support, 4)
        claim.verified = is_verified

        if is_verified:
            verified.append(claim)
        else:
            unsupported.append(claim.claim_text)

    if config.unsupported_policy == "strip":
        kept = verified
        text = " ".join(
            f"{c.claim_text} [{', '.join(c.cited_chunk_ids)}]" for c in verified
        )
    else:  # flag
        kept = [c for c in claims]
        text = " ".join(
            f"{c.claim_text} [{', '.join(c.cited_chunk_ids)}]"
            + ("" if c.verified else " (UNVERIFIED)")
            for c in claims
        )

    confidence = (
        sum(c.support_score for c in verified) / len(verified) if verified else 0.0
    )
    return Answer(
        text=text.strip(),
        citations=kept,
        unsupported_claims=unsupported,
        confidence=round(confidence, 4),
    )
