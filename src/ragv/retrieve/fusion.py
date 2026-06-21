"""Reciprocal Rank Fusion.

RRF combines rankings without normalizing incompatible score scales: each list
contributes 1/(k + rank) for each item, summed across lists. Higher = better.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
