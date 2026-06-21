"""Shared tokenization with stopword removal.

Stopwords matter for relevance: without filtering, a query and an unrelated
document collide on words like "in"/"the", manufacturing false relevance (and
defeating the no-answer path). Used by both the hashing embedder and BM25.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an the is are was were be been being of for to in on at and or do does did
    how what why when where who which my your his her its our their this that these
    those with as by from it i you we they he she me him them us not no yes can could
    will would should may might must have has had if then than so such about into over
    under out up down""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
