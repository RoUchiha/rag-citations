"""LLM provider for generation. MockProvider returns scripted answers offline;
real providers are import-safe (lazy SDK import)."""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

_CTX_LINE = re.compile(r"\[([^\]]+)\]\s*(.+)")
_TOK = re.compile(r"[a-z0-9]+")


@runtime_checkable
class GenProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...


class HeuristicProvider:
    """Offline extractive generator: cites the retrieved spans that best overlap
    the question. Produces genuinely grounded, verifiable answers without an LLM —
    used by the CLI and the live demo."""

    def __init__(self, max_claims: int = 2) -> None:
        self.max_claims = max_claims

    async def generate(self, prompt: str) -> str:
        context, _, tail = prompt.partition("QUESTION:")
        question = tail.replace("ANSWER:", "").strip().lower()
        q_tokens = set(_TOK.findall(question))

        spans: list[tuple[str, str]] = []
        for line in context.splitlines():
            m = _CTX_LINE.match(line.strip())
            if m:
                spans.append((m.group(1).strip(), m.group(2).strip()))

        scored = sorted(
            spans,
            key=lambda s: len(q_tokens & set(_TOK.findall(s[1].lower()))),
            reverse=True,
        )
        scored = [s for s in scored if q_tokens & set(_TOK.findall(s[1].lower()))]
        if not scored:
            return "I don't have enough information to answer."

        sentences = []
        for cid, text in scored[: self.max_claims]:
            # Cite ONE sentence per span so the citation stays attached to its claim.
            first_sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0].rstrip(".!?")
            snippet = " ".join(first_sentence.split()[:30])
            sentences.append(f"{snippet} [{cid}].")
        return " ".join(sentences)


class MockProvider:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class AnthropicProvider:  # pragma: no cover - needs SDK + key
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)
        resp = await client.messages.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
