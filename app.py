"""Gradio live demo for ragv.

Ask a grounded question over a small corpus and see: the hybrid-retrieved spans
(dense / BM25 / fused scores), the generated answer, and **every citation
verified** — unsupported ones are stripped. Runs offline (hashing embedder +
extractive generator). Deployable to HF Spaces.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from ragv.config import Config  # noqa: E402
from ragv.generate.provider import HeuristicProvider  # noqa: E402
from ragv.ingest.embedder import HashingEmbedder  # noqa: E402
from ragv.ingest.loader import load_text  # noqa: E402
from ragv.pipeline import RagPipeline  # noqa: E402

CORPUS = [
    "The mitochondria is the powerhouse of the cell. It generates ATP through respiration.",
    "HTTPS encrypts traffic using TLS, protecting data from eavesdroppers in transit.",
    "The HTTP status code 404 means the requested resource was not found on the server.",
    "Python's GIL allows only one thread to execute Python bytecode at a time.",
    "BM25 is a ranking function search engines use to estimate a document's relevance.",
]

PIPE = RagPipeline(HashingEmbedder(), HeuristicProvider(), Config(top_k=4))
PIPE.ingest([load_text(t, source=f"doc{i}") for i, t in enumerate(CORPUS)])


def ask(query: str):
    if not query.strip():
        return "Enter a question.", pd.DataFrame()
    # Show retrieval first (for the spans table), then the verified answer.
    from ragv.retrieve.dense import DenseRetriever
    from ragv.retrieve.retriever import HybridRetriever

    hybrid = HybridRetriever(DenseRetriever(PIPE.store, PIPE.embedder), PIPE._bm25, PIPE.config)
    spans = hybrid.retrieve(query)
    spans_df = pd.DataFrame(
        [{"chunk_id": s.chunk_id, "dense": round(s.dense_score, 3),
          "bm25": round(s.sparse_score, 2), "fused": round(s.fused_score, 4),
          "text": s.text[:70]} for s in spans]
    )

    ans = asyncio.run(PIPE.ask(query))
    if ans.no_answer:
        return f"### 🟡 {ans.text}\n_(no relevant evidence retrieved)_", spans_df
    cites = "\n".join(
        f"- `{', '.join(c.cited_chunk_ids)}` support={c.support_score} ✅"
        for c in ans.citations
    )
    md = (
        f"### Answer\n{ans.text}\n\n"
        f"**Verified citations:**\n{cites}\n\n"
        f"**Confidence:** {ans.confidence}"
    )
    if ans.unsupported_claims:
        md += f"\n\n_Stripped {len(ans.unsupported_claims)} unsupported claim(s)._"
    return md, spans_df


with gr.Blocks(title="Hybrid RAG with Verified Citations", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔎 Hybrid RAG with Verified Citations\n"
        "Dense + BM25 retrieval fused with **Reciprocal Rank Fusion**, grounded answers "
        "with inline `[chunk_id]` citations, and **every citation verified** against the "
        "retrieved spans — hallucinated or unsupported citations are stripped.\n\n"
        "_Offline demo: hashing embedder + extractive generator over a small corpus._"
    )
    with gr.Row():
        q = gr.Textbox(label="Ask", placeholder="what does HTTP 404 mean?", scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(
        ["what does HTTP 404 mean?", "what is the powerhouse of the cell?",
         "how does HTTPS protect data?", "what is BM25?", "how do volcanoes erupt?"],
        inputs=q,
    )
    out = gr.Markdown()
    spans = gr.Dataframe(label="Hybrid-retrieved spans (dense vs BM25 vs fused)")
    btn.click(ask, q, [out, spans])
    q.submit(ask, q, [out, spans])


if __name__ == "__main__":
    demo.launch()
