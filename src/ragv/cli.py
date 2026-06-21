"""Typer CLI: ingest / ask / eval (offline by default)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
import yaml
from loguru import logger
from rich.console import Console

from ragv.config import Config
from ragv.generate.provider import HeuristicProvider
from ragv.ingest.embedder import HashingEmbedder
from ragv.ingest.loader import load_dir, load_file
from ragv.pipeline import RagPipeline

app = typer.Typer(add_completion=False, help="Hybrid RAG with verified citations.")
console = Console()


@app.callback()
def _configure() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("RAGV_LOG_LEVEL", "WARNING"))


def _docs(path: str):
    p = Path(path)
    return load_dir(p) if p.is_dir() else [load_file(p)]


def _pipeline(policy: str) -> RagPipeline:
    return RagPipeline(HashingEmbedder(), HeuristicProvider(),
                       Config(unsupported_policy=policy))


@app.command()
def ingest(path: str = typer.Option(..., "--path")) -> None:
    """Ingest a file or directory and report chunk counts."""
    pipe = _pipeline("strip")
    n = pipe.ingest(_docs(path))
    console.print(f"ingested [bold]{n}[/bold] chunks from {path}")


@app.command()
def ask(
    path: str = typer.Option(..., "--path", help="docs file or directory"),
    q: str = typer.Option(..., "--q"),
    policy: str = typer.Option("strip", "--policy", help="strip|flag"),
    k: int = typer.Option(5, "--k"),
) -> None:
    """Ingest, retrieve (hybrid), generate, and verify citations."""
    pipe = _pipeline(policy)
    pipe.config.top_k = k
    pipe.ingest(_docs(path))
    answer = asyncio.run(pipe.ask(q))
    if answer.no_answer:
        console.print("[yellow]No answer:[/yellow] " + answer.text)
        return
    console.print(f"[bold]Answer:[/bold] {answer.text}")
    console.print(f"[bold]Confidence:[/bold] {answer.confidence}")
    console.print(f"[green]Verified citations:[/green] {len(answer.citations)}")
    if answer.unsupported_claims:
        console.print(f"[red]Unsupported (handled by '{policy}'):[/red] "
                      f"{len(answer.unsupported_claims)}")


@app.command()
def eval(
    qa_file: str = typer.Option(..., "--qa-file", help="YAML: {path, questions: [...]}"),
) -> None:
    """Run a question set; report how many produced supported answers."""
    spec = yaml.safe_load(Path(qa_file).read_text(encoding="utf-8"))
    pipe = _pipeline("strip")
    pipe.ingest(_docs(spec["path"]))
    answered = 0
    for question in spec["questions"]:
        ans = asyncio.run(pipe.ask(question))
        answered += 0 if ans.no_answer else 1
    console.print(f"answered {answered}/{len(spec['questions'])} with supported citations")


if __name__ == "__main__":
    app()
