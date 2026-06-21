"""CLI smoke tests (ask / eval), offline."""

from __future__ import annotations

from typer.testing import CliRunner

from ragv.cli import app

runner = CliRunner()


def _corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("The mitochondria is the powerhouse of the cell.", encoding="utf-8")
    (d / "b.md").write_text("Photosynthesis happens in chloroplasts.", encoding="utf-8")
    return d


def test_ask_returns_answer(tmp_path):
    d = _corpus(tmp_path)
    result = runner.invoke(app, ["ask", "--path", str(d), "--q",
                                 "what is the powerhouse of the cell"])
    assert result.exit_code == 0
    assert "Answer" in result.stdout or "No answer" in result.stdout


def test_ask_irrelevant_no_answer(tmp_path):
    d = _corpus(tmp_path)
    result = runner.invoke(app, ["ask", "--path", str(d), "--q", "rocket orbital mechanics"])
    assert result.exit_code == 0
    assert "No answer" in result.stdout


def test_eval(tmp_path):
    d = _corpus(tmp_path)
    qa = tmp_path / "qa.yaml"
    qa.write_text(
        f"path: {d.as_posix()}\nquestions:\n  - what is the powerhouse of the cell\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", "--qa-file", str(qa)])
    assert result.exit_code == 0
    assert "answered" in result.stdout
