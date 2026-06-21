"""Document loading for txt / md / pdf, with a stable doc_id."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragv.models import Document


def _doc_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}\x00{text}".encode()).hexdigest()[:12]


def load_text(text: str, source: str = "inline") -> Document:
    return Document(doc_id=_doc_id(source, text), text=text, source=source)


def load_file(path: str | Path) -> Document:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(p)
    else:  # .txt, .md, and anything else read as text
        text = p.read_text(encoding="utf-8")
    return Document(doc_id=_doc_id(str(p), text), text=text, source=str(p))


def load_dir(path: str | Path) -> list[Document]:
    docs = []
    for f in sorted(Path(path).rglob("*")):
        if f.suffix.lower() in {".txt", ".md", ".pdf"} and f.is_file():
            docs.append(load_file(f))
    return docs


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
