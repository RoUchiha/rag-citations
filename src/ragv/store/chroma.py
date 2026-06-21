"""Optional ChromaDB-backed vector store (same VectorStore contract).

Requires the `chroma` extra; import is lazy. Skipped in the offline test suite.
"""

from __future__ import annotations

from ragv.models import Chunk


class ChromaVectorStore:  # pragma: no cover - optional heavy dep
    def __init__(self, collection: str = "ragv", persist_dir: str | None = None) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        self._col = client.get_or_create_collection(collection)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "token_count": c.token_count} for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        res = self._col.query(query_embeddings=[query_embedding], n_results=k)
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
            strict=True,
        ):
            chunk = Chunk(chunk_id=cid, doc_id=meta.get("doc_id", ""), text=doc,
                          token_count=int(meta.get("token_count", 0)))
            out.append((chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    def chunks(self) -> list[Chunk]:
        got = self._col.get()
        return [
            Chunk(chunk_id=cid, doc_id=meta.get("doc_id", ""), text=doc,
                  token_count=int(meta.get("token_count", 0)))
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True)
        ]

    def clear(self) -> None:
        self._col.delete(where={})

    def __len__(self) -> int:
        return self._col.count()
