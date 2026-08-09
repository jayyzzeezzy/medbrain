"""Persist chunks and their vectors in Chroma.

The store is responsible for idempotency. Ingestion is expected to be re-run as
the pipeline changes, and re-embedding unchanged text costs money for no gain,
so the store compares content hashes and reports exactly what needs work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata

from ingest.models import Chunk

DEFAULT_PATH = Path("var/chroma")
COLLECTION = "medbrain"


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with the provenance needed to cite or filter it."""

    text: str
    distance: float
    # Values as Chroma declares them; Mapping keeps the field covariant so the
    # library's own metadata type is accepted without a lossy cast.
    metadata: Mapping[str, object]

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))


@dataclass(frozen=True)
class Plan:
    """What a run would change, computed before anything is embedded."""

    to_upsert: list[Chunk]
    to_delete: list[str]
    unchanged: int

    @property
    def is_noop(self) -> bool:
        return not self.to_upsert and not self.to_delete

    def describe(self) -> str:
        return (
            f"{len(self.to_upsert)} to embed, "
            f"{len(self.to_delete)} to remove, "
            f"{self.unchanged} unchanged"
        )


class ChunkStore:
    """A thin wrapper over a persistent Chroma collection."""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # Vectors are supplied by the pipeline, so Chroma must not try to
        # instantiate its own embedding model.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def existing_hashes(self) -> dict[str, str]:
        """Map stored chunk id to the content hash it was indexed with."""
        stored = self._collection.get(include=["metadatas"])
        ids: list[str] = stored.get("ids") or []
        metas: list[Metadata] | None = stored.get("metadatas")
        if not metas:
            return {}
        return {
            chunk_id: str(meta.get("content_hash", ""))
            for chunk_id, meta in zip(ids, metas, strict=False)
        }

    def plan(self, chunks: list[Chunk]) -> Plan:
        """Compare desired chunks against what is stored.

        A chunk is re-embedded only when its text changed, which is what makes
        re-running the pipeline cheap and free of duplicates.
        """
        stored = self.existing_hashes()
        desired = {c.id: c for c in chunks}

        to_upsert = [c for c in chunks if stored.get(c.id) != c.content_hash]
        to_delete = [cid for cid in stored if cid not in desired]
        unchanged = len(chunks) - len(to_upsert)
        return Plan(to_upsert=to_upsert, to_delete=to_delete, unchanged=unchanged)

    def apply(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write chunks and their vectors, replacing any earlier version."""
        if not chunks:
            return
        # Chroma declares embeddings as numpy arrays but accepts plain
        # sequences; widening the element type satisfies the checker without
        # forcing a conversion the library performs anyway.
        embeddings: list[Sequence[float] | Sequence[int]] = list(vectors)
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    def remove(self, ids: list[str]) -> None:
        """Drop chunks that no longer exist in the corpus."""
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()

    def query(self, vector: list[float], k: int) -> list[Hit]:
        """Return the k nearest chunks for an already-embedded query."""
        embedding: Sequence[float] = vector
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            Hit(text=text, distance=distance, metadata=metadata)
            for text, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]
