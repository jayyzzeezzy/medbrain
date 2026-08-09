"""Retrieve, ground, and answer.

Answering is split from prompting so the eval harness and the web API share one
implementation. Both streaming and non-streaming paths exist because the UI needs
tokens as they arrive while the judge needs a finished answer to score.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from config import answer_model
from ingest.embed import embed_texts
from ingest.store import DEFAULT_PATH, ChunkStore, Hit
from rag.llm import complete, stream
from rag.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_K = 5

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    """A retrieved document rendered for display beside an answer."""

    marker: int
    doc_id: str
    title: str
    url: str
    page: int
    # The text is carried rather than left behind in the store because both
    # consumers need it: the judge cannot check whether a citation supports a
    # claim without reading the source, and the UI shows the excerpt so a reader
    # can verify a citation without opening the PDF.
    text: str = ""
    section: str | None = None
    phase: str | None = None
    grade: str | None = None
    grade_scale: str | None = None

    @classmethod
    def from_hit(cls, marker: int, hit: Hit) -> Source:
        meta = hit.metadata
        return cls(
            marker=marker,
            text=hit.text,
            doc_id=str(meta.get("doc_id", "")),
            title=str(meta.get("title", "")),
            url=str(meta.get("url", "")),
            page=int(str(meta.get("page", 1) or 1)),
            section=str(meta["section"]) if meta.get("section") else None,
            phase=str(meta["phase"]) if meta.get("phase") else None,
            grade=str(meta["grade"]) if meta.get("grade") else None,
            grade_scale=(str(meta["grade_scale"]) if meta.get("grade_scale") else None),
        )


@dataclass
class Answer:
    """A finished answer with the sources it actually cited."""

    text: str
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Source] = field(default_factory=list)

    @property
    def cited_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(s.doc_id for s in self.sources))


def retrieve(question: str, k: int = DEFAULT_K, store_path: Path = DEFAULT_PATH) -> list[Hit]:
    """Embed a question and fetch its nearest chunks."""
    vector = embed_texts([question])[0]
    return ChunkStore(store_path).query(vector, k)


def cited_sources(text: str, hits: list[Hit]) -> list[Source]:
    """Resolve the [n] markers the answer actually used.

    Only markers present in the text become sources. Listing every retrieved
    chunk would present material the answer never relied on as though it
    supported the claims.
    """
    used = sorted({int(m) for m in CITATION.findall(text)})
    return [
        Source.from_hit(marker, hits[marker - 1]) for marker in used if 1 <= marker <= len(hits)
    ]


def answer(question: str, k: int = DEFAULT_K, store_path: Path = DEFAULT_PATH) -> Answer:
    """Produce a complete grounded answer. Used by the eval harness."""
    hits = retrieve(question, k, store_path)
    text = complete(answer_model(), SYSTEM_PROMPT, build_user_prompt(question, hits))
    return Answer(
        text=text,
        sources=cited_sources(text, hits),
        retrieved=[Source.from_hit(i, h) for i, h in enumerate(hits, start=1)],
    )


def stream_answer(
    question: str, k: int = DEFAULT_K, store_path: Path = DEFAULT_PATH
) -> Iterator[str]:
    """Yield answer tokens as they arrive."""
    hits = retrieve(question, k, store_path)
    yield from stream(answer_model(), SYSTEM_PROMPT, build_user_prompt(question, hits))


@dataclass(frozen=True)
class Event:
    """One step of a streamed answer.

    Three kinds, in order: `sources` once at the start, `token` many times, and
    `cited` once at the end. The retrieved sources are sent before the first
    token so the interface can show what the answer is about to be built from,
    and the cited subset is sent afterwards because which markers were actually
    used is not known until the text is complete.
    """

    kind: str
    text: str = ""
    sources: list[Source] = field(default_factory=list)


def stream_events(
    question: str, k: int = DEFAULT_K, store_path: Path = DEFAULT_PATH
) -> Iterator[Event]:
    """Stream an answer as typed events. Used by the web API."""
    hits = retrieve(question, k, store_path)
    yield Event("sources", sources=[Source.from_hit(i, h) for i, h in enumerate(hits, start=1)])

    parts: list[str] = []
    for token in stream(answer_model(), SYSTEM_PROMPT, build_user_prompt(question, hits)):
        parts.append(token)
        yield Event("token", text=token)

    yield Event("cited", sources=cited_sources("".join(parts), hits))
