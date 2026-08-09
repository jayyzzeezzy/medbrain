"""Data structures shared across the ingestion pipeline."""

from dataclasses import dataclass, field


def build_embedding_text(title: str, section: str | None, phase: str | None, text: str) -> str:
    """Compose the provenance-prefixed text used for embedding and hashing.

    Kept as a free function so the pipeline and the content hash cannot drift
    apart: whatever is embedded is exactly what the hash covers, so changing the
    prefix correctly invalidates the stored vectors.
    """
    parts = [title]
    if section:
        parts.append(section)
    if phase:
        parts.append(phase)
    parts.append(text)
    return " | ".join(parts)


@dataclass(frozen=True)
class Block:
    """A positioned run of text extracted from one page of a source document.

    Blocks preserve position because position carries meaning in this corpus:
    the JOSPT guidelines encode evidence grades as glyphs set beside their
    recommendation, and reading a two-column page in raw stream order
    interleaves unrelated text.
    """

    text: str
    page: int
    column: int
    x0: float
    y0: float


@dataclass(frozen=True)
class Chunk:
    """An indexed unit of text with the provenance needed to cite it.

    Provenance is not decoration. Several MGB protocols append shared
    sub-programmes whose phase numbering restarts, so "Phase I" is ambiguous
    without its section, and the milestone criteria in the operative and
    non-operative ACL protocols are near-identical without their document.
    """

    id: str
    text: str
    doc_id: str
    title: str
    org: str
    url: str
    page: int
    content_hash: str
    section: str | None = None
    phase: str | None = None
    grade: str | None = None
    grade_scale: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """The text actually sent to the embedding model.

        A chunk read in isolation often omits the words a user searches with.
        The ankle guideline's therapeutic exercise recommendation never says
        "ankle", "sprain" or "2021"; those live in the title and the section
        heading. Prepending them lets a chunk match a question phrased around
        the document it came from, while the stored document text stays exactly
        what a citation will show.
        """
        return build_embedding_text(self.title, self.section, self.phase, self.text)

    def metadata(self) -> dict[str, str | int]:
        """Flatten to a Chroma-compatible metadata mapping.

        Chroma rejects None values, so absent fields are omitted rather than
        stored as nulls.
        """
        meta: dict[str, str | int] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "org": self.org,
            "url": self.url,
            "page": self.page,
            "content_hash": self.content_hash,
        }
        for key, value in (
            ("section", self.section),
            ("phase", self.phase),
            ("grade", self.grade),
            ("grade_scale", self.grade_scale),
        ):
            if value is not None:
                meta[key] = value
        meta.update(self.extra)
        return meta
