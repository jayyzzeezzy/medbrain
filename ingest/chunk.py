"""Turn extracted blocks into indexed chunks carrying their provenance.

Chunking is routed by document family because the families are shaped
differently. The MGB protocols are phase-gated, and four of them append shared
sub-programmes whose phase numbering restarts from I, so a chunk labelled only
"Phase I" is ambiguous and splitting on phase markers alone merges unrelated
ladders. The journal documents are continuous prose where a recommendation must
stay attached to the evidence grade that qualifies it.
"""

from __future__ import annotations

import hashlib
import re

from ingest.models import Block, Chunk, build_embedding_text

# Rough character budget per chunk. Characters rather than tokens because the
# ratio is stable for this corpus and the limit only needs to be safe, not exact.
#
# A/B tested against 1400/200 with the retrieval eval at k=5. The theory behind
# sizing down was that large chunks accumulate several recommendations and
# outscore short precise ones; the measurement said otherwise: 3600 scored
# hit@5 10/11 with MRR 0.89 against 9/11 and 0.86 for 1400, and the sibling
# rotator cuff question the small budget was meant to help went from a rank-3
# hit to a complete miss under it. Larger chunks carry more of a document's
# vocabulary, which is what document-level hit rate rewards. Caveat recorded
# with the decision: the metric is document-level, and within-document
# precision is not measured until answer scoring exists, so this value is held
# by evidence that is necessary but not sufficient.
MAX_CHARS = 3600
OVERLAP_CHARS = 400
MIN_CHARS = 120

# Appended sub-programmes in the MGB protocols. Each restarts its phase
# numbering, so the section name is what makes a phase label unambiguous.
SECTION_MARKERS = re.compile(
    r"^(Return to Running Program|Agility and Plyometrics? Program)", re.IGNORECASE
)

PHASE_MARKER = re.compile(r"\bPHASE\s+([IVX]+)\s*:\s*([^•]{0,70})", re.IGNORECASE)

# JOSPT sets its grade as a leading glyph; PyMuPDF merges it into the
# recommendation block, so it survives as a leading capital.
JOSPT_GRADE = re.compile(r"^([A-F])\s+(?=[A-Z][a-z]|There\b|Clinicians\b)")

# NATA writes its grade inline at the end of each numbered recommendation, but
# the vocabulary changed between statements: the 2013 ankle statement uses
# "Evidence Category", while the 2018 statements use the Strength of
# Recommendation Taxonomy. The letters coincide and the scales do not, so the
# marker that matched decides the scale recorded alongside the grade.
NATA_GRADE = re.compile(r"(Evidence Category|Strength of Recommendation \(SOR\)|SOR):\s*([A-C])")

NATA_SCALES = {
    "Evidence Category": "NATA evidence category",
    "Strength of Recommendation (SOR)": "NATA strength of recommendation (SORT)",
    "SOR": "NATA strength of recommendation (SORT)",
}

HEADING = re.compile(r"^[A-Z][A-Z \-–—:,'()/]{12,}$")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_oversized(text: str) -> list[str]:
    """Split text that exceeds the budget, preferring sentence boundaries.

    The sentence boundary is only accepted from the back half of the window.
    Searching the whole window lets a single early period pin the cut near the
    start, and with an overlap subtracted from it the loop then advances one
    character at a time. Some source text really does contain almost no sentence
    punctuation, such as the database search strategies appended to the JOSPT
    guidelines, so this is a real input rather than a theoretical one.
    """
    if len(text) <= MAX_CHARS:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + MAX_CHARS // 2, end)
            if boundary != -1:
                end = boundary + 1
            else:
                # No sentence boundary in range, so fall back to a word
                # boundary rather than cutting mid-word.
                space = text.rfind(" ", start + MAX_CHARS // 2, end)
                if space != -1:
                    end = space
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return [p for p in parts if p]


def chunk_protocol(blocks: list[Block]) -> list[tuple[str, dict[str, str | int]]]:
    """Chunk an MGB rehabilitation protocol by section, then by phase.

    Section is resolved before phase so that the appended Return to Running and
    Agility programmes, which restart at Phase I, cannot be merged into the
    rehabilitation sequence they follow.
    """
    section = "Rehabilitation Protocol"
    phase: str | None = None
    buffer: list[str] = []
    page = blocks[0].page if blocks else 1
    out: list[tuple[str, dict[str, str | int]]] = []

    def flush() -> None:
        text = " ".join(buffer).strip()
        buffer.clear()
        if len(text) < MIN_CHARS:
            return
        for part in _split_oversized(text):
            meta: dict[str, str | int] = {"section": section, "page": page}
            if phase:
                meta["phase"] = phase
            out.append((part, meta))

    for block in blocks:
        marker = SECTION_MARKERS.match(block.text)
        if marker:
            flush()
            section = marker.group(1)
            phase = None
            page = block.page

        pieces = PHASE_MARKER.split(block.text)
        if len(pieces) == 1:
            buffer.append(block.text)
            continue

        # split() yields [before, numeral, rest, after, numeral, rest, ...]
        if pieces[0].strip():
            buffer.append(pieces[0])
        for i in range(1, len(pieces) - 1, 3):
            flush()
            page = block.page
            fake = re.match(r"(?s)(.*)", pieces[i + 1])
            phase = f"Phase {pieces[i].upper()}"
            if fake:
                tail = fake.group(1).strip()
                if tail:
                    phase = f"{phase}: {tail[:60].strip(' :')}"
            remainder = pieces[i + 2] if i + 2 < len(pieces) else ""
            if remainder.strip():
                buffer.append(remainder)
    flush()
    return out


def chunk_journal(
    blocks: list[Block], grade_scale: str | None
) -> list[tuple[str, dict[str, str | int]]]:
    """Chunk a guideline or position statement, preserving evidence grades.

    A grade is attached to the chunk that carries its recommendation. The grade
    also stays in the chunk text: keeping both means the two can be compared, and
    a disagreement between them is a parser bug rather than a silent error.
    """
    out: list[tuple[str, dict[str, str | int]]] = []
    section: str | None = None
    section_page: int | None = None
    buffer: list[str] = []
    page = blocks[0].page if blocks else 1
    grade: str | None = None

    # Where the grade sits relative to its recommendation. JOSPT sets a glyph
    # before the text; NATA writes "Evidence Category: B" after it. Treating
    # both as leading markers attaches every NATA grade to the following
    # recommendation instead of the one it belongs to.
    trailing = grade_scale == "NATA evidence category"

    def flush(scale: str | None = None) -> None:
        nonlocal grade
        text = " ".join(buffer).strip()
        buffer.clear()
        recorded = scale or grade_scale
        if len(text) >= MIN_CHARS:
            for index, part in enumerate(_split_oversized(text)):
                meta: dict[str, str | int] = {"page": page}
                if section:
                    meta["section"] = section
                # Only the first part carries the grade. A continuation holds
                # the tail of the recommendation and often the start of the
                # next section, so labelling it with the grade would assert
                # something the text does not support.
                if grade and recorded and index == 0:
                    meta["grade"] = grade
                    meta["grade_scale"] = recorded
                out.append((part, meta))
        grade = None

    for block in blocks:
        text = block.text

        # A section label expires one page after its heading. Real sections in
        # these documents turn over on the page where they appear, or spill one
        # page at most; the guideline body uses mixed-case subheadings that the
        # all-caps pattern cannot see, so without an expiry a single early
        # heading labels a hundred chunks it has nothing to do with. A label
        # such as GRADES OF RECOMMENDATION STRENGTH OF EVIDENCE then leaks
        # "strength of evidence" into the embeddings of ordinary body text,
        # which steals exactly the queries that mention evidence strength. An
        # unlabelled chunk is honest; a stale label is a false promise.
        if section_page is not None and block.page > section_page + 1:
            flush()
            section = None
            section_page = None

        if HEADING.match(text):
            flush()
            section = text
            section_page = block.page
            page = block.page
            continue

        if trailing:
            # Split so each recommendation keeps the grade that closes it.
            for segment in re.split(f"({NATA_GRADE.pattern})", text):
                if not segment or not segment.strip():
                    continue
                buffer.append(segment)
                closing = NATA_GRADE.match(segment)
                if closing:
                    grade = closing.group(2)
                    scale = NATA_SCALES.get(closing.group(1), grade_scale)
                    page = block.page
                    flush(scale)
        else:
            leading = JOSPT_GRADE.match(text)
            if leading:
                flush()
                grade = leading.group(1)
                page = block.page
            buffer.append(text)

        if sum(len(b) for b in buffer) >= MAX_CHARS:
            flush()
            page = block.page
    flush()
    return out


def chunk_web(blocks: list[Block]) -> list[tuple[str, dict[str, str | int]]]:
    """Chunk an HTML page on its headings."""
    out: list[tuple[str, dict[str, str | int]]] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(buffer).strip()
        buffer.clear()
        if len(text) < MIN_CHARS:
            return
        for part in _split_oversized(text):
            meta: dict[str, str | int] = {"page": 1}
            if section:
                meta["section"] = section
            out.append((part, meta))

    for block in blocks:
        if len(block.text) < 80 and not block.text.endswith("."):
            flush()
            section = block.text
        buffer.append(block.text)
    flush()
    return out


def build_chunks(blocks: list[Block], doc: dict[str, str]) -> list[Chunk]:
    """Chunk one document and attach the provenance needed to cite it."""
    org = doc["org"]
    if doc["doc_type"] == "html":
        pieces = chunk_web(blocks)
    elif org == "Mass General Brigham":
        pieces = chunk_protocol(blocks)
    else:
        scale = {
            "JOSPT / APTA Academy of Orthopaedic Physical Therapy": (
                "JOSPT grade of recommendation"
            ),
            "NATA": "NATA evidence category",
        }.get(org)
        pieces = chunk_journal(blocks, scale)

    chunks: list[Chunk] = []
    for index, (text, meta) in enumerate(pieces):
        section = str(meta["section"]) if "section" in meta else None
        phase = str(meta["phase"]) if "phase" in meta else None
        # Hash what is embedded, not just the raw text, so that changing how
        # provenance is composed invalidates the stored vectors.
        embedded = build_embedding_text(doc["title"], section, phase, text)
        chunks.append(
            Chunk(
                id=f"{doc['id']}::{index:04d}",
                text=text,
                doc_id=doc["id"],
                title=doc["title"],
                org=org,
                url=doc["url"],
                page=int(meta.get("page", 1)),
                content_hash=_hash(f"{doc['id']}|{embedded}"),
                section=section,
                phase=phase,
                grade=str(meta["grade"]) if "grade" in meta else None,
                grade_scale=(str(meta["grade_scale"]) if "grade_scale" in meta else None),
            )
        )
    return chunks
