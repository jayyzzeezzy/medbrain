"""Tests for the ingestion pipeline.

Extraction and chunking are exercised against the real corpus because the
behaviour worth testing is how this specific corpus is shaped. Embedding is not
exercised: it needs an API key that CI does not have, and the parts that can go
wrong without a network are the parts these tests cover.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ingest.chunk import MAX_CHARS, OVERLAP_CHARS, _split_oversized, build_chunks
from ingest.embed import DIMENSIONS
from ingest.extract import extract
from ingest.models import Chunk
from ingest.store import ChunkStore

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "corpus" / "raw"
MANIFEST = REPO_ROOT / "corpus" / "manifest.json"


def doc(doc_id: str) -> dict[str, str]:
    entries: list[dict[str, str]] = json.loads(MANIFEST.read_text())
    return next(d for d in entries if d["id"] == doc_id)


def chunks_for(doc_id: str) -> list[Chunk]:
    meta = doc(doc_id)
    path = RAW / f"{doc_id}.{meta['doc_type']}"
    return build_chunks(extract(path, meta["org"], meta["doc_type"]), meta)


def test_phase_labels_are_disambiguated_by_section() -> None:
    """Three unrelated ladders in one document each restart at Phase I.

    The Achilles protocol has seven rehabilitation phases, then appends a
    Return to Running Program and an agility programme that both number their
    own phases from I. Without the section, "Phase I" identifies nothing.
    """
    phase_ones = [c for c in chunks_for("mgb-achilles-repair") if c.phase and "Phase I:" in c.phase]
    sections = {c.section for c in phase_ones}
    assert len(sections) == 3, f"expected three distinct Phase I sections, got {sections}"
    assert "Rehabilitation Protocol" in sections
    assert any(s and "Running" in s for s in sections)
    assert any(s and "Agility" in s for s in sections)


def test_jospt_grade_attaches_to_its_own_recommendation() -> None:
    """A leading grade glyph must stay with the recommendation it qualifies."""
    match = [
        c
        for c in chunks_for("jospt-ankle-cpg")
        if "structured therapeutic exercise component" in c.text
    ]
    assert match, "therapeutic exercise recommendation not found"
    assert match[0].grade == "A"
    assert match[0].grade_scale == "JOSPT grade of recommendation"


def test_nata_grade_attaches_to_the_recommendation_it_closes() -> None:
    """NATA writes its grade after the recommendation, not before it.

    Treating it as a leading marker shifts every grade onto the following
    recommendation, which is the same misattribution that makes naive PDF
    extraction unsafe for this corpus.
    """
    match = [
        c for c in chunks_for("nata-ankle") if "Balance training should be performed" in c.text
    ]
    assert match, "balance training recommendation not found"
    assert match[0].grade == "A"
    assert match[0].grade_scale == "NATA evidence category"


def test_nata_scales_are_recorded_separately() -> None:
    """The 2013 and 2018 NATA statements use different scales with the same letters."""
    scales = {c.grade_scale for c in chunks_for("nata-acl") if c.grade}
    assert scales == {"NATA strength of recommendation (SORT)"}


def test_grade_metadata_matches_the_grade_left_in_the_text() -> None:
    """Grades are stored twice on purpose so the two can be compared."""
    for chunk in chunks_for("jospt-ankle-cpg"):
        if chunk.grade:
            assert chunk.text.lstrip().startswith(
                chunk.grade
            ), f"{chunk.id}: metadata grade {chunk.grade!r} is absent from the text"


def test_split_always_makes_progress() -> None:
    """Text with almost no sentence punctuation must not cause a crawl.

    The JOSPT guidelines append database search strategies that contain a single
    period across thousands of characters. Searching the whole window for a
    boundary pinned the cut near the start, and subtracting the overlap from it
    advanced one character per iteration, producing hundreds of near-duplicates.
    """
    text = "word " * 4000 + ". " + "word " * 4000
    parts = _split_oversized(text)

    # Each step must advance by at least the budget minus the overlap, so the
    # part count is bounded by the input size rather than by how much sentence
    # punctuation the text happens to contain. Asserting against the parameters
    # rather than a fixed number keeps this honest if the budget is retuned.
    ceiling = len(text) / (MAX_CHARS - OVERLAP_CHARS) * 1.5
    assert len(parts) < ceiling, f"expected under {ceiling:.0f} parts, got {len(parts)}"
    assert sum(len(p) for p in parts) < len(text) * 1.5, "excessive duplication"
    for part in parts:
        assert len(part) <= MAX_CHARS + OVERLAP_CHARS


def test_every_chunk_carries_citable_provenance() -> None:
    for chunk in chunks_for("mgb-acl-reconstruction"):
        assert chunk.doc_id and chunk.title and chunk.url
        assert chunk.page >= 1
        assert chunk.content_hash


def test_store_is_idempotent(tmp_path: Path) -> None:
    """Re-running with unchanged content must embed nothing and duplicate nothing."""
    chunks = chunks_for("mgb-meniscectomy")
    store = ChunkStore(tmp_path / "chroma")

    first = store.plan(chunks)
    assert len(first.to_upsert) == len(chunks)
    assert first.unchanged == 0

    vectors = [[0.0] * DIMENSIONS for _ in first.to_upsert]
    store.apply(first.to_upsert, vectors)
    assert store.count() == len(chunks)

    second = store.plan(chunks)
    assert second.is_noop, second.describe()
    assert second.unchanged == len(chunks)
    assert store.count() == len(chunks)


def test_store_removes_chunks_whose_source_is_gone(tmp_path: Path) -> None:
    chunks = chunks_for("mgb-meniscectomy")
    store = ChunkStore(tmp_path / "chroma")
    plan = store.plan(chunks)
    store.apply(plan.to_upsert, [[0.0] * DIMENSIONS for _ in plan.to_upsert])

    shrunk = chunks[:2]
    revised = store.plan(shrunk)
    assert len(revised.to_delete) == len(chunks) - 2
    store.remove(revised.to_delete)
    assert store.count() == 2


@pytest.mark.parametrize("doc_id", ["mgb-tka", "nata-pf", "cdc-return-to-sports"])
def test_documents_produce_chunks(doc_id: str) -> None:
    produced: list[Any] = chunks_for(doc_id)
    assert produced, f"{doc_id} produced no chunks"
