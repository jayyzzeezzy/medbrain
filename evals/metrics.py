"""Retrieval metrics. Pure functions so they can be tested without a network.

All metrics work on document ids rather than chunk ids: the eval set names the
documents an answer must cite, and several chunks from the right document are
interchangeable for that purpose.
"""

from __future__ import annotations


def first_relevant_rank(doc_ids: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first chunk drawn from any expected document."""
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in expected:
            return rank
    return None


def reciprocal_rank(doc_ids: list[str], expected: set[str]) -> float:
    rank = first_relevant_rank(doc_ids, expected)
    return 0.0 if rank is None else 1.0 / rank


def sources_found(doc_ids: list[str], expected: set[str], k: int) -> set[str]:
    """Which expected documents appear within the first k results."""
    return expected & set(doc_ids[:k])


def all_sources_hit(doc_ids: list[str], expected: set[str], k: int) -> bool:
    """Whether every expected document appears within the first k results.

    Deliberately all-or-nothing for multi-document questions: a synthesis
    question that retrieves only one of its two sources cannot be answered,
    and scoring that 50% would flatter a failure.
    """
    return sources_found(doc_ids, expected, k) == expected
