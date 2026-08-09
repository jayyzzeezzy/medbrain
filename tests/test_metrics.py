"""Tests for the retrieval metrics, which must be trustworthy before any
ingestion change is judged by them."""

from evals.metrics import (
    all_sources_hit,
    first_relevant_rank,
    reciprocal_rank,
    sources_found,
)


def test_first_relevant_rank_is_one_based() -> None:
    assert first_relevant_rank(["a", "b"], {"a"}) == 1
    assert first_relevant_rank(["x", "b"], {"b"}) == 2
    assert first_relevant_rank(["x", "y"], {"z"}) is None


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["x"], {"b"}) == 0.0


def test_all_sources_hit_is_all_or_nothing() -> None:
    """One of two sources retrieved is a failure for a synthesis question."""
    docs = ["cdc", "cdc", "bjsm", "other", "other"]
    assert all_sources_hit(docs, {"cdc", "bjsm"}, k=5)
    assert not all_sources_hit(docs, {"cdc", "bjsm"}, k=2)
    assert not all_sources_hit(["cdc"] * 5, {"cdc", "bjsm"}, k=5)


def test_sources_found_respects_k() -> None:
    assert sources_found(["a", "b", "c"], {"c"}, k=2) == set()
    assert sources_found(["a", "b", "c"], {"c"}, k=3) == {"c"}
