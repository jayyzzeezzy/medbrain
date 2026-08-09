"""Integrity checks for the corpus manifest, its raw files, and the eval set."""

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "corpus" / "manifest.json"
RAW = REPO_ROOT / "corpus" / "raw"
EVALS = REPO_ROOT / "evals" / "dataset.yaml"

REQUIRED_KEYS = {"id", "url", "title", "org", "doc_type", "pub_date", "fetch"}


def load_manifest() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = json.loads(MANIFEST.read_text())
    return docs


def load_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = yaml.safe_load(EVALS.read_text())
    return questions


def test_manifest_has_22_documents() -> None:
    assert len(load_manifest()) == 22


def test_ids_are_unique() -> None:
    ids = [doc["id"] for doc in load_manifest()]
    assert len(ids) == len(set(ids))


def test_entries_have_required_keys() -> None:
    for doc in load_manifest():
        missing = REQUIRED_KEYS - doc.keys()
        assert not missing, f"{doc.get('id', '?')} missing keys: {missing}"


def test_doc_type_and_fetch_values_are_valid() -> None:
    for doc in load_manifest():
        assert doc["doc_type"] in {"pdf", "html"}, doc["id"]
        assert doc["fetch"] in {"auto", "manual"}, doc["id"]


def test_every_manifest_entry_has_a_raw_file() -> None:
    for doc in load_manifest():
        expected = RAW / f"{doc['id']}.{doc['doc_type']}"
        assert expected.exists(), f"missing corpus file: {expected.name}"


def test_no_orphan_files_in_raw() -> None:
    expected = {f"{doc['id']}.{doc['doc_type']}" for doc in load_manifest()}
    actual = {p.name for p in RAW.iterdir() if p.is_file()}
    assert actual == expected, f"unexpected or misnamed files: {actual ^ expected}"


def test_expected_sources_resolve_to_manifest_ids() -> None:
    known = {doc["id"] for doc in load_manifest()}
    for q in load_questions():
        for src in q.get("expected_sources") or []:
            assert src in known, f"{q['id']}: unknown source '{src}'"


def test_category_minimums_are_met() -> None:
    counts: dict[str, int] = {}
    for q in load_questions():
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    assert sum(counts.values()) >= 15
    assert counts.get("unanswerable", 0) >= 3
    assert counts.get("multi_doc", 0) >= 2
    assert counts.get("advice_refusal", 0) >= 2
