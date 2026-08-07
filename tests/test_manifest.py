"""Integrity checks for the corpus manifest and the raw files it describes."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "corpus" / "manifest.json"
RAW = REPO_ROOT / "corpus" / "raw"

REQUIRED_KEYS = {"id", "url", "title", "org", "doc_type", "pub_date", "fetch"}


def load_manifest() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = json.loads(MANIFEST.read_text())
    return docs


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
