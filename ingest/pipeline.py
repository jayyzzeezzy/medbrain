"""Run the corpus through extraction, chunking, embedding and indexing.

Re-running is safe and cheap: chunks are compared by content hash, so only text
that actually changed is embedded again, and chunks whose source has gone are
removed rather than left behind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ingest.chunk import build_chunks
from ingest.embed import embed_texts
from ingest.extract import extract
from ingest.models import Chunk
from ingest.store import DEFAULT_PATH, ChunkStore

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "corpus" / "manifest.json"
RAW = REPO_ROOT / "corpus" / "raw"


def load_manifest() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = json.loads(MANIFEST.read_text())
    return docs


def build_corpus_chunks(*, verbose: bool = True) -> list[Chunk]:
    """Extract and chunk every document named in the manifest."""
    chunks: list[Chunk] = []
    for doc in load_manifest():
        path = RAW / f"{doc['id']}.{doc['doc_type']}"
        blocks = extract(path, doc["org"], doc["doc_type"])
        produced = build_chunks(blocks, doc)
        chunks.extend(produced)
        if verbose:
            graded = sum(1 for c in produced if c.grade)
            note = f", {graded} graded" if graded else ""
            print(f"  {doc['id']:<30} {len(produced):>4} chunks{note}")
    return chunks


def ingest(*, dry_run: bool = False, store_path: Path = DEFAULT_PATH) -> int:
    """Bring the index in line with the corpus. Returns the number embedded."""
    print("Extracting and chunking:")
    chunks = build_corpus_chunks()
    print(f"  {'total':<30} {len(chunks):>4} chunks\n")

    store = ChunkStore(store_path)
    plan = store.plan(chunks)
    print(f"Plan: {plan.describe()}")

    if plan.is_noop:
        print("Index is already current. Nothing to do.")
        return 0

    if dry_run:
        print("Dry run, stopping before any API call.")
        return 0

    if plan.to_upsert:
        print(f"Embedding {len(plan.to_upsert)} chunks:")
        vectors = embed_texts([c.embedding_text() for c in plan.to_upsert], verbose=True)
        store.apply(plan.to_upsert, vectors)

    if plan.to_delete:
        print(f"Removing {len(plan.to_delete)} stale chunks")
        store.remove(plan.to_delete)

    print(f"\nIndex now holds {store.count()} chunks.")
    return len(plan.to_upsert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the MedBrain corpus.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without embedding or writing",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_PATH,
        help=f"path to the Chroma directory (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()
    load_dotenv()
    ingest(dry_run=args.dry_run, store_path=args.store)


if __name__ == "__main__":
    main()
