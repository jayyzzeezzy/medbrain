"""Score retrieval quality against the hand-authored question set.

This is the retrieval half of the eval harness. It embeds every question, runs
each against the index, and scores the results against the expected_sources
recorded in the dataset. Answer-quality scoring joins once the generation layer
exists; retrieval scoring deliberately does not wait for it, because ingestion
decisions need measuring now, against all questions at once, rather than by
hand-tuning against whichever single query is under the microscope.

Questions whose categories retrieve nothing by design (unanswerable,
advice_refusal, clarification) are not scored for hit rate. Their retrieved
context is logged instead: what the index confidently serves for a question
the app must decline is the pressure the grounding layer has to withstand.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from evals.metrics import all_sources_hit, first_relevant_rank, reciprocal_rank
from ingest.embed import embed_texts
from ingest.store import DEFAULT_PATH, ChunkStore

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "evals" / "dataset.yaml"
REPORTS = REPO_ROOT / "evals" / "reports"

SCORED_CATEGORIES = {"answerable", "multi_doc", "bounded_answer"}


@dataclass(frozen=True)
class Scored:
    question_id: str
    category: str
    expected: set[str]
    retrieved_docs: list[str]
    rank: int | None
    rr: float
    hit: bool
    expected_to_fail: bool


@dataclass(frozen=True)
class Logged:
    question_id: str
    category: str
    retrieved_docs: list[str]
    distances: list[float]


def load_dataset() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = yaml.safe_load(DATASET.read_text())
    return questions


def run(k: int, store_path: Path) -> tuple[list[Scored], list[Logged]]:
    questions = load_dataset()
    vectors = embed_texts([q["question"] for q in questions])
    store = ChunkStore(store_path)

    scored: list[Scored] = []
    logged: list[Logged] = []
    for question, vector in zip(questions, vectors, strict=True):
        hits = store.query(vector, k)
        doc_ids = [h.doc_id for h in hits]
        distances = [h.distance for h in hits]

        if question["category"] in SCORED_CATEGORIES:
            expected = set(question.get("expected_sources") or [])
            scored.append(
                Scored(
                    question_id=question["id"],
                    category=question["category"],
                    expected=expected,
                    retrieved_docs=doc_ids,
                    rank=first_relevant_rank(doc_ids, expected),
                    rr=reciprocal_rank(doc_ids, expected),
                    hit=all_sources_hit(doc_ids, expected, k),
                    expected_to_fail=bool(question.get("expected_to_fail")),
                )
            )
        else:
            logged.append(
                Logged(
                    question_id=question["id"],
                    category=question["category"],
                    retrieved_docs=doc_ids[:3],
                    distances=[round(d, 3) for d in distances[:3]],
                )
            )
    return scored, logged


def print_report(scored: list[Scored], logged: list[Logged], k: int) -> None:
    print(f"\nRETRIEVAL (k={k})")
    print(f"  {'id':42} {'cat':14} {'hit':4} {'RR':>5}")
    regular = [s for s in scored if not s.expected_to_fail]
    for s in scored:
        marker = "*" if s.expected_to_fail else " "
        hit = "yes" if s.hit else "NO"
        print(f" {marker}{s.question_id:42} {s.category:14} {hit:4} {s.rr:5.2f}")
        if not s.hit:
            missing = sorted(s.expected - set(s.retrieved_docs[:k]))
            got = ", ".join(dict.fromkeys(s.retrieved_docs[:k]))
            print(f"    missing: {', '.join(missing)}")
            print(f"    got:     {got}")
    hits = sum(1 for s in regular if s.hit)
    mrr = sum(s.rr for s in regular) / len(regular) if regular else 0.0
    print(f"\n  all-sources hit@{k}: {hits}/{len(regular)}   MRR: {mrr:.2f}")
    print("  (* = expected_to_fail, excluded from totals)")

    print("\nCONTEXT PRESSURE (categories the app must decline or redirect)")
    for entry in logged:
        docs = ", ".join(dict.fromkeys(entry.retrieved_docs)) or "(nothing)"
        print(f"  {entry.question_id:42} {entry.category:14} {docs}")


def save_report(scored: list[Scored], logged: list[Logged], k: int, count: int) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": stamp,
        "k": k,
        "index_chunks": count,
        "scored": [
            {
                "id": s.question_id,
                "category": s.category,
                "hit": s.hit,
                "rr": round(s.rr, 4),
                "rank": s.rank,
                "expected": sorted(s.expected),
                "retrieved": s.retrieved_docs,
                "expected_to_fail": s.expected_to_fail,
            }
            for s in scored
        ],
        "logged": [
            {
                "id": entry.question_id,
                "category": entry.category,
                "retrieved": entry.retrieved_docs,
                "distances": entry.distances,
            }
            for entry in logged
        ],
    }
    path = REPORTS / f"retrieval-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval against the eval set.")
    parser.add_argument("--k", type=int, default=5, help="results per query (default 5)")
    parser.add_argument("--store", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    load_dotenv()

    store = ChunkStore(args.store)
    count = store.count()
    if count == 0:
        raise SystemExit("Index is empty. Run `make ingest` first.")

    scored, logged = run(args.k, args.store)
    print_report(scored, logged, args.k)
    path = save_report(scored, logged, args.k, count)
    print(f"\nreport: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
