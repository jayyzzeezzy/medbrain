"""Score the system against the hand-authored question set.

Two halves that measure different things. Retrieval scoring asks whether the
right document reached the model, and runs on every invocation because it is
fast and free. Answer scoring asks what the model did with it, and costs an API
call per question in each direction, so --retrieval-only exists for the tight
loop while the full run stays the default.

The halves are reported separately rather than averaged, because a question can
pass one and fail the other, and which one it failed is the whole diagnosis.
One question in this set retrieves its source at rank 1 and then misattributes
an evidence grade; a combined score would have shown that as a partial credit
somewhere in the middle and pointed at nothing.

Questions whose categories retrieve nothing by design (unanswerable,
advice_refusal, clarification) are not scored for hit rate. Their retrieved
context is logged instead: what the index confidently serves for a question
the app must decline is the pressure the grounding layer has to withstand.
Answer scoring, by contrast, covers every question, since declining well is a
behaviour worth measuring rather than an absence of one.
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

from config import answer_model, judge_model
from evals.judge import JudgeError, Verdict, judge
from evals.metrics import all_sources_hit, first_relevant_rank, reciprocal_rank
from ingest.embed import embed_texts
from ingest.store import DEFAULT_PATH, ChunkStore
from rag.answer import answer

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


def run_answers(k: int, store_path: Path) -> list[Verdict]:
    """Generate an answer for every question and score it.

    Answering goes through the same entry point the web API uses rather than a
    shortcut that reuses the vectors embedded above. An eval that exercises a
    different retrieval path from production measures something production does
    not do, and the duplicated embedding cost is a rounding error against the
    generation calls.
    """
    verdicts: list[Verdict] = []
    questions = load_dataset()
    for index, question in enumerate(questions, start=1):
        print(f"  [{index}/{len(questions)}] {question['id']}", flush=True)
        try:
            verdicts.append(judge(question, answer(question["question"], k, store_path)))
        except JudgeError as exc:
            # An unreadable verdict is a harness fault, not an answer fault.
            # Scoring it as a failure would blame the system under test.
            print(f"      skipped: {exc}")
    return verdicts


def print_answer_report(verdicts: list[Verdict]) -> None:
    print(f"\nANSWERS (generator {answer_model()}, judge {judge_model()})")
    print(f"  {'id':42} {'cat':14} {'pass':5} {'beh':4} {'grnd':5} {'ovr':4} {'facts':>6}")
    for v in verdicts:
        marker = "*" if v.expected_to_fail else " "
        print(
            f" {marker}{v.question_id:42} {v.category:14} "
            f"{'yes' if v.passed else 'NO':5} "
            f"{'ok' if v.behavior else 'NO':4} "
            f"{'ok' if v.grounding else 'NO':5} "
            f"{'ok' if v.overreach else 'NO':4} "
            f"{v.facts_present}/{v.facts_total:<4}"
        )
        if not v.passed:
            print(f"    {v.summary}")
            for claim in v.unsupported:
                print(f"    unsupported: {claim}")

    regular = [v for v in verdicts if not v.expected_to_fail]
    if not regular:
        return
    total = len(regular)
    facts_total = sum(v.facts_total for v in regular)
    facts_present = sum(v.facts_present for v in regular)
    print(f"\n  passed:    {sum(1 for v in regular if v.passed)}/{total}")
    print(f"  behavior:  {sum(1 for v in regular if v.behavior)}/{total}")
    print(f"  grounding: {sum(1 for v in regular if v.grounding)}/{total}")
    print(f"  overreach: {sum(1 for v in regular if v.overreach)}/{total}")
    if facts_total:
        print(f"  key facts: {facts_present}/{facts_total}")
    print("  (* = expected_to_fail, excluded from totals)")


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


def save_report(
    scored: list[Scored],
    logged: list[Logged],
    verdicts: list[Verdict],
    k: int,
    count: int,
) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "timestamp": stamp,
        "k": k,
        "index_chunks": count,
        # Recorded because the scores are not comparable across models, and a
        # report that does not say which models produced it invites exactly that
        # comparison later.
        "answer_model": answer_model(),
        "judge_model": judge_model() if verdicts else None,
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
        "answers": [
            {
                "id": v.question_id,
                "category": v.category,
                "passed": v.passed,
                "behavior": v.behavior,
                "grounding": v.grounding,
                "overreach": v.overreach,
                "facts_present": v.facts_present,
                "facts_total": v.facts_total,
                "unsupported": v.unsupported,
                "notes": v.notes,
                "summary": v.summary,
                "expected_to_fail": v.expected_to_fail,
            }
            for v in verdicts
        ],
    }
    path = REPORTS / f"eval-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MedBrain against the eval set.")
    parser.add_argument("--k", type=int, default=5, help="results per query (default 5)")
    parser.add_argument("--store", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip answer generation and judging, which is the part that costs money",
    )
    args = parser.parse_args()
    load_dotenv()

    store = ChunkStore(args.store)
    count = store.count()
    if count == 0:
        raise SystemExit("Index is empty. Run `make ingest` first.")

    scored, logged = run(args.k, args.store)
    print_report(scored, logged, args.k)

    verdicts: list[Verdict] = []
    if not args.retrieval_only:
        print("\nGENERATING AND JUDGING")
        verdicts = run_answers(args.k, args.store)
        print_answer_report(verdicts)

    path = save_report(scored, logged, verdicts, args.k, count)
    print(f"\nreport: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
