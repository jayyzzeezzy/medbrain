# MedBrain — Design

A document lookup tool covering 22 sports medicine and physical therapy documents:

- 11 Mass General Brigham rehabilitation protocols
- 4 NATA position statements
- 4 JOSPT/APTA clinical practice guidelines
- 1 BJSM consensus statement
- 2 CDC HEADS UP pages

741 indexed chunks. Live on [Render](https://medbrain-zlk9.onrender.com).

The corpus was chosen because its documents have similar layouts but disagree with each other. Rotator cuff protocols differ by tear size, meniscus rehab differs by
repair versus meniscectomy, and CDC and BJSM give different concussion
return-to-play guidance. Some documents were written in two-column style.

I chose these documents deliberately, quirks included. Because the protocols overlap in subject matter but differ in specifics, a retriever that ignores document boundaries produces answers that are fluent, cited, and about the wrong procedure. That failure is the one worth designing against.

## Method: the eval set came first

The 19 questions in `evals/dataset.yaml` were written against the source
documents before the ingestion pipeline existed. I made this design decision because I wanted the eval questions to be honest.

## Key decisions and rejected alternatives

**Extraction: parse by position, not by column count.** `pdftotext` reading-order
mode detaches JOSPT evidence grades, which are set as marginal glyphs, and
attaches them to the wrong recommendation. Grade A means strong evidence and
grade D means the evidence conflicts, so this inverts how much a clinician should
trust an answer, silently and fluently, with the error already baked into the
index where no reranking or prompting can detect it. Layout mode fixes that but
interleaves NATA's two prose columns.

*Rejected:* both `pdftotext` modes, and routing between them (three code paths to
test for no gain).

*Chosen:* PyMuPDF block coordinates for every PDF, configuration tailored towards each document family. Column count was the wrong variable — NATA is also two-column,
but its grades are inline text, so position doesn't matter there. What matters is
whether a document encodes meaning in position.

**Chunk size: 3600 characters** I expected smaller chunks
to retrieve more precisely, since large ones accumulate unrelated
recommendations. Measured at k=5:

| | 3600 chars | 1400 chars |
|---|---|---|
| all-sources hit@5 | **10/11** | 9/11 |
| MRR | **0.89** | 0.86 |
| `rc-small-medium-arom-timing` | rank-3 hit | complete miss |

My hypothesis **did not hold**, and the question the smaller budget was meant to help got worse.
Larger chunks carry more of a document's vocabulary, which is what
document-level hit rate rewards.

I decided to revert to 3600 characters, with the numbers in the code comment
and the caveat that this metric cannot see within-document precision — evidence
that is necessary but not sufficient.

**Retrieval: provenance in the embedded text.** The change that actually moved
retrieval was not chunk size. Tracking one target chunk across every ingestion
change:

| Change | Rank | Distance |
|---|---|---|
| Baseline | 201/739 | 0.5041 |
| Back matter trimmed | 294 | 0.5041 |
| 3600 → 1400 chars | 294 | 0.5041 |
| Embed `title \| section \| phase \| text` | **67** | **0.3804** |
| Section labels expire after one page | **57** | 0.3804 |

The distance column is the useful one: it did not move for the first two changes,
which is what proved they were not the cause. The content hash covers the
embedded string rather than the raw text, so changing how provenance is composed
correctly invalidates the stored vectors.

**Embedding and vector store: `text-embedding-3-small` and Chroma.** Both chosen
for being unremarkable. The embedding model is deliberately *not* configurable —
changing it invalidates every stored vector, and a setting that quietly leaves
the index in two vector spaces is worse than editing one constant on purpose.
Chroma needs no separate service, which keeps `make ingest` a single command.
*Rejected:* pgvector and Qdrant, which are better at scale and would have added a
container to a corpus that fits in 45 MB.

**Prompt structure: inline `[n]` markers, resolved after the fact.** Only markers
that appear in the finished text become displayed sources. Listing everything
retrieved would present material the answer never used as though it backed the
claims. Grades render with their scale (`Grade B · NATA evidence category`)
because three scales in this corpus share letters and are not interchangeable.

**Models: OpenAI generates, Anthropic judges.** A judge from the same family as
the generator rewards its own house style, so a score can improve without the
answer becoming better grounded. Cross-provider makes agreement evidence rather
than family resemblance. Either side swaps via one environment variable.

## Failure analysis

Baseline, `gpt-4.1-mini` generating, `claude-opus-5` judging, k=5. Retrieval:
**hit@5 10/11, MRR 0.89**. Answers, scored on four dimensions:

| Passed (all four) | Behavior | Grounding | Overreach | Key facts |
|---|---|---|---|---|
| 3/18 | 12/18 | 10/18 | 15/18 | 26/56 |

Four dimensions rather than one verdict because they fail for different reasons
and are worth different amounts: a missing fact is incompleteness, an unsupported
claim is a defect, and a clinical judgement the documents do not license is the
failure this application exists to avoid.

**Retrieval and answer quality are close to uncorrelated.** Eight questions
scored `RR 1.00` and still failed judging. On retrieval numbers alone this system
looks close to solved, which is the strongest argument for building the judge model.

**Overreach is 0/2 on `bounded_answer` — the worst result and the most
consequential.** Both questions asking the system to make a clinical call got
one: `acl-squats-safety-verdict` answered "Yes" to a safety question, and
`acl-rts-clearance-12wk-85pct` adjudicated criteria in both directions. The
prompt already forbids this. Prompting alone is evidently not sufficient, and
this is the first thing I would fix.

**Grounding is 0/2 on `advice_refusal`.** Both declined correctly and then
misstated the documents while declining — one reported a return-to-play
progression as "several days to weeks" where the source says weeks to months.
Declining is not sufficient; what is said while declining still has to be true.

**Document-boundary leakage.** `acl-recon-soccer-return` fabricated soccer
coverage for a question designed to be unanswerable. The same class of error
answers a JOSPT question from an MGB protocol.

**One structural failure, deliberately not fixed.** For
`concussion-rtp-cdc-vs-amsterdam`, BJSM occupies ranks 1–22 unbroken and the
first CDC chunk sits at rank 23. BJSM has 49 chunks to CDC's 9. Top-k ranks
chunks independently and has no coverage objective, so no threshold or prompt
fixes it — a multi-document question is being served by a single-document
retriever. It needs per-document quotas. Left in place and characterised rather
than hidden. `mgb-protocols-md-clearance-aggregate` is marked `expected_to_fail`
for the related reason that it is an aggregation query, which top-k cannot answer
in principle; the report segregates it so a permanent red does not train me to
ignore reds.

## With another week

**Fix overreach first.** Two candidates: a classifier pass that routes
decision-shaped questions to a template which cannot render a verdict, or a
second generation pass that strips adjudicating language. The eval already
measures it, so both are testable in one run.

**Retrieval with a coverage objective.** Per-document quotas at retrieval time,
which should fix the concussion failure and the document-boundary leakage together.
This is the change I would make before any reranker.

**Scaling to 10,000 documents.** 22 documents produce 741 chunks, so 10,000
produce roughly 340,000. Chroma's brute-force search is 7 ms here and would not
survive that; pgvector or Qdrant with HNSW, and ingestion becomes a parallel,
resumable job rather than one serial process. The harder problem is that
per-family extraction config does not scale to documents from unknown sources: it
needs a layout classifier plus a conservative fallback, and the eval set needs
per-family canaries so a regression in one family is visible.

**Cost and latency.** Rough measurements: query embedding 384 ms, vector search
7 ms, first token ~1.5 s, complete answer ~4.2 s. Retrieval is not the budget;
it is the embedding round-trip and generation. A question-embedding cache and a
smaller k would take most of the remainder. Ingestion is already hash-diffed, so
re-running costs nothing when nothing changed. Judging is the expensive half of
the loop rather than answering: it reads the full retrieved source text for
every question, and a day of it on `claude-opus-5` came to $1.03. Embedding and
generation are billed separately on OpenAI at a fraction of that.

**Multi-tenancy.** One Chroma collection today. Per-tenant collections, since
metadata filtering leaks through nearest-neighbour scoring in ways that are hard
to prove safe, and a corpus is per-tenant anyway.

## Known shortcuts

- **Eval scores carry run-to-run variance.** `temperature=0` never guaranteed
  identical outputs, and newer Claude models reject the parameter entirely. A
  one-question delta is not a regression. Fixing this means N judge runs per
  question and a majority verdict.
- **No reranker, no hybrid retrieval.** Dense-only. The measurements above did
  not yet justify the complexity, and coverage-aware retrieval is the higher-value
  change.
- **The judge is unvalidated against human labels.** I read its verdicts and they
  were specific and checkable, but I have not measured judge-human agreement.
- **Session history is client-side only.** No persistence, no accounts.
- **Frontend is plain HTML/CSS/JS**, so the UI and the retrieval code it calls
  ship in the same commit with no second build or CORS surface. Deliberate at
  this size; I would use a framework for a real product.
