# MedBrain

Ask natural-language questions across sports medicine rehabilitation protocols
and clinical practice guidelines, and get answers grounded in the source
documents with inline citations to the exact section and page.

**Live: <https://medbrain-zlk9.onrender.com>**

MedBrain is a document lookup tool for clinical professionals. It reports what
the indexed guidelines say and does not give medical advice: it declines
personal-advice questions, and when asked to make a clinical call it reports the
documented criteria and leaves the decision to the treating clinician.

## What is in the corpus

22 public documents, 741 indexed chunks:

| Family | Count | Why it is here |
|---|---|---|
| Mass General Brigham rehabilitation protocols | 11 | Phase-gated tables; four append shared sub-programmes that restart phase numbering |
| NATA position statements | 4 | Two-column prose with evidence grades written inline |
| JOSPT / APTA clinical practice guidelines | 4 | Evidence grades set as marginal glyphs beside each recommendation |
| BJSM concussion consensus | 1 | Disagrees with the CDC guidance on return-to-play |
| CDC HEADS UP pages | 2 | HTML rather than PDF |

The documents disagree with each other in useful ways: rotator cuff protocols
differ by tear size, meniscus rehab differs by repair versus meniscectomy, and
CDC and BJSM give different return-to-play progressions. Sources and metadata are
in `corpus/manifest.json`; the files themselves are committed under `corpus/raw/`.

## Run it

Copy `.env.example` to `.env` and add your OpenAI key, then:

```bash
docker compose up --build
```

Open <http://localhost:8000>. The first start ingests the corpus, which takes
about 20 seconds; the index persists in a named volume, so later starts skip it.

<details>
<summary>Without Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate
make install
make ingest     # embeds the corpus, about 20s
make serve      # http://localhost:8000
```

</details>

## Ingestion

```bash
make ingest       # parse, chunk, embed, index
make ingest-dry   # report what would change, embed nothing
```

Idempotent by content hash. Re-running an unchanged corpus reports
`0 to embed, 0 to remove, 741 unchanged` and exits without calling the API. The
hash covers the *embedded* string — title, section, phase and text — so changing
how provenance is composed correctly invalidates the affected vectors and leaves
the rest alone.

## Evals

```bash
make eval                            # retrieval + answer scoring
make eval ARGS="--retrieval-only"    # skip the half that costs money
```

19 hand-authored questions in `evals/dataset.yaml`, written against the source
documents *before* the ingestion pipeline existed, so the measurement could
disagree with the implementation. Coverage: 6 answerable, 4 multi-document,
4 unanswerable, 2 personal-advice refusals, 2 bounded-answer, 1 clarification.

Two halves, reported separately rather than averaged, because a question can pass
one and fail the other and which one it failed is the diagnosis:

- **Retrieval** — hit rate and MRR against expected sources. Free and fast, runs
  every time.
- **Answers** — an LLM judge that reads the source text and scores four
  dimensions: behavior, grounding, overreach, and key-fact coverage. Costs an API
  call per question in each direction.

Reports are written to `evals/reports/`. Current baseline is in
[DESIGN.md](DESIGN.md), together with the failure analysis and the two failures
left unfixed on purpose.

`make eval` needs `ANTHROPIC_API_KEY` as well, since the judge runs on a
different provider from the generator. The deployed app does not.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Always | — |
| `ANTHROPIC_API_KEY` | For `make eval` | — |
| `MEDBRAIN_ANSWER_MODEL` | No | `gpt-4.1-mini` |
| `MEDBRAIN_JUDGE_MODEL` | No | `claude-opus-5` |

The provider is inferred from the model id, so either model can be pointed at the
other provider without a code change. Generation and judging default to different
providers so that a judge does not grade a model from its own family.

The embedding model is deliberately not configurable: changing it invalidates
every stored vector, and a setting that quietly leaves the index in two vector
spaces is worse than editing one constant on purpose.

Keys are read server-side only. `.env` is gitignored and never enters an image
layer.

## Development

```bash
make check      # lint, typecheck, test
make format
```

CI runs `make check` on every push: ruff, mypy, and 46 tests. The tests that
matter cover the parts that fail silently — grade metadata agreeing with the
grade left in the chunk text, chunk splitting on text with almost no sentence
punctuation, eval category minimums, and provider dispatch.

## Layout

```text
corpus/        manifest.json and the source documents
ingest/        extract, chunk, embed, index
rag/           prompts, retrieval, generation, provider dispatch
evals/         dataset.yaml, metrics, judge, runner
app/           FastAPI backend and the static frontend
tests/
```
