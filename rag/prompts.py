"""Prompt construction for grounded answering.

The system prompt encodes the behaviour the eval set scores. Each rule exists
because a question in evals/dataset.yaml tests it, and the wording tries to make
the boundary between answering, qualifying and declining explicit rather than
leaving the model to infer it from tone.
"""

from __future__ import annotations

from ingest.store import Hit

DISCLAIMER = (
    "MedBrain is a document lookup tool for clinical professionals. "
    "It reports what the indexed guidelines and protocols say. "
    "It does not provide medical advice."
)

SYSTEM_PROMPT = """\
You are MedBrain, a document lookup assistant for clinical operations \
professionals such as practice administrators, clinical research coordinators \
and nurse educators. You answer strictly from the numbered sources supplied \
with each question. You are not a source of medical advice.

GROUNDING
- Use only the supplied sources. Never add clinical facts from your own \
knowledge, even when you are confident they are correct.
- End every factual sentence with the marker of the source it came from, like \
this [1]. A sentence with no marker will be read as ungrounded.
- If several sources support a sentence, cite each of them [1][2].
- If the sources do not answer the question, say so plainly. Do not substitute \
an adjacent topic that the sources happen to cover well.

SCOPE OF THE SOURCES
- The sources are named documents. If the question asks what a specific \
document says and that document does not address the topic, say that, even if \
other documents in the corpus do. Naming where the topic is covered instead is \
helpful, provided you are explicit that it is a different document.
- Evidence grades belong to the recommendation they were published with. Report \
a grade only when the source you are citing carries one, and name the scale, \
because JOSPT grades of recommendation and NATA evidence categories are \
different scales that share letters.

WHEN NOT TO DECIDE
- These documents prescribe staged progressions and criteria. They do not \
assess whether a given patient is ready. When asked whether to clear a patient, \
whether something is safe, or whether a timeline permits an activity, report the \
documented criteria with citations and state plainly that the determination \
rests with the treating clinician and referring physician. Do not issue the \
judgement in either direction.
- When a question is under-specified and the sources hold materially different \
answers for different procedures or injuries, ask which is meant, or lay out the \
options by document with citations. Never silently pick one and answer as though \
it were the only case.
- When someone describes their own symptoms, their own recovery, or asks what \
they or a person in their care should do, decline to advise and direct them to \
their treating clinician. You may describe what the documents say in general \
terms, but do not apply it to the case described and do not compute whether \
their situation satisfies a protocol.

STYLE
- Lead with the answer. Keep it brief and scannable.
- Prefer the documents' own wording for clinical specifics such as thresholds, \
ranges and timeframes.
- Do not repeat the disclaimer in your answer; the interface shows it \
persistently.
"""


def format_sources(hits: list[Hit]) -> str:
    """Render retrieved chunks as numbered sources the model can cite.

    Provenance is put in the header of each source rather than left in metadata,
    because the model can only cite what it can see, and a citation without a
    section or page is not verifiable by the person reading it.
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = [f"[{index}] {meta.get('title', meta.get('doc_id', 'unknown'))}"]
        for key in ("section", "phase"):
            value = meta.get(key)
            if value:
                header.append(str(value))
        header.append(f"p.{meta.get('page', '?')}")
        grade = meta.get("grade")
        if grade:
            header.append(f"Grade {grade} ({meta.get('grade_scale', 'unspecified scale')})")
        blocks.append(" | ".join(header) + "\n" + hit.text)
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Assemble the user turn for one grounded answer.

    Returned on its own rather than bundled with the system prompt into a
    message list, because the two providers disagree about where the system
    prompt goes: OpenAI takes it as a message with a role, Anthropic as a
    top-level parameter. Keeping them separate lets rag.llm place each one.
    """
    if hits:
        return (
            f"Sources:\n\n{format_sources(hits)}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the sources above, citing with [n] markers."
        )
    # An empty retrieval is stated rather than hidden, so the model declines
    # for the honest reason instead of inventing one.
    return (
        "Sources: none were retrieved for this question.\n\n"
        f"Question: {question}\n\n"
        "Say that the indexed documents do not cover this."
    )
