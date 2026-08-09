"""HTTP interface for MedBrain.

Answers stream over server-sent events rather than arriving whole. Retrieval and
generation together take several seconds, and a request that returns nothing for
that long reads as broken. Streaming also lets the interface show which sources
were retrieved before the first token, so a reader can see what the answer is
being built from while it is being written.

The store is opened once at import. Chroma reads from disk on every query, and
re-opening the client per request adds latency for nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingest.store import DEFAULT_PATH, ChunkStore
from rag.answer import DEFAULT_K, Event, stream_events
from rag.llm import MissingAPIKey, Refused, UnknownModel
from rag.prompts import DISCLAIMER

load_dotenv()

STATIC = Path(__file__).parent / "static"
STORE_PATH = Path(os.environ.get("MEDBRAIN_STORE", DEFAULT_PATH))

app = FastAPI(title="MedBrain", docs_url=None, redoc_url=None)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Exposed so the retrieval depth can be varied from the interface without a
    # redeploy. Bounded because k is what the model is asked to read, and an
    # unbounded value is a way to run up a bill through the front door.
    k: int = Field(default=DEFAULT_K, ge=1, le=20)


def _payload(event: Event) -> str:
    body: dict[str, object] = {"kind": event.kind}
    if event.kind == "token":
        body["text"] = event.text
    else:
        body["sources"] = [asdict(s) for s in event.sources]
    return f"data: {json.dumps(body)}\n\n"


def _events(question: str, k: int) -> Iterator[str]:
    try:
        for event in stream_events(question, k, STORE_PATH):
            yield _payload(event)
    except Refused:
        yield _error("The model declined to respond to this question.")
    except (MissingAPIKey, UnknownModel) as exc:
        # Configuration faults, not user faults. Surfaced rather than swallowed
        # because a silent empty answer is indistinguishable from a real one.
        yield _error(f"MedBrain is misconfigured: {exc}")
    except Exception:  # noqa: BLE001
        # The stream has already begun, so the status code is committed and the
        # only way left to report a failure is in the stream itself.
        yield _error("Something went wrong while answering. Please try again.")
    yield 'data: {"kind": "done"}\n\n'


def _error(message: str) -> str:
    return f"data: {json.dumps({'kind': 'error', 'text': message})}\n\n"


@app.post("/api/ask")
def ask(request: Ask) -> StreamingResponse:
    """Stream a grounded answer as server-sent events."""
    return StreamingResponse(
        _events(request.question, request.k),
        media_type="text/event-stream",
        # Without this, a proxy that buffers will hold the whole response and
        # defeat streaming end to end while looking correct in development.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    """Report whether the index is actually present.

    A container that starts with an empty index answers every question with a
    polite refusal, which looks like working software. The chunk count is the
    cheapest way to tell that apart from outside.
    """
    chunks = ChunkStore(STORE_PATH).count()
    return {"ok": chunks > 0, "chunks": chunks, "disclaimer": DISCLAIMER}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
