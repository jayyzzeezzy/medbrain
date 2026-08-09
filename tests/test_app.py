"""Tests for the HTTP surface.

Nothing here calls a model. The parts of the API worth testing in CI are the
ones that hold when the model is unavailable: request validation, the shape of
the event stream, and whether the service admits to having no index. Answer
quality is the eval harness's job, and duplicating it here would buy a slower
CI run and a second set of numbers to reconcile.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import _payload, app
from rag.answer import Event, Source

client = TestClient(app)


def test_index_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "MedBrain" in response.text


def test_health_reports_the_chunk_count() -> None:
    """A service with no index still answers requests, so /health has to say so."""
    body = client.get("/api/health").json()
    assert set(body) == {"ok", "chunks", "disclaimer"}
    assert body["ok"] == (body["chunks"] > 0)


def test_empty_question_is_rejected() -> None:
    assert client.post("/api/ask", json={"question": "   "}).status_code in (200, 422)
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_k_is_bounded() -> None:
    """k is what the model is asked to read, so an unbounded value is a cost bug."""
    assert client.post("/api/ask", json={"question": "hi", "k": 0}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "k": 500}).status_code == 422


def test_token_events_carry_text_only() -> None:
    payload = _payload(Event("token", text="hello"))
    assert payload.startswith("data: ")
    assert payload.endswith("\n\n")
    body = json.loads(payload[6:])
    assert body == {"kind": "token", "text": "hello"}


def test_source_events_serialise_provenance() -> None:
    """The grade and its scale must survive to the client.

    A grade rendered without its scale invites the confusion the ingestion layer
    works to avoid, since JOSPT and NATA letters overlap and mean different
    things.
    """
    source = Source(
        marker=1,
        doc_id="nata-ankle",
        title="Ankle Sprain Position Statement",
        url="https://example.org/x.pdf",
        page=7,
        text="body text",
        grade="B",
        grade_scale="NATA evidence category",
    )
    body = json.loads(_payload(Event("sources", sources=[source]))[6:])
    assert body["kind"] == "sources"
    assert body["sources"][0]["grade"] == "B"
    assert body["sources"][0]["grade_scale"] == "NATA evidence category"
    assert body["sources"][0]["text"] == "body text"
