"""Turn chunk text into vectors using the OpenAI embeddings API.

Embedding is the only step in ingestion that costs money and requires network
access, which is why it is isolated here. Extraction and chunking stay pure so
they can be tested in CI, where no API key exists.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

from openai import OpenAI

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536

# The API accepts far larger batches, but keeping them modest bounds the cost of
# a failed request and keeps progress reporting meaningful on a corpus this size.
BATCH_SIZE = 100


class MissingAPIKey(RuntimeError):
    """Raised when embedding is attempted without a configured key."""


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKey(
            "OPENAI_API_KEY is not set. Add it to .env, which is gitignored, "
            "or export it in the shell before running ingestion."
        )
    # The SDK retries transient failures and rate limits on its own.
    return OpenAI(api_key=key, max_retries=5)


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_texts(texts: Sequence[str], *, verbose: bool = False) -> list[list[float]]:
    """Embed texts in order, returning one vector per input."""
    if not texts:
        return []
    client = _client()
    vectors: list[list[float]] = []
    for batch in _batched(texts, BATCH_SIZE):
        response = client.embeddings.create(model=MODEL, input=list(batch))
        # The API preserves input order, but sorting by index makes that
        # assumption explicit rather than load-bearing and invisible.
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)
        if verbose:
            print(f"    embedded {len(vectors)}/{len(texts)}")
    return vectors
