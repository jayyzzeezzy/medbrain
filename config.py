"""Model selection for the generation and evaluation layers.

Values are read when asked for rather than at import, because the CLI entry
points call load_dotenv() after their imports have already run. Reading at
import would silently ignore .env.

The embedding model is deliberately not configurable here. Changing it
invalidates every stored vector.
"""

from __future__ import annotations

import os

# The generator and the judge default to different providers on purpose.
# This design choice is trying to mitigate any judgement biases.
DEFAULT_ANSWER_MODEL = "gpt-4.1-mini"
DEFAULT_JUDGE_MODEL = "claude-opus-5"


def answer_model() -> str:
    """Model that generates grounded answers. Set MEDBRAIN_ANSWER_MODEL to change."""
    return os.environ.get("MEDBRAIN_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)


def judge_model() -> str:
    """Model that scores answers in the eval harness.

    Defaults to a stronger model than the generator. A judge no better than the
    system it grades tends to agree with it, which makes the eval look healthier
    than it is. Set MEDBRAIN_JUDGE_MODEL to change.
    """
    return os.environ.get("MEDBRAIN_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
