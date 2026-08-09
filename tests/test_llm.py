"""Tests for provider dispatch.

Provider inference is the only part of the LLM layer that can be tested without
a key, and it is also the part that fails silently: an unrecognised model id
routed to the wrong SDK produces an authentication error that reads like a
missing key rather than a bad model name.
"""

from __future__ import annotations

import pytest

import config
from rag.llm import UnknownModel, provider_for


@pytest.mark.parametrize(
    "model",
    ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "o3-mini", "chatgpt-4o-latest"],
)
def test_openai_ids_route_to_openai(model: str) -> None:
    assert provider_for(model) == "openai"


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5"],
)
def test_anthropic_ids_route_to_anthropic(model: str) -> None:
    assert provider_for(model) == "anthropic"


@pytest.mark.parametrize("model", ["", "llama-3", "gemini-2.5-pro", "mistral-large"])
def test_unrecognised_ids_raise_rather_than_guess(model: str) -> None:
    with pytest.raises(UnknownModel):
        provider_for(model)


def test_defaults_are_resolvable() -> None:
    """A default nobody can route is a broken install, not a runtime surprise."""
    assert provider_for(config.DEFAULT_ANSWER_MODEL) == "openai"
    assert provider_for(config.DEFAULT_JUDGE_MODEL) == "anthropic"


def test_generator_and_judge_default_to_different_providers() -> None:
    """The separation is the point; a change that collapses it should fail here."""
    assert provider_for(config.DEFAULT_ANSWER_MODEL) != provider_for(config.DEFAULT_JUDGE_MODEL)


def test_environment_overrides_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDBRAIN_ANSWER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("MEDBRAIN_JUDGE_MODEL", "gpt-4.1")
    assert provider_for(config.answer_model()) == "anthropic"
    assert provider_for(config.judge_model()) == "openai"
