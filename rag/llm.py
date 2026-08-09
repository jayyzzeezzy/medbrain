"""One text-completion interface over two providers.

The generator and the judge are deliberately allowed to run on different
providers. 

The provider is inferred from the model id rather than configured separately,
because a provider setting that disagrees with the model name is a failure mode
with no upside.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Literal, cast

Provider = Literal["openai", "anthropic"]

# Generous enough that a grounded answer is never truncated, and, on Anthropic
# models where thinking is on by default, large enough that reasoning tokens do
# not eat the budget before the answer starts. Answers here are short; this is
# headroom, not a target.
MAX_TOKENS = 4096

# Only sent to OpenAI. Sampling parameters were removed on Claude Opus 5,
# Fable 5, Sonnet 5 and Opus 4.8/4.7, where sending temperature at all returns a
# 400. Note that temperature=0 never guaranteed identical outputs on any
# provider, so this buys less reproducibility than it appears to; see the
# variance note in DESIGN.md.
OPENAI_TEMPERATURE = 0.0


class UnknownModel(ValueError):
    """Raised when a configured model id belongs to no known provider."""


class MissingAPIKey(RuntimeError):
    """Raised when the key for the selected provider is not configured."""


class Refused(RuntimeError):
    """Raised when the provider declined to generate at all.

    Distinct from the model declining to answer in prose. A refusal produces no
    usable text, so the caller has to handle it rather than score it.
    """


def provider_for(model: str) -> Provider:
    """Infer which SDK to call from the model id."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise UnknownModel(
        f"Cannot tell which provider serves {model!r}. Expected an id beginning "
        "with 'claude' or 'gpt-'."
    )


def _key(name: str, provider: Provider) -> str:
    key = os.environ.get(name)
    if not key:
        raise MissingAPIKey(
            f"{name} is not set, but the selected model runs on {provider}. "
            "Add it to .env, which is gitignored, or export it in the shell."
        )
    return key


def complete(model: str, system: str, user: str) -> str:
    """Return a finished completion. Used by the eval harness and the judge."""
    if provider_for(model) == "anthropic":
        return _anthropic_complete(model, system, user)
    return _openai_complete(model, system, user)


def stream(model: str, system: str, user: str) -> Iterator[str]:
    """Yield completion text as it arrives. Used by the web API."""
    if provider_for(model) == "anthropic":
        yield from _anthropic_stream(model, system, user)
    else:
        yield from _openai_stream(model, system, user)


def _openai_complete(model: str, system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_key("OPENAI_API_KEY", "openai"), max_retries=3)
    response = client.chat.completions.create(
        model=model,
        temperature=OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise Refused(f"{model} declined to generate for this input.")
    return choice.message.content or ""


def _openai_stream(model: str, system: str, user: str) -> Iterator[str]:
    from openai import OpenAI, Stream
    from openai.types.chat import ChatCompletionChunk

    client = OpenAI(api_key=_key("OPENAI_API_KEY", "openai"), max_retries=3)
    # create() is overloaded on the literal value of `stream`, and mypy cannot
    # narrow it through a keyword argument, so the streaming return type is
    # asserted once here rather than silenced at each use site.
    chunks = cast(
        "Stream[ChatCompletionChunk]",
        client.chat.completions.create(
            model=model,
            temperature=OPENAI_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        ),
    )
    for chunk in chunks:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _anthropic_complete(model: str, system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "anthropic"), max_retries=3)
    # `system` is a top-level parameter here, not a message with a role. No
    # temperature is sent: the newer Claude models reject it outright, and the
    # older ones behave acceptably at their default.
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise Refused(f"{model} declined to generate for this input.")
    # Thinking blocks may precede the answer; only text blocks are the answer.
    return "".join(block.text for block in response.content if block.type == "text")


def _anthropic_stream(model: str, system: str, user: str) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "anthropic"), max_retries=3)
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as events:
        # text_stream yields only text deltas, so thinking output cannot leak
        # into the answer shown to the user.
        yield from events.text_stream
        if events.get_final_message().stop_reason == "refusal":
            raise Refused(f"{model} declined to generate for this input.")
