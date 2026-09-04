"""Fresh-subprocess sanity check for :func:`braintrust.auto.auto_instrument`.

Usage: ``python _run_smoke.py <integration_name>``

Runs ``auto_instrument(**kwargs)`` twice in a clean Python process and asserts
that the target integration is reported as successfully instrumented on both
calls (i.e. patching works and is idempotent).

Deliberately minimal — span shape, provider metadata, patching topology, and
real API calls are all covered by the in-process ``test_*.py`` files for each
integration.  The only thing a subprocess uniquely proves is: from a cold
Python process, ``auto_instrument()`` successfully sets up this integration.
"""

import sys


# ``auto_instrument`` accepts a bool kwarg per integration and returns a dict
# keyed by a display name (usually identical to the kwarg — ``bedrock_runtime``
# is the one exception, where the kwarg is ``bedrock``).  Each entry below maps
# the display name to any kwarg overrides on top of the all-True defaults.
SMOKES: dict[str, dict[str, bool]] = {
    "adk": {},
    "agentscope": {},
    "agno": {},
    "ai_sdk": {},
    "anthropic": {},
    "autogen": {},
    "bedrock_runtime": {},
    "claude_agent_sdk": {},
    "cohere": {},
    "crewai": {},
    "cursor_sdk": {},
    "dspy": {},
    "google_genai": {},
    "huggingface_hub": {},
    "instructor": {},
    "langchain": {},
    # LiteLLM's OpenAI-backed chat path would otherwise produce both a LiteLLM
    # span and an OpenAI span; disable OpenAI so this smoke stays scoped.
    "litellm": {"openai": False},
    "livekit_agents": {},
    "mistral": {},
    "openai": {},
    "openai_agents": {},
    "openrouter": {},
    "pipecat": {},
    "pydantic_ai": {},
    "temporal": {},
    "transformers": {},
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <integration_name>")
    name = sys.argv[1]
    if name not in SMOKES:
        raise SystemExit(f"unknown integration: {name!r}. Add to SMOKES in {__file__}.")

    from braintrust.auto import auto_instrument

    kwargs = SMOKES[name]

    first = auto_instrument(**kwargs)
    assert first.get(name) is True, f"auto_instrument returned {first!r} for {name!r}"

    second = auto_instrument(**kwargs)
    assert second.get(name) is True, f"auto_instrument (2nd call) returned {second!r} for {name!r}"

    print("SUCCESS")


if __name__ == "__main__":
    main()
