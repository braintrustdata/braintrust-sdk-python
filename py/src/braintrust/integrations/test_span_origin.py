"""Cross-integration assertions for `context.span_origin.instrumentation.name`.

Each integration's `tracing.py` (or `callbacks.py`/`plugin.py`) shadows the
module-level `start_span` to stamp the integration's identifier. This test
suite opens a span through each shadow and confirms the emitted context
carries the expected `<provider>-auto` name. Provider packages are NOT
imported here — only the local shadow — so the file runs cleanly under
`test_core`.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from braintrust import logger
from braintrust.test_helpers import init_test_logger

# Each entry: (module path, expected `<provider>-auto` name).
_SHADOW_TARGETS: list[tuple[str, str]] = [
    ("braintrust.integrations.adk.tracing", "adk-auto"),
    ("braintrust.integrations.agentscope.tracing", "agentscope-auto"),
    ("braintrust.integrations.agno.tracing", "agno-auto"),
    ("braintrust.integrations.anthropic.tracing", "anthropic-auto"),
    ("braintrust.integrations.autogen.tracing", "autogen-auto"),
    ("braintrust.integrations.bedrock_runtime.tracing", "bedrock-runtime-auto"),
    ("braintrust.integrations.claude_agent_sdk.tracing", "claude-agent-sdk-auto"),
    ("braintrust.integrations.cohere.tracing", "cohere-auto"),
    ("braintrust.integrations.crewai.tracing", "crewai-auto"),
    ("braintrust.integrations.dspy.tracing", "dspy-auto"),
    ("braintrust.integrations.google_genai.tracing", "google-genai-auto"),
    ("braintrust.integrations.huggingface_hub.tracing", "huggingface-hub-auto"),
    ("braintrust.integrations.instructor.tracing", "instructor-auto"),
    ("braintrust.integrations.langchain.callbacks", "langchain-auto"),
    ("braintrust.integrations.litellm.tracing", "litellm-auto"),
    ("braintrust.integrations.livekit_agents.tracing", "livekit-agents-auto"),
    ("braintrust.integrations.llamaindex.tracing", "llamaindex-auto"),
    ("braintrust.integrations.mistral.tracing", "mistral-auto"),
    ("braintrust.integrations.openai.tracing", "openai-auto"),
    ("braintrust.integrations.openai_agents.tracing", "openai-agents-auto"),
    ("braintrust.integrations.openrouter.tracing", "openrouter-auto"),
    ("braintrust.integrations.pydantic_ai.tracing", "pydantic-ai-auto"),
]

_PROJECT = "test-span-origin"


@pytest.fixture
def memory_logger() -> Any:
    init_test_logger(_PROJECT)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl
    logger._state.current_experiment = None
    logger._state.reset_parent_state()


def _try_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


@pytest.mark.parametrize(("module_path", "expected"), _SHADOW_TARGETS)
def test_integration_shadow_declares_instrumentation(module_path: str, expected: str) -> None:
    """Every integration exposes `_INSTRUMENTATION = "<provider>-auto"` at module scope."""
    module = _try_import(module_path)
    if module is None:
        pytest.skip(f"{module_path}: optional provider dep not installed")
    assert getattr(module, "_INSTRUMENTATION", None) == expected, (
        f"{module_path} should declare _INSTRUMENTATION = {expected!r}"
    )


@pytest.mark.parametrize(("module_path", "expected"), _SHADOW_TARGETS)
def test_integration_shadow_stamps_span_origin(module_path: str, expected: str, memory_logger) -> None:
    """A span opened via the integration's local `start_span` carries the expected instrumentation name."""
    module = _try_import(module_path)
    if module is None:
        pytest.skip(f"{module_path}: optional provider dep not installed")
    start_span = getattr(module, "start_span", None)
    if start_span is None:
        pytest.skip(f"{module_path}: no module-level start_span shadow")

    with start_span(name="span-origin-check") as span:
        # SpanImpl exposes the resolved instrumentation name as `_instrumentation`.
        assert getattr(span, "_instrumentation", None) == expected

    spans = memory_logger.pop()
    assert spans, f"{module_path}: expected at least one logged span"
    origin = spans[0]["context"]["span_origin"]
    assert origin["instrumentation"]["name"] == expected, (
        f"{module_path} emitted {origin['instrumentation']['name']!r}, expected {expected!r}"
    )


def test_temporal_plugin_declares_instrumentation() -> None:
    """Temporal passes `instrumentation="temporal-auto"` at every `logger.start_span(...)` site."""
    from pathlib import Path

    src = Path(__file__).parent.joinpath("temporal", "plugin.py").read_text()
    # Every logger.start_span( occurrence should be followed nearby by
    # instrumentation="temporal-auto".
    logger_calls = src.count("logger.start_span(")
    stamped = src.count('instrumentation="temporal-auto"')
    assert stamped == logger_calls, (
        f"temporal/plugin.py has {logger_calls} logger.start_span() sites but only "
        f"{stamped} carry instrumentation=\"temporal-auto\""
    )


def test_openai_agents_stamps_instrumentation_on_method_start_span() -> None:
    """openai_agents opens spans via `parent.start_span(...)` and `logger.start_span(...)`;
    every such site must pass `instrumentation=_INSTRUMENTATION` explicitly."""
    from pathlib import Path

    src = Path(__file__).parent.joinpath("openai_agents", "tracing.py").read_text()
    # Both `current_context.start_span(...)` and `self._logger.start_span(...)`
    # and `parent.start_span(...)` should include instrumentation=_INSTRUMENTATION.
    method_calls = (
        src.count("current_context.start_span(")
        + src.count("self._logger.start_span(")
        + src.count("parent.start_span(")
    )
    stamped = src.count("instrumentation=_INSTRUMENTATION")
    assert stamped == method_calls, (
        f"openai_agents/tracing.py has {method_calls} span-method call sites but only "
        f"{stamped} carry instrumentation=_INSTRUMENTATION"
    )
