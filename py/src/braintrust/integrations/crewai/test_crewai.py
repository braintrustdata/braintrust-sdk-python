"""Tests for the Braintrust CrewAI integration.

The integration instruments CrewAI by subscribing to its event bus, so the
most precise tests are to emit CrewAI events directly against the bus and
assert the resulting Braintrust spans.  That avoids the well-known
pytest-vcr + httpcore interaction bug triggered by CrewAI's native
provider classes (they eagerly build an openai.OpenAI client during
``OpenAICompletion`` pydantic ``model_post_init``, which replaces
``httpcore.ConnectionPool.handle_request`` on the class after vcrpy has
patched it, and real HTTP traffic escapes the cassette).

We keep:

- direct-event unit tests for the span shaping logic — these are both the
  source of truth for the integration and fully deterministic
- a LiteLLM ``mock_response`` smoke test that exercises the full
  ``crew.kickoff()`` code path without any network I/O.
"""

# pylint: disable=import-error

import time
from typing import Any

import pytest
from braintrust import logger
from braintrust.integrations.crewai import (
    BraintrustCrewAIListener,
    CrewAIIntegration,
    patch_crewai,
    setup_crewai,
)
from braintrust.integrations.crewai.patchers import _get_registered_listener, _reset_for_testing
from braintrust.integrations.test_utils import run_in_subprocess, verify_autoinstrument_script
from braintrust.logger import Attachment, start_span
from braintrust.test_helpers import init_test_logger
from braintrust.util import LazyValue


PROJECT_NAME = "test-crewai-app"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_logger():
    """Install a cross-thread memory background logger.

    The stock ``_internal_with_memory_background_logger`` uses a
    ``threading.local`` override, but CrewAI's event bus dispatches sync
    handlers through a ``ThreadPoolExecutor`` whose workers cannot see the
    override.  Swapping ``_state._global_bg_logger`` directly is visible
    from any thread for the lifetime of the fixture.
    """
    init_test_logger(PROJECT_NAME)
    ml = logger._MemoryBackgroundLogger()
    state = logger._state
    original = state._global_bg_logger
    state._global_bg_logger = LazyValue(lambda: ml, use_mutex=False)
    try:
        yield ml
    finally:
        state._global_bg_logger = original


@pytest.fixture(autouse=True)
def _reset_listener():
    """Force a clean CrewAI listener for each test.

    The module-level ``_LISTENER`` singleton leaks between tests otherwise.
    We deliberately do *not* touch LiteLLM patch markers here: stripping them
    would confuse ``is_patched`` and let a subsequent ``patch_litellm()``
    double-wrap the module. The leaf-only token-metric tests below instead
    monkeypatch ``is_litellm_patched`` directly when they need to pretend
    LiteLLM is (or is not) patched.
    """
    _reset_for_testing()
    yield


# ---------------------------------------------------------------------------
# Helpers to build minimal CrewAI event payloads
# ---------------------------------------------------------------------------


def _flush_event_bus(event_bus: Any, timeout: float) -> None:
    """Best-effort wrapper around ``CrewAIEventsBus.flush``.

    CrewAI exposes ``flush`` at runtime, but pylint cannot infer it reliably
    from the package's dynamic event-bus surface.
    """
    flush = getattr(event_bus, "flush", None)
    if callable(flush):
        flush(timeout=timeout)


def _emit(event: Any) -> None:
    """Emit *event* on the crewai bus and wait for the sync handlers to finish."""
    from crewai.events.event_bus import crewai_event_bus

    future = crewai_event_bus.emit(None, event)
    if future is not None:
        # Handlers run on a ThreadPoolExecutor; wait for them synchronously.
        try:
            future.result(timeout=5.0)
        except Exception:
            pass
    _flush_event_bus(crewai_event_bus, timeout=5.0)


def _build_kickoff_started(**overrides: Any) -> Any:
    from crewai.events import CrewKickoffStartedEvent

    payload = dict(crew_name="Test Crew", inputs={"topic": "arith"}, crew=None)
    payload.update(overrides)
    return CrewKickoffStartedEvent(**payload)


def _build_kickoff_completed(started: Any, output: Any = "final answer") -> Any:
    from crewai.events import CrewKickoffCompletedEvent

    return CrewKickoffCompletedEvent(
        crew_name=started.crew_name,
        crew=None,
        output=output,
        parent_event_id=started.parent_event_id,
        started_event_id=started.event_id,
    )


def _build_llm_started(
    *,
    messages: Any = None,
    parent_event_id: str | None = None,
    model: str = "gpt-4o-mini",
    tools: Any = None,
) -> Any:
    from crewai.events import LLMCallStartedEvent

    return LLMCallStartedEvent(
        model=model,
        call_id="call-1",
        messages=messages or [{"role": "user", "content": "2+2?"}],
        tools=tools,
        parent_event_id=parent_event_id,
    )


def _build_llm_completed(started: Any, usage: dict[str, Any] | None = None, response: Any = "4") -> Any:
    from crewai.events import LLMCallCompletedEvent
    from crewai.events.types.llm_events import LLMCallType

    return LLMCallCompletedEvent(
        model=started.model,
        call_id=started.call_id,
        messages=started.messages,
        response=response,
        call_type=LLMCallType.LLM_CALL,
        usage=usage,
        parent_event_id=started.parent_event_id,
        started_event_id=started.event_id,
    )


def _build_llm_failed(started: Any, error: str = "boom") -> Any:
    from crewai.events import LLMCallFailedEvent

    return LLMCallFailedEvent(
        model=started.model,
        call_id=started.call_id,
        error=error,
        parent_event_id=started.parent_event_id,
        started_event_id=started.event_id,
    )


def _build_stream_chunk(started: Any, chunk: str = "hi") -> Any:
    from crewai.events import LLMStreamChunkEvent

    return LLMStreamChunkEvent(
        model=started.model,
        call_id=started.call_id,
        chunk=chunk,
        parent_event_id=started.event_id,  # chunks live under the open LLM span
    )


def _build_tool_started(
    *, parent_event_id: str | None = None, tool_name: str = "search", tool_args: Any = None
) -> Any:
    from crewai.events import ToolUsageStartedEvent

    return ToolUsageStartedEvent(
        tool_name=tool_name,
        tool_args=tool_args or {"query": "weather"},
        parent_event_id=parent_event_id,
    )


def _build_tool_finished(started: Any, output: Any = "result") -> Any:
    import datetime

    from crewai.events import ToolUsageFinishedEvent

    now = datetime.datetime.now()
    return ToolUsageFinishedEvent(
        tool_name=started.tool_name,
        tool_args=started.tool_args,
        output=output,
        started_at=now,
        finished_at=now,
        from_cache=False,
        run_attempts=1,
        parent_event_id=started.parent_event_id,
        started_event_id=started.event_id,
    )


def _build_tool_error(started: Any, error: str = "tool crashed") -> Any:
    from crewai.events import ToolUsageErrorEvent

    return ToolUsageErrorEvent(
        tool_name=started.tool_name,
        tool_args=started.tool_args,
        error=error,
        parent_event_id=started.parent_event_id,
        started_event_id=started.event_id,
    )


def _spans_by_name(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        name = span["span_attributes"]["name"]
        out.setdefault(name, []).append(span)
    return out


# ---------------------------------------------------------------------------
# Event -> span shape tests
# ---------------------------------------------------------------------------


def test_kickoff_llm_event_tree_parents_and_shape(memory_logger):
    """Emitted events must map to a kickoff -> llm Braintrust span tree."""
    patch_crewai()

    kickoff = _build_kickoff_started()
    _emit(kickoff)
    llm_started = _build_llm_started(parent_event_id=kickoff.event_id)
    _emit(llm_started)
    _emit(_build_llm_completed(llm_started, usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}))
    _emit(_build_kickoff_completed(kickoff))

    spans = memory_logger.pop()
    by_name = _spans_by_name(spans)
    assert set(by_name) == {"crewai.kickoff", "crewai.llm"}, sorted(by_name)

    kickoff_span = by_name["crewai.kickoff"][0]
    llm_span = by_name["crewai.llm"][0]

    assert kickoff_span["span_attributes"]["type"] == "task"
    assert llm_span["span_attributes"]["type"] == "llm"

    # LLM span parents onto the kickoff span via CrewAI's causal chain.
    assert llm_span["span_parents"][0] == kickoff_span["span_id"]

    # Causal ids propagate so users can correlate spans with raw events.
    assert kickoff_span["metadata"]["crewai_event_id"] == kickoff.event_id
    assert llm_span["metadata"]["crewai_event_id"] == llm_started.event_id
    assert llm_span["metadata"]["crewai_parent_event_id"] == kickoff.event_id

    # Shape assertions.
    assert llm_span["input"]["messages"] == llm_started.messages
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["metadata"]["call_id"] == "call-1"
    assert llm_span["output"] == "4"
    assert kickoff_span["output"] == "final answer"
    assert kickoff_span["input"] == kickoff.inputs


def test_llm_tokens_skipped_when_litellm_patched(memory_logger, monkeypatch):
    """Leaf-only rule: LiteLLM patched -> no token metrics on crewai.llm."""
    # Pretend LiteLLM is patched without actually wrapping the module, so
    # tests later in the suite that rely on a clean LiteLLM still see it.
    monkeypatch.setattr(
        "braintrust.integrations.crewai.tracing._litellm_owns_leaf_span",
        lambda: True,
    )
    patch_crewai()

    llm_started = _build_llm_started()
    _emit(llm_started)
    _emit(
        _build_llm_completed(
            llm_started,
            usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        )
    )

    spans = memory_logger.pop()
    by_name = _spans_by_name(spans)
    assert by_name.get("crewai.llm"), f"Missing crewai.llm in {sorted(by_name)}"
    metrics = by_name["crewai.llm"][0]["metrics"]

    assert "start" in metrics and "end" in metrics
    for token_key in ("tokens", "prompt_tokens", "completion_tokens"):
        assert token_key not in metrics, f"crewai.llm leaked {token_key}={metrics[token_key]} while litellm is patched"


def test_llm_tokens_emitted_when_litellm_not_patched(memory_logger, monkeypatch):
    """Leaf: LiteLLM unpatched -> crewai.llm owns token metrics."""
    monkeypatch.setattr(
        "braintrust.integrations.crewai.tracing._litellm_owns_leaf_span",
        lambda: False,
    )
    patch_crewai()

    llm_started = _build_llm_started()
    _emit(llm_started)
    _emit(
        _build_llm_completed(
            llm_started,
            usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        )
    )

    metrics = memory_logger.pop()[0]["metrics"]
    assert metrics["prompt_tokens"] == 11
    assert metrics["completion_tokens"] == 22
    assert metrics["tokens"] == 33


def test_llm_call_failed_logs_error(memory_logger):
    patch_crewai()

    started = _build_llm_started()
    _emit(started)
    _emit(_build_llm_failed(started, error="upstream 500"))

    span = memory_logger.pop()[0]
    assert span["span_attributes"]["name"] == "crewai.llm"
    assert "upstream 500" in str(span.get("error"))


def test_llm_streaming_time_to_first_token(memory_logger):
    patch_crewai()

    started = _build_llm_started()
    _emit(started)
    # Simulate a slight gap before the first token arrives so the metric
    # is a visible positive number.
    time.sleep(0.01)
    _emit(_build_stream_chunk(started, chunk="part1"))
    _emit(_build_stream_chunk(started, chunk="part2"))  # second chunk shouldn't overwrite
    _emit(_build_llm_completed(started))

    metrics = memory_logger.pop()[0]["metrics"]
    assert "time_to_first_token" in metrics
    assert metrics["time_to_first_token"] > 0


def test_tool_usage_span(memory_logger):
    patch_crewai()

    kickoff = _build_kickoff_started()
    _emit(kickoff)
    tool_started = _build_tool_started(parent_event_id=kickoff.event_id, tool_name="search")
    _emit(tool_started)
    _emit(_build_tool_finished(tool_started, output={"count": 3}))
    _emit(_build_kickoff_completed(kickoff))

    by_name = _spans_by_name(memory_logger.pop())
    tool_span = by_name["crewai.tool.search"][0]
    assert tool_span["span_attributes"]["type"] == "tool"
    assert tool_span["span_parents"][0] == by_name["crewai.kickoff"][0]["span_id"]
    assert tool_span["input"] == tool_started.tool_args
    assert tool_span["output"] == {"count": 3}
    assert tool_span["metadata"]["tool_name"] == "search"
    assert tool_span["metadata"]["run_attempts"] == 1
    assert tool_span["metadata"]["from_cache"] is False


def test_tool_error_logs_error(memory_logger):
    patch_crewai()

    tool_started = _build_tool_started(tool_name="search")
    _emit(tool_started)
    _emit(_build_tool_error(tool_started, error="network unreachable"))

    span = memory_logger.pop()[0]
    assert span["span_attributes"]["name"] == "crewai.tool.search"
    assert "network unreachable" in str(span.get("error"))


def test_multimodal_messages_materialize_attachments(memory_logger):
    """Base64 image parts in ``messages`` should be converted to Attachments."""
    patch_crewai()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ],
        }
    ]

    started = _build_llm_started(messages=messages)
    _emit(started)
    _emit(_build_llm_completed(started))

    span = memory_logger.pop()[0]
    materialized_content = span["input"]["messages"][0]["content"]
    image_part = next(part for part in materialized_content if part.get("type") == "image_url")
    url_value = image_part["image_url"]["url"]
    assert isinstance(url_value, Attachment), f"Expected Attachment, got {type(url_value).__name__}"
    assert url_value.reference["content_type"] == "image/png"


def test_kickoff_end_clears_orphaned_span_state(memory_logger):
    """Orphan entries from missing inner end-events must not leak past kickoff end.

    Simulates a failure path where the inner LLM scope never emits its
    completion event: when the outer kickoff closes, the listener's span
    maps should be empty again so long-running services do not accumulate
    orphaned entries.
    """
    patch_crewai()
    listener = _get_registered_listener()

    kickoff = _build_kickoff_started()
    _emit(kickoff)
    llm_started = _build_llm_started(parent_event_id=kickoff.event_id)
    _emit(llm_started)
    # Deliberately *no* LLMCallCompletedEvent — simulates a dropped event.
    _emit(_build_kickoff_completed(kickoff))

    assert listener._spans == {}
    assert listener._span_start_times == {}
    assert listener._first_token_times == {}


def test_open_span_ignored_when_event_has_no_id(memory_logger):
    """No ``event_id`` on the start event means no span opened, no leak."""
    patch_crewai()
    listener = _get_registered_listener()

    llm_started = _build_llm_started()
    llm_started.event_id = ""
    _emit(llm_started)

    assert listener._spans == {}
    assert not memory_logger.pop()


def test_nested_under_user_span(memory_logger):
    """When a user opens an outer Braintrust span, kickoff spans nest under it."""
    patch_crewai()

    with start_span(name="user.outer") as outer:
        outer_id = outer.span_id
        kickoff = _build_kickoff_started()
        _emit(kickoff)
        _emit(_build_kickoff_completed(kickoff))

    by_name = _spans_by_name(memory_logger.pop())
    kickoff_span = by_name["crewai.kickoff"][0]
    assert kickoff_span["span_parents"][0] == outer_id


# ---------------------------------------------------------------------------
# End-to-end smoke test via LiteLLM mock_response (no network)
# ---------------------------------------------------------------------------


def test_crew_kickoff_smoke_via_litellm_mock(memory_logger):
    """Drive a full ``crew.kickoff()`` path with no network I/O.

    CrewAI 1.x routes to a native openai provider for ``gpt-4o-mini``,
    which eagerly instantiates an openai client whose httpcore layer
    confuses pytest-vcr (see the module docstring).  We side-step that by
    forcing ``is_litellm=True`` and using LiteLLM's ``mock_response`` to
    return a canned answer without touching the network.
    """
    patch_crewai()

    from crewai import LLM, Agent, Crew, Task

    llm = LLM(model="gpt-4o-mini", is_litellm=True, mock_response="24")
    agent = Agent(
        role="Calculator",
        goal="Answer arithmetic",
        backstory="Respond with only the number.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        tools=[],
    )
    task = Task(description="What is 12 + 12?", expected_output="A number", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    result = crew.kickoff()
    assert result is not None

    from crewai.events.event_bus import crewai_event_bus

    _flush_event_bus(crewai_event_bus, timeout=10.0)

    by_name = _spans_by_name(memory_logger.pop())
    # The full scope family must be present.
    for expected in ("crewai.kickoff", "crewai.task", "crewai.agent", "crewai.llm"):
        assert by_name.get(expected), f"Missing {expected} in {sorted(by_name)}"

    llm_span = by_name["crewai.llm"][0]
    agent_span = by_name["crewai.agent"][0]
    task_span = by_name["crewai.task"][0]
    kickoff_span = by_name["crewai.kickoff"][0]

    # The direct-event tests above own the strict parent-chain assertions. The
    # real ``crew.kickoff()`` path dispatches handlers on CrewAI's thread pool,
    # and some intermediary runtime events do not always carry the same causal
    # ids across Python / platform combinations. For this smoke test, require
    # only that any recorded parent points at another span in the observed
    # CrewAI scope family.
    family_span_ids = {span["span_id"] for span in (llm_span, agent_span, task_span, kickoff_span)}
    for span in (llm_span, agent_span, task_span):
        parents = span.get("span_parents") or []
        if parents:
            assert parents[0] in family_span_ids - {span["span_id"]}

    # Metadata captured from real CrewAI objects.
    assert agent_span["metadata"].get("agent_role") == "Calculator"


# ---------------------------------------------------------------------------
# Listener lifecycle tests
# ---------------------------------------------------------------------------


def test_integration_setup_is_idempotent():
    """Registering the listener twice must not stack handlers."""
    from crewai.events import CrewKickoffStartedEvent
    from crewai.events.event_bus import crewai_event_bus

    assert CrewAIIntegration.setup() is True
    listener1 = _get_registered_listener()
    assert listener1 is not None

    before = len(crewai_event_bus._sync_handlers.get(CrewKickoffStartedEvent, frozenset()))
    assert CrewAIIntegration.setup() is True
    listener2 = _get_registered_listener()
    assert listener2 is listener1
    after = len(crewai_event_bus._sync_handlers.get(CrewKickoffStartedEvent, frozenset()))
    assert before == after, "Repeated setup() should not register additional handlers"


def test_listener_is_braintrust_listener_instance():
    """The registered listener must satisfy ``isinstance(BraintrustCrewAIListener)``."""
    assert CrewAIIntegration.setup() is True
    assert isinstance(_get_registered_listener(), BraintrustCrewAIListener)


def test_setup_crewai_returns_true_under_active_logger():
    """``setup_crewai()`` must not crash when a logger is already active."""
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger():
        assert setup_crewai() is True


# ---------------------------------------------------------------------------
# Auto-instrument / subprocess tests
# ---------------------------------------------------------------------------


class TestAutoInstrumentCrewAI:
    def test_auto_instrument_crewai(self):
        verify_autoinstrument_script("test_auto_crewai.py")

    def test_patch_crewai_subprocess(self):
        result = run_in_subprocess(
            """
            from braintrust.integrations.crewai import patch_crewai
            from braintrust.integrations.crewai.patchers import _get_registered_listener
            assert patch_crewai()
            assert _get_registered_listener() is not None
            assert patch_crewai()  # idempotent
            print("SUCCESS")
            """
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout
