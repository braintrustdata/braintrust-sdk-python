"""CrewAI tracing helpers — event-bus listener and span shaping.

This module implements :class:`BraintrustCrewAIListener`, a single
:class:`crewai.events.BaseEventListener` subclass that subscribes to the
CrewAI event bus and materializes each scope (crew, task, agent, LLM call,
tool call) as a Braintrust span.

We use the event-bus path instead of method patching for two reasons:

1. CrewAI's singleton ``crewai_event_bus`` already gives us precise
   ``started`` / ``completed`` / ``failed`` events with ``event_id`` /
   ``parent_event_id`` / ``started_event_id`` causal fields, so we do not
   need to wrap four different methods to reconstruct the hierarchy.
2. The bus dispatches sync handlers through a ``ThreadPoolExecutor`` with
   ``contextvars.copy_context()``, so the caller's Braintrust
   ``current_span()`` is visible when each handler fires. That means nested
   spans opened here correctly parent onto any user-opened outer span.

Thread-safety: the bus uses up to ``max_workers=10`` workers, so start and
end events for the same scope can race across threads. A listener-level
:class:`threading.Lock` guards the ``event_id -> Span`` map.

Token-metric rule (leaf-only):

CrewAI delegates LLM calls to LiteLLM.  When the LiteLLM integration is
also patched, the ``Completion`` span LiteLLM produces is the leaf span
and already owns token accounting.  Emitting tokens on the enclosing
``crewai.llm`` span in that configuration would make trace-tree rollup
double-count the same tokens at every ancestor.

We therefore follow the same pattern as the pydantic_ai integration
(see ``_wrapper_span_metrics`` in ``pydantic_ai/tracing.py``):
``crewai.llm`` emits timing + ``time_to_first_token`` unconditionally, and
token metrics only when LiteLLM is *not* patched.  The check happens at
log time via the internal ``_is_litellm_patched`` helper on the LiteLLM
integration.
"""

# pylint: disable=import-error

import threading
import time
from typing import TYPE_CHECKING, Any

from braintrust.integrations.utils import (
    _normalize_chat_messages,
    _parse_openai_usage_metrics,
    _try_to_dict,
)
from braintrust.logger import NOOP_SPAN, Span, current_span, start_span
from braintrust.span_types import SpanTypeAttribute


if TYPE_CHECKING:
    from crewai.events.event_bus import CrewAIEventsBus


# LiteLLM / LiteLLM-over-OpenAI usage field translation.  Mirrors the
# mapping in ``braintrust.integrations.litellm.tracing`` so ``crewai.llm``
# spans use identical metric names when LiteLLM is not the leaf.
_TOKEN_NAME_MAP: dict[str, str] = {
    "total_tokens": "tokens",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "tokens": "tokens",
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
}

_TOKEN_PREFIX_MAP: dict[str, str] = {
    "input": "prompt",
    "output": "completion",
}


# Provider parameters we want to surface on `crewai.llm` metadata.
_LLM_CONFIG_FIELDS: tuple[str, ...] = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "top_k",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stream",
    "reasoning_effort",
    "response_format",
    "logprobs",
    "timeout",
)


def _litellm_owns_leaf_span() -> bool:
    """Return True when LiteLLM's completion entry points have been patched.

    Late-imported so that importing the CrewAI integration never forces a
    LiteLLM import.  Errors are swallowed so a broken LiteLLM install
    never prevents CrewAI tracing from running.
    """
    try:
        from braintrust.integrations.litellm import _is_litellm_patched

        return _is_litellm_patched()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Payload normalization helpers
# ---------------------------------------------------------------------------


def _agent_metadata(agent: Any) -> dict[str, Any]:
    """Extract identity + configuration metadata from a CrewAI agent object."""
    if agent is None:
        return {}
    meta: dict[str, Any] = {}
    for attr in ("id", "role", "goal", "backstory", "allow_delegation", "verbose"):
        value = getattr(agent, attr, None)
        if value is not None and not callable(value):
            meta[f"agent_{attr}"] = str(value) if attr == "id" else value
    llm = getattr(agent, "llm", None)
    if llm is not None:
        meta["agent_llm"] = getattr(llm, "model", None) or str(llm)
    return meta


def _task_metadata(task: Any) -> dict[str, Any]:
    """Extract identity metadata from a CrewAI task object."""
    if task is None:
        return {}
    meta: dict[str, Any] = {}
    for attr in ("id", "name", "description", "expected_output", "async_execution", "human_input"):
        value = getattr(task, attr, None)
        if value is not None and not callable(value):
            meta[f"task_{attr}"] = str(value) if attr == "id" else value
    return meta


def _crew_metadata(crew: Any) -> dict[str, Any]:
    """Extract identity metadata from a CrewAI crew object."""
    if crew is None:
        return {}
    meta: dict[str, Any] = {}
    for attr in ("id", "name", "process", "verbose"):
        value = getattr(crew, attr, None)
        if value is not None and not callable(value):
            meta[f"crew_{attr}"] = str(value) if attr in ("id", "process") else value
    fingerprint = getattr(crew, "fingerprint", None)
    if fingerprint is not None:
        uuid_str = getattr(fingerprint, "uuid_str", None)
        if uuid_str:
            meta["crew_fingerprint"] = uuid_str
    return meta


def _causal_metadata(event: Any) -> dict[str, Any]:
    """Return the CrewAI causal-id fields we propagate to span metadata."""
    meta: dict[str, Any] = {}
    for attr, out_key in (
        ("event_id", "crewai_event_id"),
        ("parent_event_id", "crewai_parent_event_id"),
        ("source_fingerprint", "crewai_source_fingerprint"),
        ("source_type", "crewai_source_type"),
    ):
        value = getattr(event, attr, None)
        if value:
            meta[out_key] = value
    return meta


def _llm_config_metadata(source: Any) -> dict[str, Any]:
    """Extract provider-config fields from the emitting ``LLM`` object."""
    if source is None:
        return {}
    meta: dict[str, Any] = {}
    for field in _LLM_CONFIG_FIELDS:
        value = getattr(source, field, None)
        if value is None or callable(value):
            continue
        meta[field] = value
    return meta


def _normalize_tools(tools: Any) -> Any:
    """Normalize a list of CrewAI tool descriptors for logging."""
    if tools is None:
        return None
    if isinstance(tools, (list, tuple)):
        out = []
        for tool in tools:
            if isinstance(tool, dict):
                out.append(tool)
            else:
                out.append(_try_to_dict(tool))
        return out
    return _try_to_dict(tools)


def _normalize_output(value: Any) -> Any:
    """Coerce provider-owned output objects into plain dicts/strings for logging."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, dict, list)):
        return value
    coerced = _try_to_dict(value)
    if coerced is value and not isinstance(value, (str, int, float, bool, dict, list)):
        # Last-resort: render as a string rather than log a raw SDK object.
        return str(value)
    return coerced


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


_BASE_LISTENER_CLS: Any  # populated lazily below


def _get_base_listener_cls() -> Any:
    """Return ``crewai.events.BaseEventListener`` (lazily imported).

    CrewAI is an optional dep, so we defer this import.  Loading the
    listener class lazily also avoids executing module-level event-bus
    side-effects at ``braintrust`` import time.
    """
    global _BASE_LISTENER_CLS  # noqa: PLW0603
    try:
        return _BASE_LISTENER_CLS
    except NameError:
        pass
    from crewai.events.base_event_listener import BaseEventListener

    _BASE_LISTENER_CLS = BaseEventListener
    return _BASE_LISTENER_CLS


class BraintrustCrewAIListener:
    """CrewAI event-bus listener that maps events into Braintrust spans.

    The concrete runtime class is built lazily via ``__new__`` and
    subclasses :class:`crewai.events.BaseEventListener` in addition to
    this class, so ``isinstance(listener, BraintrustCrewAIListener)``
    remains true while :meth:`crewai.events.BaseEventListener.__init__`
    performs its event-bus registration.

    Deferring CrewAI's import until instantiation keeps this module
    import-cheap for users who never touch CrewAI.

    Advanced users can instantiate this class manually (e.g. in tests or
    when owning a custom event bus) instead of going through
    :func:`setup_crewai`.
    """

    # Populated after lazy class generation. Isolated per subclass so a
    # subclass does not accidentally reuse the parent's runtime class.
    _cls: type | None = None

    # Instance state — populated by ``_listener_init`` on the runtime
    # subclass. Declared here so static analyzers (pylint, pyright) know
    # they exist without having to inspect the dynamic class.
    _spans: "dict[str, Span]"
    _span_start_times: "dict[str, float]"
    _first_token_times: "dict[str, float]"
    _lock: "threading.Lock"

    def __new__(cls, *args: Any, **kwargs: Any) -> "BraintrustCrewAIListener":
        # Build the real subclass on first instantiation so that consumers
        # who never touch CrewAI do not pay its import cost. All instance
        # methods are bound through the type() dict here rather than
        # monkey-patched on the class at module scope, so the pattern is
        # self-contained.
        if cls.__dict__.get("_cls") is None:
            base = _get_base_listener_cls()
            cls._cls = type(
                "_BraintrustCrewAIListenerImpl",
                (cls, base),
                {
                    "__init__": _listener_init,
                    "setup_listeners": _listener_setup_listeners,
                    "_open_span": _open_span,
                    "_end_span": _end_span,
                    "_end_span_by_event_id": _end_span_by_event_id,
                    "_lookup_parent": _lookup_parent,
                    "_record_first_token": _record_first_token,
                    "_clear_span_state": _clear_span_state,
                },
            )
        return object.__new__(cls._cls)


def _listener_init(self: Any) -> None:
    """Real ``__init__`` used by the runtime-built listener subclass."""
    self._spans: dict[str, Span] = {}
    self._span_start_times: dict[str, float] = {}
    self._first_token_times: dict[str, float] = {}
    self._lock = threading.Lock()
    # Calling BaseEventListener.__init__ registers us on crewai_event_bus.
    base = _get_base_listener_cls()
    base.__init__(self)


def _listener_setup_listeners(self: Any, crewai_event_bus: "CrewAIEventsBus") -> None:
    """Wire handlers onto the CrewAI event bus.

    The list of subscribed events matches the v1 scope documented in
    ``BraintrustCrewAIListener``: crew kickoff, tasks, agent execution,
    LLM calls (including streaming first-token timing), and tool usage.
    """
    # Imports are scoped here so that importing this module never forces a
    # CrewAI import.
    from crewai.events import (
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
        CrewKickoffStartedEvent,
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
        LLMStreamChunkEvent,
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskStartedEvent,
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
    from crewai.events.types.agent_events import (
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
    )

    # ------------------------------------------------------------------
    # Crew kickoff
    # ------------------------------------------------------------------

    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def on_crew_kickoff_started(source: Any, event: CrewKickoffStartedEvent) -> None:
        metadata = {**_crew_metadata(getattr(event, "crew", None) or source), **_causal_metadata(event)}
        if getattr(event, "crew_name", None):
            metadata["crew_name"] = event.crew_name
        self._open_span(
            event,
            name="crewai.kickoff",
            span_type=SpanTypeAttribute.TASK,
            input=getattr(event, "inputs", None),
            metadata=metadata,
        )

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def on_crew_kickoff_completed(_source: Any, event: CrewKickoffCompletedEvent) -> None:
        self._end_span(event, output=_normalize_output(getattr(event, "output", None)))
        # Kickoff is the outermost scope; drop any orphan entries left over
        # when an inner end-event was never delivered so state does not grow
        # unbounded in long-running services.
        self._clear_span_state()

    @crewai_event_bus.on(CrewKickoffFailedEvent)
    def on_crew_kickoff_failed(_source: Any, event: CrewKickoffFailedEvent) -> None:
        self._end_span(event, error=getattr(event, "error", None))
        self._clear_span_state()

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @crewai_event_bus.on(TaskStartedEvent)
    def on_task_started(source: Any, event: TaskStartedEvent) -> None:
        task = getattr(event, "task", None) or source
        metadata = {**_task_metadata(task), **_causal_metadata(event)}
        context = getattr(event, "context", None)
        agent = getattr(task, "agent", None)
        if agent is not None:
            metadata.update(_agent_metadata(agent))
        self._open_span(
            event,
            name="crewai.task",
            span_type=SpanTypeAttribute.TASK,
            input={"context": context} if context is not None else None,
            metadata=metadata,
        )

    @crewai_event_bus.on(TaskCompletedEvent)
    def on_task_completed(_source: Any, event: TaskCompletedEvent) -> None:
        self._end_span(event, output=_normalize_output(getattr(event, "output", None)))

    @crewai_event_bus.on(TaskFailedEvent)
    def on_task_failed(_source: Any, event: TaskFailedEvent) -> None:
        self._end_span(event, error=getattr(event, "error", None))

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def on_agent_started(_source: Any, event: AgentExecutionStartedEvent) -> None:
        agent = getattr(event, "agent", None)
        task = getattr(event, "task", None)
        metadata = {
            **_agent_metadata(agent),
            **_task_metadata(task),
            **_causal_metadata(event),
        }
        metadata["tools"] = _normalize_tools(getattr(event, "tools", None))
        self._open_span(
            event,
            name="crewai.agent",
            span_type=SpanTypeAttribute.TASK,
            input={"task_prompt": getattr(event, "task_prompt", None)},
            metadata=metadata,
        )

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def on_agent_completed(_source: Any, event: AgentExecutionCompletedEvent) -> None:
        self._end_span(event, output=getattr(event, "output", None))

    @crewai_event_bus.on(AgentExecutionErrorEvent)
    def on_agent_error(_source: Any, event: AgentExecutionErrorEvent) -> None:
        self._end_span(event, error=getattr(event, "error", None))

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    @crewai_event_bus.on(LLMCallStartedEvent)
    def on_llm_started(source: Any, event: LLMCallStartedEvent) -> None:
        metadata: dict[str, Any] = {
            **_llm_config_metadata(source),
            **_causal_metadata(event),
        }
        if getattr(event, "model", None):
            metadata["model"] = event.model
        if getattr(event, "call_id", None):
            metadata["call_id"] = event.call_id
        available_functions = getattr(event, "available_functions", None)
        if available_functions:
            metadata["available_functions"] = list(available_functions)
        span_input: dict[str, Any] = {
            "messages": getattr(event, "messages", None),
        }
        tools = getattr(event, "tools", None)
        if tools:
            span_input["tools"] = _normalize_tools(tools)
        span_input["messages"] = _normalize_chat_messages(span_input["messages"])
        self._open_span(
            event,
            name="crewai.llm",
            span_type=SpanTypeAttribute.LLM,
            input=span_input,
            metadata=metadata,
        )

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def on_llm_stream_chunk(_source: Any, event: LLMStreamChunkEvent) -> None:
        # Used only for time_to_first_token — the aggregated completion
        # payload comes on LLMCallCompletedEvent.
        self._record_first_token(event)

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def on_llm_completed(_source: Any, event: LLMCallCompletedEvent) -> None:
        metadata: dict[str, Any] = {}
        call_type = getattr(event, "call_type", None)
        if call_type is not None:
            metadata["call_type"] = getattr(call_type, "value", str(call_type))
        if getattr(event, "model", None):
            metadata["model"] = event.model

        # Timing metrics are always safe.  Token metrics are only emitted
        # when LiteLLM is NOT patching the completion entry points, to
        # avoid double-counting with the downstream ``Completion`` span.
        extra_metrics: dict[str, Any] = {}
        if not _litellm_owns_leaf_span():
            usage = getattr(event, "usage", None)
            if usage is not None:
                extra_metrics = _parse_openai_usage_metrics(
                    usage,
                    token_name_map=_TOKEN_NAME_MAP,
                    token_prefix_map=_TOKEN_PREFIX_MAP,
                )

        self._end_span(
            event,
            output=_normalize_output(getattr(event, "response", None)),
            metadata=metadata or None,
            extra_metrics=extra_metrics,
        )

    @crewai_event_bus.on(LLMCallFailedEvent)
    def on_llm_failed(_source: Any, event: LLMCallFailedEvent) -> None:
        self._end_span(event, error=getattr(event, "error", None))

    # ------------------------------------------------------------------
    # Tool usage
    # ------------------------------------------------------------------

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def on_tool_started(_source: Any, event: ToolUsageStartedEvent) -> None:
        tool_name = getattr(event, "tool_name", None) or "tool"
        metadata = {
            "tool_name": tool_name,
            "tool_class": getattr(event, "tool_class", None),
            "run_attempts": getattr(event, "run_attempts", None),
            "delegations": getattr(event, "delegations", None),
            **_causal_metadata(event),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        self._open_span(
            event,
            name=f"crewai.tool.{tool_name}",
            span_type=SpanTypeAttribute.TOOL,
            input=getattr(event, "tool_args", None),
            metadata=metadata,
        )

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def on_tool_finished(_source: Any, event: ToolUsageFinishedEvent) -> None:
        extra_metadata: dict[str, Any] = {}
        from_cache = getattr(event, "from_cache", None)
        if from_cache is not None:
            extra_metadata["from_cache"] = from_cache
        run_attempts = getattr(event, "run_attempts", None)
        if run_attempts is not None:
            extra_metadata["run_attempts"] = run_attempts
        self._end_span(
            event,
            output=_normalize_output(getattr(event, "output", None)),
            metadata=extra_metadata or None,
        )

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def on_tool_error(_source: Any, event: ToolUsageErrorEvent) -> None:
        self._end_span(event, error=getattr(event, "error", None))


# ---------------------------------------------------------------------------
# Shared implementation bound onto the runtime listener subclass
# ---------------------------------------------------------------------------


def _open_span(
    self: Any,
    event: Any,
    *,
    name: str,
    span_type: SpanTypeAttribute,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Span | None:
    """Start a Braintrust span for *event* and record it in the span map.

    Parent resolution prefers the CrewAI causal chain
    (``event.parent_event_id``) so sibling scopes that happen to run on
    different worker threads still parent onto the correct start event.
    When the parent id is unknown (e.g. the outermost kickoff), we fall
    back to the caller's ``current_span()`` — this is how nesting under a
    user-opened outer span works, since the bus's ``copy_context`` carries
    Braintrust's ``ContextVar`` into the handler.

    Returns ``None`` without opening a span if the event carries no
    ``event_id``: without an id we have no way to close the span later, so
    opening one would leak it.
    """
    event_id = getattr(event, "event_id", None)
    if not event_id:
        return None
    parent = self._lookup_parent(event)

    clean_metadata: dict[str, Any] = {k: v for k, v in (metadata or {}).items() if v is not None}
    start_time = time.time()

    start_kwargs: dict[str, Any] = {
        "name": name,
        "span_attributes": {"type": span_type},
        "start_time": start_time,
    }
    if input is not None:
        start_kwargs["input"] = input
    if clean_metadata:
        start_kwargs["metadata"] = clean_metadata
    if parent is not None:
        start_kwargs["parent"] = parent

    span = start_span(**start_kwargs)

    with self._lock:
        self._spans[event_id] = span
        self._span_start_times[event_id] = start_time
    return span


def _lookup_parent(self: Any, event: Any) -> str | None:
    """Resolve the parent-span export for *event* using CrewAI causal fields."""
    parent_event_id = getattr(event, "parent_event_id", None)
    if parent_event_id:
        with self._lock:
            parent_span = self._spans.get(parent_event_id)
        if parent_span is not None:
            return parent_span.export()

    # Fallback: any user-opened Braintrust span (e.g. the outer span in a
    # manual wrapper) that the event-bus executor propagates via
    # ``copy_context``.
    span = current_span()
    if span is NOOP_SPAN:
        return None
    return span.export()


def _record_first_token(self: Any, event: Any) -> None:
    """On the first ``LLMStreamChunkEvent`` for a call, record TTFT.

    Stream chunks are children of the LLM start event, so
    ``parent_event_id`` points at the open ``crewai.llm`` span. We key TTFT
    by that parent id so it matches the span we'll eventually close.
    """
    parent_event_id = getattr(event, "parent_event_id", None)
    if not parent_event_id:
        return
    with self._lock:
        if parent_event_id in self._first_token_times:
            return
        if parent_event_id not in self._spans:
            return
        self._first_token_times[parent_event_id] = time.time()


def _end_span(
    self: Any,
    event: Any,
    *,
    output: Any = None,
    error: Any = None,
    metadata: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """Close the span opened by the matching start event.

    Uses ``event.started_event_id`` when present (CrewAI 1.11+), falling
    back to ``event.event_id`` so we remain tolerant of late-arriving end
    events that are missing scope metadata.
    """
    started_event_id = getattr(event, "started_event_id", None) or getattr(event, "event_id", None)
    if not started_event_id:
        return
    self._end_span_by_event_id(
        started_event_id, output=output, error=error, metadata=metadata, extra_metrics=extra_metrics
    )


def _end_span_by_event_id(
    self: Any,
    started_event_id: str,
    *,
    output: Any = None,
    error: Any = None,
    metadata: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    with self._lock:
        span = self._spans.pop(started_event_id, None)
        start_time = self._span_start_times.pop(started_event_id, None)
        first_token_time = self._first_token_times.pop(started_event_id, None)
    if span is None:
        return

    end_time = time.time()
    metrics: dict[str, Any] = {"end": end_time}
    if first_token_time is not None and start_time is not None:
        metrics["time_to_first_token"] = first_token_time - start_time
    if extra_metrics:
        metrics.update(extra_metrics)

    log_payload: dict[str, Any] = {"metrics": metrics}
    if output is not None:
        log_payload["output"] = output
    if metadata:
        log_payload["metadata"] = metadata
    if error is not None:
        log_payload["error"] = error

    span.log(**log_payload)
    span.end(end_time=end_time)


def _clear_span_state(self: Any) -> None:
    """Drop orphan span state left behind by missing end events.

    Called when the outermost scope (a crew kickoff) ends. Any entries
    still in the maps at that point come from inner scopes whose
    ``*CompletedEvent`` / ``*FailedEvent`` were never delivered, and
    keeping them around would leak for the lifetime of the listener
    (which is a process-level singleton).
    """
    with self._lock:
        self._spans.clear()
        self._span_start_times.clear()
        self._first_token_times.clear()
