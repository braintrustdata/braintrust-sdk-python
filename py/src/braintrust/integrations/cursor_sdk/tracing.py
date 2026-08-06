"""Tracing lifecycle helpers for the Cursor Python SDK."""

import contextvars
import dataclasses
import json
import time
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from braintrust.integrations.utils import (
    _camel_to_snake,
    _is_supported_metric_value,
    _log_and_end_span,
    _materialize_attachment,
    _try_to_dict,
)
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "cursor-sdk-auto"
# Cursor spells terminal failure statuses both bare ("error") and as wire enum
# reprs ("RUN_LIFECYCLE_STATUS_ERROR"), so match on the suffix.
_ERROR_STATUS_SUFFIXES = ("cancelled", "expired", "error")
# `Run.__init__` primes the event stream, so `_handle_event` can fire before
# `send()` hands back the run we would otherwise hang the tracker off.
_PENDING_TRACKER: contextvars.ContextVar[Any] = contextvars.ContextVar("braintrust_cursor_tracker", default=None)
# Cursor dispatches custom tools on a loopback HTTP thread that receives only an
# agent id -- no run, no agent, and a fresh context -- so that one lookup cannot
# be answered from the call itself. Holding the agent weakly keeps this from
# growing without bound; a caller that drops its agent mid-run loses custom-tool
# attribution but keeps the run trace, which is finished off the run itself.
# Every access below is a single dict/set operation, so no lock is needed.
_AGENTS_BY_ID: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
_AGENT_TRACKERS_ATTR = "_braintrust_cursor_trackers"


def _agent_trackers(agent: Any) -> set[Any]:
    """Return the agent's live tracker set, creating it on first send.

    The set lives on the agent so it is collected with it, and `setdefault` on
    the instance dict keeps concurrent first sends from racing.
    """
    return agent.__dict__.setdefault(_AGENT_TRACKERS_ATTR, set())


def start_span(*args: Any, **kwargs: Any) -> Any:
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    # `dict` first: this runs several times per streamed delta, and the
    # `Mapping` ABC check is markedly slower than the concrete type check.
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    model_id = _value(value, "id")
    return str(model_id) if model_id else None


def _send_model(instance: Any, args: Any, kwargs: Any) -> str | None:
    options = args[1] if len(args) > 1 else kwargs.get("options")
    override = _value(options, "model")
    return _model_id(override) or _model_id(getattr(instance, "model", None))


def _image_part(image: Any) -> dict[str, Any] | None:
    url = _value(image, "url")
    if isinstance(url, str) and url:
        return {"type": "image_url", "image_url": {"url": url}}

    data = _value(image, "data")
    mime_type = _value(image, "mime_type") or _value(image, "mimeType")
    if data is None or not isinstance(mime_type, str) or not mime_type:
        return None
    resolved = _materialize_attachment(data, mime_type=mime_type, label="image")
    if resolved is None:
        return {"type": "image_url", "image_url": {"url": data}}
    return {"type": "image_url", **resolved.multimodal_part_payload}


def _normalize_input(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, str):
        return [{"role": "user", "content": message}]

    text = _value(message, "text", "")
    images = _value(message, "images") or ()
    parts: list[dict[str, Any]] = []
    if isinstance(text, str) and text:
        parts.append({"type": "text", "text": text})
    for image in images:
        part = _image_part(image)
        if part is not None:
            parts.append(part)
    if parts:
        return [{"role": "user", "content": parts}]
    # Keep unknown SDK values intact; Braintrust serializes them at log/export time.
    return [{"role": "user", "content": message}]


def _content_blocks(message: Any) -> Any:
    content = _value(_value(message, "message"), "content", ())
    return content if isinstance(content, (list, tuple)) else ()


def _non_negative_int(value: Any) -> int | None:
    if not _is_supported_metric_value(value):
        return None
    result = int(value)
    return result if result >= 0 else None


def _usage_fields(usage: Any) -> dict[str, Any]:
    """Return usage as a snake_case dict; Cursor reports either casing."""
    payload = _try_to_dict(usage)
    if not isinstance(payload, dict):
        return {}
    return {_camel_to_snake(str(name)): value for name, value in payload.items()}


def _usage_metrics(usage: Any) -> dict[str, int]:
    fields = _usage_fields(usage)
    input_tokens = _non_negative_int(fields.get("input_tokens"))
    output_tokens = _non_negative_int(fields.get("output_tokens"))
    cache_read = _non_negative_int(fields.get("cache_read_tokens"))
    cache_write = _non_negative_int(fields.get("cache_write_tokens"))
    reasoning = _non_negative_int(fields.get("reasoning_tokens"))

    metrics: dict[str, int] = {}
    prompt_parts = [value for value in (input_tokens, cache_read, cache_write) if value is not None]
    prompt_tokens = sum(prompt_parts) if prompt_parts else None
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    if output_tokens is not None:
        metrics["completion_tokens"] = output_tokens
    if prompt_tokens is not None and output_tokens is not None:
        metrics["tokens"] = prompt_tokens + output_tokens
    if cache_read is not None:
        metrics["prompt_cached_tokens"] = cache_read
    if cache_write is not None:
        metrics["prompt_cache_creation_tokens"] = cache_write
    if reasoning is not None:
        metrics["completion_reasoning_tokens"] = reasoning
    return metrics


def _canonical_tool_call(call_id: str, name: str, args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        arguments = args
    else:
        try:
            # OpenAI's canonical tool-call shape specifically requires a JSON string.
            arguments = json.dumps(args)
        except (TypeError, ValueError):
            arguments = str(args)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@dataclasses.dataclass
class _ToolState:
    span: Any
    completed: bool = False

    def finish(self, *, output: Any = None, error: Any = None) -> None:
        if self.completed:
            return
        event: dict[str, Any] = {}
        if output is not None:
            event["output"] = output
        if error is not None:
            event["error"] = error
        if event:
            self.span.log(**event)
        self.span.end()
        self.completed = True


class CursorRunTracker:
    """Own one Cursor parent run and its model/tool child spans."""

    def __init__(self, *, message: Any, model: str | None) -> None:
        self.start_time = time.time()
        self.model = model
        self.history = _normalize_input(message)
        self.root_span = start_span(
            name="Cursor Agent",
            span_attributes={"type": SpanTypeAttribute.TASK},
            input=list(self.history),
            start_time=self.start_time,
            set_current=False,
        )
        self.root_export = self.root_span.export()
        self.run: Any = None
        self.agent_trackers: set[Any] = set()
        self.llm_span: Any = None
        self.llm_text = ""
        self.reasoning_text = ""
        self.llm_tool_calls: list[dict[str, Any]] = []
        self.llm_start = self.start_time
        self.turn_ttft = 0.0
        self.turn_count = 0
        self.tools: dict[str, _ToolState] = {}
        self.aggregate_metrics: dict[str, int] = {}
        # `observe()` is a second, independent stream over the same run, so an
        # observed run can redeliver events already seen via `_apply_event_state`.
        self.seen_events: set[tuple[Any, Any]] = set()
        self.finished = False

    def _update_model(self, candidate: Any) -> None:
        resolved = _model_id(candidate)
        if resolved:
            self.model = resolved

    def attach(self, run: Any, agent: Any) -> None:
        self.run = run
        setattr(run, "_braintrust_cursor_tracker", self)
        self._update_model(getattr(run, "model", None))
        self.agent_trackers = _agent_trackers(agent)
        self.agent_trackers.add(self)
        agent_id = str(getattr(run, "agent_id", "") or "")
        if agent_id:
            _AGENTS_BY_ID[agent_id] = agent
        # Nothing to replay: the run primes its stream inside `send()`, so any
        # event it has already buffered reached the tracker through
        # `_apply_event_state` while `_PENDING_TRACKER` was still set.

    def _start_llm(self, now: float) -> None:
        if self.llm_span is not None:
            return
        # The first turn is anchored to the send so its duration covers the
        # queueing latency Cursor spends before streaming any tokens.
        self.llm_start = self.start_time if self.turn_count == 0 else now
        self.turn_ttft = max(0.0, now - self.llm_start)
        self.llm_span = start_span(
            name="Cursor Model Turn",
            span_attributes={"type": SpanTypeAttribute.LLM},
            input=list(self.history),
            metadata={"model": self.model or "unknown", "provider": "cursor"},
            parent=self.root_export,
            start_time=self.llm_start,
            set_current=False,
        )

    def _assistant_message(self, *, include_reasoning: bool) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.llm_text or None}
        if self.llm_tool_calls:
            message["tool_calls"] = list(self.llm_tool_calls)
        if include_reasoning and self.reasoning_text:
            message["reasoning"] = self.reasoning_text
        return message

    def _llm_output(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if self.reasoning_text:
            output.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": self.reasoning_text}],
                }
            )
        output.append(
            {
                "index": 0,
                "finish_reason": "tool_calls" if self.llm_tool_calls else "stop",
                "message": self._assistant_message(include_reasoning=False),
            }
        )
        return output

    def _finish_llm(self, usage: Any = None) -> None:
        if self.llm_span is None:
            return
        metrics: dict[str, Any] = dict(_usage_metrics(usage))
        for key, value in metrics.items():
            self.aggregate_metrics[key] = self.aggregate_metrics.get(key, 0) + value
        metrics["time_to_first_token"] = self.turn_ttft
        _log_and_end_span(
            self.llm_span,
            output=self._llm_output(),
            metrics=metrics,
            metadata={"model": self.model, "provider": "cursor"} if self.model else None,
        )

        self.history.append(self._assistant_message(include_reasoning=True))
        self.llm_span = None
        self.llm_text = ""
        self.reasoning_text = ""
        self.llm_tool_calls = []
        self.turn_count += 1

    def start_tool(self, call_id: str, name: str, args: Any) -> _ToolState:
        existing = self.tools.get(call_id)
        if existing is not None:
            return existing
        self._start_llm(time.time())
        parent = self.llm_span.export() if self.llm_span is not None else self.root_export
        tool_name = name or "tool"
        # Not `set_current=False`: that fixes `can_set_current` off for the
        # span's lifetime, and `_wrap_tool_dispatch` makes this span current
        # later so a custom tool's own spans nest under it.
        span = start_span(
            name=tool_name,
            span_attributes={"type": SpanTypeAttribute.TOOL},
            input=args,
            parent=parent,
        )
        span.unset_current()
        state = _ToolState(span=span)
        self.tools[call_id] = state
        self.llm_tool_calls.append(_canonical_tool_call(call_id, tool_name, args))
        return state

    def _finish_tool(self, call_id: str, *, output: Any, is_error: bool) -> None:
        state = self.tools.get(call_id)
        if state is None:
            state = self.start_tool(call_id, "tool", None)
        state.finish(output=output, error=str(output) if is_error else None)
        self.history.append({"role": "tool", "tool_call_id": call_id, "content": output})

    def add_event(self, event: Any, run: Any) -> None:
        if self.finished:
            return
        offset = getattr(event, "offset", None)
        if offset is not None:
            event_key = (getattr(event, "kind", ""), offset)
            if event_key in self.seen_events:
                return
            self.seen_events.add(event_key)
        message = getattr(event, "sdk_message", None)
        message_type = _value(message, "type", "")
        if message_type == "thinking":
            self._start_llm(time.time())
            text = _value(message, "text", "")
            if isinstance(text, str):
                self.reasoning_text += text
        elif message_type == "assistant":
            self._start_llm(time.time())
            for block in _content_blocks(message):
                block_type = _value(block, "type")
                if block_type == "text":
                    text = _value(block, "text", "")
                    if isinstance(text, str):
                        self.llm_text += text
                elif block_type == "tool_use":
                    self.start_tool(
                        str(_value(block, "id", "")),
                        str(_value(block, "name", "tool")),
                        _value(block, "input"),
                    )
        elif message_type == "tool_call":
            call_id = str(_value(message, "call_id", ""))
            status = str(_value(message, "status", ""))
            if status == "running":
                self.start_tool(call_id, str(_value(message, "name", "tool")), _value(message, "args"))
            elif status in {"completed", "error"}:
                self._finish_tool(call_id, output=_value(message, "result"), is_error=status == "error")
        elif message_type == "usage":
            self._finish_llm(_value(message, "usage"))

        if run is not None:
            self._update_model(getattr(run, "model", None))
        if getattr(event, "result_is_full", False):
            result = getattr(event, "result", None)
            self._update_model(_value(result, "model"))
            self.finish(run, result=result)

    def finish(self, run: Any = None, *, result: Any = None, cancelled: bool = False) -> None:
        if self.finished:
            return
        self._finish_llm()
        for state in self.tools.values():
            if not state.completed:
                state.finish(error="Cursor run cancelled" if cancelled else None)

        target = run or self.run
        output = getattr(target, "result", None) or _value(result, "result")
        usage = getattr(target, "usage", None) or _value(result, "usage")
        metrics = _usage_metrics(usage) if usage is not None else dict(self.aggregate_metrics)
        event: dict[str, Any] = {}
        if output:
            event["output"] = output
        if metrics:
            event["metrics"] = metrics
        status = str(_value(result, "status") or getattr(target, "status", "")).lower()
        if cancelled or status.endswith(_ERROR_STATUS_SUFFIXES):
            event["error"] = f"Cursor run {('cancelled' if cancelled else status)}"
        if event:
            self.root_span.log(**event)
        self.root_span.end()
        self.finished = True
        self.agent_trackers.discard(self)

    def fail(self, error: BaseException) -> None:
        if self.finished:
            return
        self.root_span.log(error=error)
        self.finish()


def _tracker_for_run(run: Any) -> CursorRunTracker | None:
    tracker = getattr(run, "_braintrust_cursor_tracker", None)
    if tracker is None:
        tracker = _PENDING_TRACKER.get()
    return tracker


def _wrap_sync_send(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    message = args[0] if args else kwargs.get("message")
    tracker = CursorRunTracker(message=message, model=_send_model(instance, args, kwargs))
    token = _PENDING_TRACKER.set(tracker)
    try:
        run = wrapped(*args, **kwargs)
    except Exception as error:
        tracker.fail(error)
        raise
    finally:
        _PENDING_TRACKER.reset(token)
    tracker.attach(run, instance)
    return run


async def _wrap_async_send(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    message = args[0] if args else kwargs.get("message")
    tracker = CursorRunTracker(message=message, model=_send_model(instance, args, kwargs))
    token = _PENDING_TRACKER.set(tracker)
    try:
        run = await wrapped(*args, **kwargs)
    except Exception as error:
        tracker.fail(error)
        raise
    finally:
        _PENDING_TRACKER.reset(token)
    tracker.attach(run, instance)
    return run


def _run_lifecycle_hooks(
    post: Callable[[Any, Any, Any, Any], None],
    *,
    fail_on_error: bool = True,
) -> tuple[Any, Any]:
    """Build the ``(sync, async)`` wrapper pair for one Run lifecycle method.

    Cursor mirrors every ``Run`` method on ``AsyncRun``, so the two halves
    differ only by ``await``. Generating them together keeps the tracker
    protocol in one place instead of one copy per flavor.
    """

    def sync_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        tracker = _tracker_for_run(instance)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            if fail_on_error and tracker is not None:
                tracker.fail(error)
            raise
        if tracker is not None:
            post(tracker, instance, args, kwargs)
        return result

    async def async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        tracker = _tracker_for_run(instance)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as error:
            if fail_on_error and tracker is not None:
                tracker.fail(error)
            raise
        if tracker is not None:
            post(tracker, instance, args, kwargs)
        return result

    return sync_wrapper, async_wrapper


def _wrap_apply_event_state(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """Trace one stream event.

    `Run` and `AsyncRun` both funnel every event through this one synchronous
    base method -- it is where all the state the tracker reads is applied -- so
    a single wrapper covers both flavors and every consumption path.
    """
    tracker = _tracker_for_run(instance)
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        if tracker is not None:
            tracker.fail(error)
        raise
    if tracker is not None:
        tracker.add_event(args[0] if args else kwargs.get("event"), instance)
    return result


_wrap_sync_wait, _wrap_async_wait = _run_lifecycle_hooks(
    lambda tracker, instance, args, kwargs: tracker.finish(instance)
)
# `cancel()` failing is not a run failure, so it does not fail the trace.
_wrap_sync_cancel, _wrap_async_cancel = _run_lifecycle_hooks(
    lambda tracker, instance, args, kwargs: tracker.finish(instance, cancelled=True),
    fail_on_error=False,
)


def _wrap_sync_observe(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    stream = wrapped(*args, **kwargs)
    tracker = _tracker_for_run(instance)

    def traced_stream():
        try:
            for event in stream:
                if tracker is not None:
                    tracker.add_event(event, instance)
                yield event
        except Exception as error:
            if tracker is not None:
                tracker.fail(error)
            raise

    return traced_stream()


def _wrap_async_observe(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    stream = wrapped(*args, **kwargs)
    tracker = _tracker_for_run(instance)

    async def traced_stream():
        try:
            async for event in stream:
                if tracker is not None:
                    tracker.add_event(event, instance)
                yield event
        except Exception as error:
            if tracker is not None:
                tracker.fail(error)
            raise

    return traced_stream()


def _finish_agent_trackers(instance: Any) -> None:
    for tracker in list(getattr(instance, _AGENT_TRACKERS_ATTR, ())):
        status = str(getattr(tracker.run, "status", "")).lower()
        # A run still in flight when its agent closes is abandoned, not done.
        tracker.finish(
            tracker.run,
            cancelled=not status.endswith(("finished", *_ERROR_STATUS_SUFFIXES)),
        )


def _wrap_sync_close(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    try:
        return wrapped(*args, **kwargs)
    finally:
        _finish_agent_trackers(instance)


async def _wrap_async_close(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _finish_agent_trackers(instance)


def _wrap_tool_dispatch(wrapped: Any, _instance: Any, args: Any, kwargs: Any) -> Any:
    request = args[2] if len(args) > 2 else kwargs.get("request", {})
    agent_id = str(_value(request, "agentId", _value(request, "agent_id", "")))
    call_id = str(_value(request, "toolCallId", _value(request, "tool_call_id", "")))
    name = str(_value(request, "toolName", _value(request, "tool_name", "tool")))
    tool_args = _value(request, "args", {})
    agent = _AGENTS_BY_ID.get(agent_id)
    trackers = getattr(agent, _AGENT_TRACKERS_ATTR, ()) if agent is not None else ()
    tracker = next(iter(trackers), None)
    if tracker is None:
        return wrapped(*args, **kwargs)
    # `start_tool` returns the existing state when the run already opened a
    # span for this call, so it doubles as the lookup.
    state = tracker.start_tool(call_id, name, tool_args)
    state.span.set_current()
    try:
        return wrapped(*args, **kwargs)
    finally:
        state.span.unset_current()
