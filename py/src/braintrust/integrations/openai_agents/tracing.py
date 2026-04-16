"""OpenAI Agents SDK tracing processor."""

import datetime
from typing import Any

from agents import tracing
from braintrust.logger import NOOP_SPAN, Experiment, Logger, Span, current_span, flush, start_span
from braintrust.span_types import SpanTypeAttribute


# TaskSpanData and TurnSpanData were added in openai-agents 0.14.0.
_TaskSpanData = getattr(tracing, "TaskSpanData", None)
_TurnSpanData = getattr(tracing, "TurnSpanData", None)


def _span_type(span: tracing.Span[Any]) -> SpanTypeAttribute:
    if span.span_data.type in ["agent", "handoff", "custom", "speech_group", "task", "turn"]:
        return SpanTypeAttribute.TASK
    elif span.span_data.type in ["function", "guardrail", "mcp_tools"]:
        return SpanTypeAttribute.TOOL
    elif span.span_data.type in ["generation", "response", "transcription", "speech"]:
        return SpanTypeAttribute.LLM
    else:
        return SpanTypeAttribute.TASK


def _span_name(span: tracing.Span[Any]) -> str:
    # TODO(sachin): span name should also come from the span_data.
    if (
        isinstance(span.span_data, tracing.AgentSpanData)
        or isinstance(span.span_data, tracing.FunctionSpanData)
        or isinstance(span.span_data, tracing.GuardrailSpanData)
        or isinstance(span.span_data, tracing.CustomSpanData)
    ):
        return span.span_data.name
    elif isinstance(span.span_data, tracing.GenerationSpanData):
        return "Generation"
    elif isinstance(span.span_data, tracing.ResponseSpanData):
        return "Response"
    elif isinstance(span.span_data, tracing.HandoffSpanData):
        return "Handoff"
    elif isinstance(span.span_data, tracing.MCPListToolsSpanData):
        if span.span_data.server:
            return f"List Tools ({span.span_data.server})"
        return "MCP List Tools"
    elif isinstance(span.span_data, tracing.TranscriptionSpanData):
        return "Transcription"
    elif isinstance(span.span_data, tracing.SpeechSpanData):
        return "Speech"
    elif isinstance(span.span_data, tracing.SpeechGroupSpanData):
        return "Speech Group"
    elif _TaskSpanData is not None and isinstance(span.span_data, _TaskSpanData):
        return span.span_data.name
    elif _TurnSpanData is not None and isinstance(span.span_data, _TurnSpanData):
        return f"Turn {span.span_data.turn} ({span.span_data.agent_name})"
    else:
        return "Unknown"


def _timestamp_from_maybe_iso(timestamp: str | None) -> float | None:
    if timestamp is None:
        return None
    return datetime.datetime.fromisoformat(timestamp).timestamp()


def _maybe_timestamp_elapsed(end: str | None, start: str | None) -> float | None:
    if start is None or end is None:
        return None
    return (datetime.datetime.fromisoformat(end) - datetime.datetime.fromisoformat(start)).total_seconds()


def _usage_to_metrics(usage: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style usage dict to Braintrust metrics."""
    metrics: dict[str, Any] = {}
    if "prompt_tokens" in usage:
        metrics["prompt_tokens"] = usage["prompt_tokens"]
    elif "input_tokens" in usage:
        metrics["prompt_tokens"] = usage["input_tokens"]

    if "completion_tokens" in usage:
        metrics["completion_tokens"] = usage["completion_tokens"]
    elif "output_tokens" in usage:
        metrics["completion_tokens"] = usage["output_tokens"]

    if "total_tokens" in usage:
        metrics["tokens"] = usage["total_tokens"]
    elif "input_tokens" in usage and "output_tokens" in usage:
        metrics["tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return metrics


class BraintrustTracingProcessor(tracing.TracingProcessor):
    """Tracing processor that logs OpenAI Agents SDK traces to Braintrust."""

    def __init__(self, logger: Span | Experiment | Logger | None = None):
        self._logger = logger
        self._spans: dict[str, Span] = {}
        self._first_input: dict[str, Any] = {}
        self._last_output: dict[str, Any] = {}

    def on_trace_start(self, trace: tracing.Trace) -> None:
        trace_meta = trace.export() or {}
        metadata = {
            "group_id": trace_meta.get("group_id"),
            **(trace_meta.get("metadata") or {}),
        }

        current_context = current_span()
        if current_context != NOOP_SPAN:
            span = current_context.start_span(
                name=trace.name,
                span_attributes={"type": "task", "name": trace.name},
                metadata=metadata,
            )
        elif self._logger is not None:
            span = self._logger.start_span(
                span_attributes={"type": "task", "name": trace.name},
                span_id=trace.trace_id,
                root_span_id=trace.trace_id,
                metadata=metadata,
            )
        else:
            span = start_span(
                id=trace.trace_id,
                span_attributes={"type": "task", "name": trace.name},
                metadata=metadata,
            )
        if span != NOOP_SPAN:
            span.set_current()
        self._spans[trace.trace_id] = span

    def on_trace_end(self, trace: tracing.Trace) -> None:
        span = self._spans.pop(trace.trace_id)
        trace_first_input = self._first_input.pop(trace.trace_id, None)
        trace_last_output = self._last_output.pop(trace.trace_id, None)
        span.log(input=trace_first_input, output=trace_last_output)
        span.end()
        span.unset_current()

    def _agent_log_data(self, span: tracing.Span[tracing.AgentSpanData]) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "tools": span.span_data.tools,
            "handoffs": span.span_data.handoffs,
            "output_type": span.span_data.output_type,
        }
        # AgentSpanData gained a metadata slot in openai-agents 0.14.0.
        agent_metadata = getattr(span.span_data, "metadata", None)
        if agent_metadata:
            metadata.update(agent_metadata)
        return {"metadata": metadata}

    def _response_log_data(self, span: tracing.Span[tracing.ResponseSpanData]) -> dict[str, Any]:
        data = {}
        if span.span_data.input is not None:
            data["input"] = span.span_data.input
        if span.span_data.response is not None:
            data["output"] = span.span_data.response.output
        if span.span_data.response is not None:
            data["metadata"] = span.span_data.response.metadata or {}
            data["metadata"].update(
                span.span_data.response.model_dump(exclude={"input", "output", "metadata", "usage"})
            )

        data["metrics"] = {}
        ttft = _maybe_timestamp_elapsed(span.ended_at, span.started_at)
        if ttft is not None:
            data["metrics"]["time_to_first_token"] = ttft
        if span.span_data.response is not None and span.span_data.response.usage is not None:
            data["metrics"]["tokens"] = span.span_data.response.usage.total_tokens
            data["metrics"]["prompt_tokens"] = span.span_data.response.usage.input_tokens
            data["metrics"]["completion_tokens"] = span.span_data.response.usage.output_tokens

        return data

    def _function_log_data(self, span: tracing.Span[tracing.FunctionSpanData]) -> dict[str, Any]:
        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
        }

    def _handoff_log_data(self, span: tracing.Span[tracing.HandoffSpanData]) -> dict[str, Any]:
        return {
            "metadata": {
                "from_agent": span.span_data.from_agent,
                "to_agent": span.span_data.to_agent,
            }
        }

    def _guardrail_log_data(self, span: tracing.Span[tracing.GuardrailSpanData]) -> dict[str, Any]:
        return {
            "metadata": {
                "triggered": span.span_data.triggered,
            }
        }

    def _generation_log_data(self, span: tracing.Span[tracing.GenerationSpanData]) -> dict[str, Any]:
        metrics = _usage_to_metrics(span.span_data.usage or {})
        ttft = _maybe_timestamp_elapsed(span.ended_at, span.started_at)
        if ttft is not None:
            metrics["time_to_first_token"] = ttft

        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
            "metadata": {
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
            "metrics": metrics,
        }

    def _custom_log_data(self, span: tracing.Span[tracing.CustomSpanData]) -> dict[str, Any]:
        return span.span_data.data

    def _mcp_list_tools_log_data(self, span: tracing.Span[tracing.MCPListToolsSpanData]) -> dict[str, Any]:
        return {
            "output": span.span_data.result,
            "metadata": {
                "server": span.span_data.server,
            },
        }

    def _transcription_log_data(self, span: tracing.Span[tracing.TranscriptionSpanData]) -> dict[str, Any]:
        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
            "metadata": {
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
        }

    def _speech_log_data(self, span: tracing.Span[tracing.SpeechSpanData]) -> dict[str, Any]:
        return {
            "input": span.span_data.input,
            "output": span.span_data.output,
            "metadata": {
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
        }

    def _speech_group_log_data(self, span: tracing.Span[tracing.SpeechGroupSpanData]) -> dict[str, Any]:
        return {
            "input": span.span_data.input,
        }

    def _task_log_data(self, span: "tracing.Span[Any]") -> dict[str, Any]:
        """Handle TaskSpanData (openai-agents >= 0.14.0)."""
        data: dict[str, Any] = {}
        metadata = getattr(span.span_data, "metadata", None)
        if metadata:
            data["metadata"] = metadata
        usage = getattr(span.span_data, "usage", None)
        if usage:
            metrics = _usage_to_metrics(usage)
            if metrics:
                data["metrics"] = metrics
        return data

    def _turn_log_data(self, span: "tracing.Span[Any]") -> dict[str, Any]:
        """Handle TurnSpanData (openai-agents >= 0.14.0)."""
        data: dict[str, Any] = {
            "metadata": {
                "turn": span.span_data.turn,
                "agent_name": span.span_data.agent_name,
            }
        }
        turn_metadata = getattr(span.span_data, "metadata", None)
        if turn_metadata:
            data["metadata"].update(turn_metadata)
        usage = getattr(span.span_data, "usage", None)
        if usage:
            metrics = _usage_to_metrics(usage)
            if metrics:
                data["metrics"] = metrics
        return data

    def _log_data(self, span: tracing.Span[Any]) -> dict[str, Any]:
        if isinstance(span.span_data, tracing.AgentSpanData):
            return self._agent_log_data(span)
        elif isinstance(span.span_data, tracing.ResponseSpanData):
            return self._response_log_data(span)
        elif isinstance(span.span_data, tracing.FunctionSpanData):
            return self._function_log_data(span)
        elif isinstance(span.span_data, tracing.HandoffSpanData):
            return self._handoff_log_data(span)
        elif isinstance(span.span_data, tracing.GuardrailSpanData):
            return self._guardrail_log_data(span)
        elif isinstance(span.span_data, tracing.GenerationSpanData):
            return self._generation_log_data(span)
        elif isinstance(span.span_data, tracing.CustomSpanData):
            return self._custom_log_data(span)
        elif isinstance(span.span_data, tracing.MCPListToolsSpanData):
            return self._mcp_list_tools_log_data(span)
        elif isinstance(span.span_data, tracing.TranscriptionSpanData):
            return self._transcription_log_data(span)
        elif isinstance(span.span_data, tracing.SpeechSpanData):
            return self._speech_log_data(span)
        elif isinstance(span.span_data, tracing.SpeechGroupSpanData):
            return self._speech_group_log_data(span)
        elif _TaskSpanData is not None and isinstance(span.span_data, _TaskSpanData):
            return self._task_log_data(span)
        elif _TurnSpanData is not None and isinstance(span.span_data, _TurnSpanData):
            return self._turn_log_data(span)
        else:
            return {}

    def on_span_start(self, span: tracing.Span[tracing.SpanData]) -> None:
        if span.parent_id is not None:
            parent = self._spans[span.parent_id]
        else:
            parent = self._spans[span.trace_id]
        created_span = parent.start_span(
            id=span.span_id,
            name=_span_name(span),
            type=_span_type(span),
            start_time=_timestamp_from_maybe_iso(span.started_at),
        )
        self._spans[span.span_id] = created_span
        created_span.set_current()

    def on_span_end(self, span: tracing.Span[tracing.SpanData]) -> None:
        s = self._spans.pop(span.span_id)
        event = dict(error=span.error, **self._log_data(span))
        s.log(**event)
        s.unset_current()
        s.end(_timestamp_from_maybe_iso(span.ended_at))

        input_ = event.get("input")
        output = event.get("output")
        trace_id = span.trace_id
        if trace_id not in self._first_input and input_ is not None:
            self._first_input[trace_id] = input_

        if output is not None:
            self._last_output[trace_id] = output

    def shutdown(self) -> None:
        if self._logger is not None:
            self._logger.flush()
        else:
            flush()

    def force_flush(self) -> None:
        if self._logger is not None:
            self._logger.flush()
        else:
            flush()


# ---------------------------------------------------------------------------
# Setup helpers — used by OpenAIAgentsTracingPatcher in patchers.py
# ---------------------------------------------------------------------------


def _get_trace_provider():
    return tracing.get_trace_provider()


def _get_processors():
    provider = _get_trace_provider()
    return getattr(getattr(provider, "_multi_processor", None), "_processors", ())


def _has_braintrust_tracing_processor() -> bool:
    provider = _get_trace_provider()
    has_processor = any(isinstance(p, BraintrustTracingProcessor) for p in _get_processors())
    return has_processor and not getattr(provider, "_disabled", False)


def _setup_openai_agents_tracing() -> None:
    import agents as agents_mod

    if not any(isinstance(p, BraintrustTracingProcessor) for p in _get_processors()):
        agents_mod.add_trace_processor(BraintrustTracingProcessor())

    agents_mod.set_tracing_disabled(False)
