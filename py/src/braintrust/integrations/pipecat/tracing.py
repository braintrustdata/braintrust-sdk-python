"""Observer-based tracing for Pipecat pipelines."""

import json
from typing import Any

from braintrust.integrations.utils import (
    _is_not_given,
    _normalize_chat_messages,
    _pcm_to_wav,
    _resolve_audio_attachment_options,
)
from braintrust.logger import NOOP_SPAN, Attachment, SpanTypeAttribute, current_span, start_span


try:
    from pipecat.observers.base_observer import BaseObserver
except ImportError:  # pragma: no cover - exercised when Pipecat is not installed.

    class BaseObserver:  # type: ignore[no-redef]
        """Fallback base so this module is importable without Pipecat."""

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def cleanup(self) -> None:
            pass


_TERMINAL_FRAME_TYPES = {"EndFrame", "StopFrame", "CancelFrame"}
_SPEC_METRIC_NAMES = {
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "time_to_first_token",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
}
_ALLOWED_LLM_METADATA_FIELDS = {
    "model",
    "provider",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "tool_choice",
}


class BraintrustPipecatObserver(BaseObserver):
    """Pipecat observer that emits Braintrust task, llm, and tool spans."""

    def __init__(
        self,
        *,
        capture_audio_attachments: bool | None = None,
        capture_user_audio_attachments: bool | None = None,
        capture_agent_audio_attachments: bool | None = None,
        trace_turns: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        (
            self.capture_user_audio_attachments,
            self.capture_agent_audio_attachments,
        ) = _resolve_audio_attachment_options(
            capture_audio_attachments=capture_audio_attachments,
            capture_user_audio_attachments=capture_user_audio_attachments,
            capture_agent_audio_attachments=capture_agent_audio_attachments,
        )
        self.capture_audio_attachments = self.capture_user_audio_attachments or self.capture_agent_audio_attachments
        self.trace_turns = trace_turns
        self._parent = _current_parent_export()
        self._pipeline_span: Any | None = None
        self._pipeline_parent: str | None = None
        self._seen_frame_ids: set[int] = set()
        self._latest_llm_input: Any = None
        self._latest_llm_metadata: dict[str, Any] = {}
        self._llm_span: Any | None = None
        self._llm_parent: str | None = None
        self._llm_text_parts: list[str] = []
        self._llm_metrics: dict[str, Any] = {}
        self._llm_metadata: dict[str, Any] = {}
        self._llm_tool_calls: list[dict[str, Any]] = []
        self._tool_call_parents: dict[str, str] = {}
        self._tool_spans: dict[str, Any] = {}
        self._tts_spans: dict[str, Any] = {}
        self._tts_default_span: Any | None = None
        self._tts_audio: dict[str, bytearray] = {}
        self._tts_audio_metadata: dict[str, dict[str, Any]] = {}
        self._user_audio_span: Any | None = None
        self._user_audio: bytearray | None = None
        self._user_audio_metadata: dict[str, Any] = {}

    async def on_pipeline_started(self) -> None:
        self._ensure_pipeline_span()

    async def on_process_frame(self, data: Any) -> None:
        await self._handle_frame(getattr(data, "frame", None), processor=getattr(data, "processor", None))

    async def on_push_frame(self, data: Any) -> None:
        await self._handle_frame(getattr(data, "frame", None), processor=getattr(data, "source", None))

    async def cleanup(self) -> None:
        self._close_all_open_spans()
        await super().cleanup()

    async def _handle_frame(self, frame: Any, *, processor: Any = None) -> None:
        if frame is None:
            return
        frame_type = type(frame).__name__
        if frame_type in _TERMINAL_FRAME_TYPES and not _is_pipeline_sink_processor(processor):
            return

        frame_id = getattr(frame, "id", None)
        if isinstance(frame_id, int):
            if frame_id in self._seen_frame_ids:
                return
            self._seen_frame_ids.add(frame_id)
        if frame_type == "StartFrame":
            self._ensure_pipeline_span(frame=frame, processor=processor)
        elif frame_type == "LLMContextFrame":
            self._capture_llm_context(frame, processor)
        elif frame_type == "LLMFullResponseStartFrame":
            self._start_llm_span(processor)
        elif frame_type == "LLMTextFrame":
            if self._llm_span is not None:
                self._llm_text_parts.append(getattr(frame, "text", ""))
        elif frame_type == "FunctionCallsStartedFrame":
            self._capture_llm_tool_calls(frame)
        elif frame_type == "LLMFullResponseEndFrame":
            self._end_llm_span()
        elif frame_type == "FunctionCallInProgressFrame":
            self._start_tool_span(frame)
        elif frame_type == "FunctionCallResultFrame":
            self._end_tool_span(frame)
        elif frame_type == "FunctionCallCancelFrame":
            self._cancel_tool_span(frame)
        elif frame_type in {"UserStartedSpeakingFrame", "VADUserStartedSpeakingFrame"}:
            self._start_user_audio_span(frame)
        elif frame_type in {"InputAudioRawFrame", "UserAudioRawFrame"}:
            self._capture_user_audio(frame)
        elif frame_type in {"UserStoppedSpeakingFrame", "VADUserStoppedSpeakingFrame"}:
            self._end_user_audio_span(frame)
        elif frame_type == "TranscriptionFrame":
            self._log_transcription(frame)
        elif frame_type == "TTSStartedFrame":
            self._start_tts_span(frame)
        elif frame_type == "TTSTextFrame":
            self._append_tts_text(frame)
        elif frame_type == "TTSAudioRawFrame":
            self._log_tts_audio(frame)
        elif frame_type == "TTSStoppedFrame":
            self._end_tts_span(frame)
        elif frame_type == "MetricsFrame":
            self._capture_metrics(frame, processor)
        elif frame_type == "ErrorFrame":
            self._log_error_frame(frame)
        elif frame_type in _TERMINAL_FRAME_TYPES:
            self._end_pipeline_span(frame)

    def _ensure_pipeline_span(self, *, frame: Any | None = None, processor: Any = None) -> None:
        if self._pipeline_span is not None:
            return
        input_payload: dict[str, Any] = {}
        metadata: dict[str, Any] = {"framework": "pipecat"}
        if frame is not None:
            input_payload.update(
                {
                    "audio_in_sample_rate": getattr(frame, "audio_in_sample_rate", None),
                    "audio_out_sample_rate": getattr(frame, "audio_out_sample_rate", None),
                    "enable_metrics": getattr(frame, "enable_metrics", None),
                    "enable_usage_metrics": getattr(frame, "enable_usage_metrics", None),
                }
            )
            metadata.update(getattr(frame, "metadata", {}) or {})
        processor_name = _processor_name(processor)
        if processor_name:
            metadata["processor"] = processor_name
        input_payload = {k: v for k, v in input_payload.items() if v is not None}
        self._pipeline_span = start_span(
            name="pipecat_pipeline",
            type=SpanTypeAttribute.TASK,
            input=input_payload or None,
            parent=self._parent,
            set_current=False,
        )
        self._pipeline_parent = self._pipeline_span.export()
        if metadata:
            self._pipeline_span.log(metadata=metadata)

    def _capture_llm_context(self, frame: Any, processor: Any) -> None:
        self._ensure_pipeline_span(processor=processor)
        context = getattr(frame, "context", None)
        messages = _messages_from_context(context)
        self._latest_llm_input = _normalize_chat_messages(messages)
        metadata = _metadata_from_context(context)
        metadata.update(_metadata_from_processor(processor))
        self._latest_llm_metadata = _filter_llm_metadata(metadata)

    def _start_llm_span(self, processor: Any) -> None:
        self._ensure_pipeline_span(processor=processor)
        if self._llm_span is not None:
            return
        metadata = {**self._latest_llm_metadata, **_metadata_from_processor(processor)}
        self._llm_metadata = _filter_llm_metadata(metadata)
        self._llm_text_parts = []
        self._llm_metrics = {}
        self._llm_tool_calls = []
        self._llm_span = start_span(
            name="pipecat_llm_response",
            type=SpanTypeAttribute.TASK,
            input=self._latest_llm_input,
            metadata=self._llm_metadata or None,
            parent=self._pipeline_parent,
            set_current=False,
        )
        self._llm_parent = self._llm_span.export()

    def _capture_llm_tool_calls(self, frame: Any) -> None:
        calls = getattr(frame, "function_calls", None) or []
        for call in calls:
            tool_call = _tool_call_from_pipecat_call(call)
            if tool_call:
                self._llm_tool_calls.append(tool_call)
                tool_id = tool_call.get("id")
                if isinstance(tool_id, str) and self._llm_parent:
                    self._tool_call_parents[tool_id] = self._llm_parent

    def _end_llm_span(self) -> None:
        if self._llm_span is None:
            return
        content = "".join(part for part in self._llm_text_parts if isinstance(part, str))
        message: dict[str, Any] = {"role": "assistant", "content": None if self._llm_tool_calls else content}
        if self._llm_tool_calls:
            if content:
                message["content"] = content
            message["tool_calls"] = self._llm_tool_calls
        output = [
            {
                "index": 0,
                "finish_reason": "tool_calls" if self._llm_tool_calls else "stop",
                "message": message,
            }
        ]
        event: dict[str, Any] = {"output": output}
        if self._llm_metrics:
            event["metrics"] = {k: v for k, v in self._llm_metrics.items() if k in _SPEC_METRIC_NAMES}
        metadata = _filter_llm_metadata(self._llm_metadata)
        if metadata:
            event["metadata"] = metadata
        self._llm_span.log(**event)
        self._llm_span.end()
        self._llm_span = None
        self._llm_parent = None
        self._llm_text_parts = []
        self._llm_metrics = {}
        self._llm_metadata = {}
        self._llm_tool_calls = []

    def _start_tool_span(self, frame: Any) -> None:
        self._ensure_pipeline_span()
        tool_call_id = getattr(frame, "tool_call_id", None)
        parent = self._tool_call_parents.get(tool_call_id) or self._llm_parent or self._pipeline_parent
        name = getattr(frame, "function_name", None) or "pipecat_tool"
        span = start_span(
            name=name,
            type=SpanTypeAttribute.TOOL,
            input=_parse_jsonish(getattr(frame, "arguments", None)),
            parent=parent,
            set_current=False,
        )
        if isinstance(tool_call_id, str):
            self._tool_spans[tool_call_id] = span
            self._tool_call_parents.setdefault(tool_call_id, span.export())

    def _end_tool_span(self, frame: Any) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        span = self._tool_spans.pop(tool_call_id, None) if isinstance(tool_call_id, str) else None
        if span is None:
            self._start_tool_span(frame)
            span = self._tool_spans.pop(tool_call_id, None) if isinstance(tool_call_id, str) else None
        if span is None:
            return
        span.log(output=getattr(frame, "result", None), metadata={"tool_call_id": tool_call_id})
        span.end()

    def _cancel_tool_span(self, frame: Any) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        span = self._tool_spans.pop(tool_call_id, None) if isinstance(tool_call_id, str) else None
        if span is not None:
            span.log(error=f"Tool call {tool_call_id} was cancelled")
            span.end()

    def _log_transcription(self, frame: Any) -> None:
        self._ensure_pipeline_span()
        metadata = {
            "user_id": getattr(frame, "user_id", None),
            "timestamp": getattr(frame, "timestamp", None),
            "language": _enum_value(getattr(frame, "language", None)),
            "finalized": getattr(frame, "finalized", None),
        }
        span = start_span(
            name="stt_transcription",
            type=SpanTypeAttribute.TASK,
            parent=self._pipeline_parent,
            set_current=False,
        )
        span.log(
            output={"text": getattr(frame, "text", None), "result": getattr(frame, "result", None)},
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
        span.end()

    def _start_tts_span(self, frame: Any) -> None:
        self._ensure_pipeline_span()
        context_id = getattr(frame, "context_id", None) or "__default__"
        span = start_span(
            name="tts_response",
            type=SpanTypeAttribute.TASK,
            parent=self._pipeline_parent,
            set_current=False,
        )
        metadata = {
            "context_id": getattr(frame, "context_id", None),
            "append_to_context": getattr(frame, "append_to_context", None),
        }
        span.log(metadata={k: v for k, v in metadata.items() if v is not None})
        self._tts_spans[context_id] = span
        if context_id == "__default__":
            self._tts_default_span = span

    def _append_tts_text(self, frame: Any) -> None:
        span = self._tts_span_for_frame(frame)
        if span is not None:
            span.log(input={"text": getattr(frame, "text", None)})

    def _log_tts_audio(self, frame: Any) -> None:
        span = self._tts_span_for_frame(frame)
        if span is None and self.capture_agent_audio_attachments:
            self._start_tts_span(frame)
            span = self._tts_span_for_frame(frame)
        if span is None:
            return
        audio = _audio_bytes(frame)
        metadata = _audio_frame_metadata(frame)
        if audio is not None:
            metadata["audio_size_bytes"] = len(audio)
            if self.capture_agent_audio_attachments:
                context_id = getattr(frame, "context_id", None) or "__default__"
                self._append_audio_chunk(
                    self._tts_audio,
                    self._tts_audio_metadata,
                    context_id,
                    audio,
                    metadata,
                )
        span.log(metadata={k: v for k, v in metadata.items() if v is not None})

    def _end_tts_span(self, frame: Any) -> None:
        context_id = getattr(frame, "context_id", None) or "__default__"
        span = self._tts_spans.pop(context_id, None)
        if span is not None:
            output = self._pop_tts_audio_output(context_id)
            if output:
                span.log(output=output)
            span.end()
        else:
            self._discard_tts_audio(context_id)
        if context_id == "__default__":
            self._tts_default_span = None

    def _tts_span_for_frame(self, frame: Any) -> Any | None:
        context_id = getattr(frame, "context_id", None) or "__default__"
        return self._tts_spans.get(context_id) or self._tts_default_span

    def _start_user_audio_span(self, frame: Any | None = None) -> None:
        if not self.capture_user_audio_attachments:
            return
        self._ensure_pipeline_span()
        if self._user_audio_span is not None:
            return
        self._user_audio = bytearray()
        self._user_audio_metadata = _audio_frame_metadata(frame) if frame is not None else {}
        self._user_audio_span = start_span(
            name="user_speaking",
            type=SpanTypeAttribute.TASK,
            parent=self._pipeline_parent,
            set_current=False,
        )

    def _capture_user_audio(self, frame: Any) -> None:
        if not self.capture_user_audio_attachments:
            return
        audio = _audio_bytes(frame)
        if audio is None:
            return
        self._ensure_pipeline_span()
        if self._user_audio_span is None:
            self._start_user_audio_span()
        if self._user_audio is None:
            self._user_audio = bytearray()
        self._user_audio.extend(audio)
        _merge_audio_metadata(self._user_audio_metadata, _audio_frame_metadata(frame))

    def _end_user_audio_span(self, frame: Any | None = None) -> None:
        if self._user_audio_span is None:
            return
        if frame is not None:
            _merge_audio_metadata(self._user_audio_metadata, _audio_frame_metadata(frame))
        output = _audio_output(
            self._user_audio,
            self._user_audio_metadata,
            filename_prefix="user_speaking",
        )
        if output:
            self._user_audio_span.log(input=output)
        self._user_audio_span.end()
        self._user_audio_span = None
        self._user_audio = None
        self._user_audio_metadata = {}

    def _append_audio_chunk(
        self,
        audio_by_context: dict[str, bytearray],
        metadata_by_context: dict[str, dict[str, Any]],
        context_id: str,
        audio: bytes,
        metadata: dict[str, Any],
    ) -> None:
        audio_by_context.setdefault(context_id, bytearray()).extend(audio)
        _merge_audio_metadata(metadata_by_context.setdefault(context_id, {}), metadata)

    def _pop_tts_audio_output(self, context_id: str) -> dict[str, Any]:
        audio = self._tts_audio.pop(context_id, None)
        metadata = self._tts_audio_metadata.pop(context_id, {})
        return _audio_output(audio, metadata, filename_prefix="tts_response")

    def _discard_tts_audio(self, context_id: str) -> None:
        self._tts_audio.pop(context_id, None)
        self._tts_audio_metadata.pop(context_id, None)

    def _capture_metrics(self, frame: Any, processor: Any) -> None:
        for metric in getattr(frame, "data", []) or []:
            metric_type = type(metric).__name__
            if metric_type == "LLMUsageMetricsData":
                self._llm_metrics.update(_llm_usage_metrics(getattr(metric, "value", None)))
                self._llm_metadata.update(_metadata_from_metric(metric))
            elif metric_type == "TTFBMetricsData" and self._llm_span is not None:
                value = getattr(metric, "value", None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._llm_metrics["time_to_first_token"] = value
                self._llm_metadata.update(_metadata_from_metric(metric))
        self._llm_metadata.update(_metadata_from_processor(processor))
        self._llm_metadata = _filter_llm_metadata(self._llm_metadata)

    def _log_error_frame(self, frame: Any) -> None:
        self._ensure_pipeline_span(processor=getattr(frame, "processor", None))
        error = getattr(frame, "exception", None) or getattr(frame, "error", None)
        if self._llm_span is not None:
            self._llm_span.log(error=error)
            self._llm_span.end()
            self._llm_span = None
        if self._pipeline_span is not None:
            self._pipeline_span.log(error=error, metadata={"fatal": getattr(frame, "fatal", None)})

    def _end_pipeline_span(self, frame: Any) -> None:
        if self._pipeline_span is None:
            return
        metadata = {"terminal_frame": type(frame).__name__}
        reason = getattr(frame, "reason", None)
        if reason is not None:
            metadata["reason"] = reason
        self._close_child_spans()
        self._pipeline_span.log(metadata=metadata)
        self._pipeline_span.end()
        self._pipeline_span = None
        self._pipeline_parent = None

    def _close_child_spans(self) -> None:
        if self._llm_span is not None:
            self._end_llm_span()
        for span in list(self._tool_spans.values()):
            span.end()
        self._tool_spans.clear()
        self._end_user_audio_span()
        for context_id, span in list(self._tts_spans.items()):
            output = self._pop_tts_audio_output(context_id)
            if output:
                span.log(output=output)
            span.end()
        self._tts_spans.clear()
        self._tts_audio.clear()
        self._tts_audio_metadata.clear()
        self._tts_default_span = None

    def _close_all_open_spans(self) -> None:
        self._close_child_spans()
        if self._pipeline_span is not None:
            self._pipeline_span.end()
            self._pipeline_span = None
            self._pipeline_parent = None


def _audio_bytes(frame: Any) -> bytes | None:
    audio = getattr(frame, "audio", None)
    if audio is None:
        return None
    try:
        data = bytes(audio)
    except Exception:
        return None
    return data if data else None


def _audio_frame_metadata(frame: Any) -> dict[str, Any]:
    return {
        "sample_rate": getattr(frame, "sample_rate", None),
        "num_channels": getattr(frame, "num_channels", None),
        "num_frames": getattr(frame, "num_frames", None),
        "context_id": getattr(frame, "context_id", None),
        "user_id": getattr(frame, "user_id", None),
        "transport_source": getattr(frame, "transport_source", None),
        "transport_destination": getattr(frame, "transport_destination", None),
    }


def _merge_audio_metadata(target: dict[str, Any], metadata: dict[str, Any]) -> None:
    existing_num_frames = target.get("num_frames")
    chunk_num_frames = metadata.get("num_frames")
    for key, value in metadata.items():
        if value is not None and key not in {"audio_size_bytes", "num_frames"}:
            target[key] = value
    if isinstance(chunk_num_frames, int) and not isinstance(chunk_num_frames, bool):
        if isinstance(existing_num_frames, int) and not isinstance(existing_num_frames, bool):
            target["num_frames"] = existing_num_frames + chunk_num_frames
        else:
            target["num_frames"] = chunk_num_frames


def _audio_output(
    audio: bytearray | bytes | None,
    metadata: dict[str, Any],
    *,
    filename_prefix: str,
) -> dict[str, Any]:
    if not audio:
        return {}
    audio_bytes = bytes(audio)
    sample_rate = metadata.get("sample_rate")
    num_channels = metadata.get("num_channels")
    suffix = f"_{sample_rate}hz_{num_channels}ch" if sample_rate and num_channels else ""
    if isinstance(sample_rate, int) and isinstance(num_channels, int) and sample_rate > 0 and num_channels > 0:
        attachment = Attachment(
            data=_pcm_to_wav(audio_bytes, sample_rate=sample_rate, num_channels=num_channels),
            filename=f"{filename_prefix}{suffix}.wav",
            content_type="audio/wav",
        )
    else:
        attachment = Attachment(
            data=audio_bytes,
            filename=f"{filename_prefix}{suffix}.pcm",
            content_type="audio/pcm",
        )
    return {
        **{k: v for k, v in metadata.items() if v is not None and k != "audio_size_bytes"},
        "audio": attachment,
        "audio_size_bytes": len(audio_bytes),
    }


def _current_parent_export() -> str | None:
    span = current_span()
    if span == NOOP_SPAN:
        return None
    try:
        return span.export()
    except Exception:
        return None


def _messages_from_context(context: Any) -> Any:
    get_messages = getattr(context, "get_messages", None)
    if callable(get_messages):
        try:
            return get_messages()
        except Exception:
            return getattr(context, "messages", None)
    return getattr(context, "messages", None)


def _metadata_from_context(context: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    tools = getattr(context, "tools", None)
    if tools is not None and not _is_not_given(tools):
        converted = _tools_schema_to_openai_tools(tools)
        if converted:
            metadata["tools"] = converted
    tool_choice = getattr(context, "tool_choice", None)
    if tool_choice is not None and not _is_not_given(tool_choice):
        metadata["tool_choice"] = tool_choice
    return metadata


def _tools_schema_to_openai_tools(tools: Any) -> list[dict[str, Any]]:
    standard_tools = getattr(tools, "standard_tools", None)
    if standard_tools is None and isinstance(tools, list):
        standard_tools = tools
    ret: list[dict[str, Any]] = []
    for tool in standard_tools or []:
        to_default_dict = getattr(tool, "to_default_dict", None)
        if callable(to_default_dict):
            payload = to_default_dict()
        elif isinstance(tool, dict):
            payload = tool
        else:
            continue
        if isinstance(payload, dict) and payload.get("type") == "function":
            ret.append(payload)
        elif isinstance(payload, dict):
            ret.append({"type": "function", "function": payload})
    custom_tools = getattr(tools, "custom_tools", None)
    if isinstance(custom_tools, dict):
        for values in custom_tools.values():
            if isinstance(values, list):
                ret.extend(item for item in values if isinstance(item, dict))
    return ret


def _metadata_from_processor(processor: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    settings = getattr(processor, "_settings", None)
    model = getattr(settings, "model", None) or getattr(processor, "model", None)
    if isinstance(model, str):
        metadata["model"] = model
    provider = _provider_from_processor(processor)
    if provider:
        metadata["provider"] = provider
    return metadata


def _metadata_from_metric(metric: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    model = getattr(metric, "model", None)
    if isinstance(model, str):
        metadata["model"] = model
    processor = getattr(metric, "processor", None)
    if isinstance(processor, str):
        provider = _provider_from_name(processor)
        if provider:
            metadata["provider"] = provider
    return metadata


def _provider_from_processor(processor: Any) -> str | None:
    module = getattr(type(processor), "__module__", "")
    return _provider_from_name(module)


def _provider_from_name(name: str) -> str | None:
    lowered = name.lower()
    providers = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "gemini": "google",
        "mistral": "mistral",
        "cohere": "cohere",
        "bedrock": "bedrock",
        "aws": "bedrock",
        "openrouter": "openrouter",
    }
    for needle, provider in providers.items():
        if needle in lowered:
            return provider
    return None


def _processor_name(processor: Any) -> str | None:
    name = getattr(processor, "name", None)
    if isinstance(name, str):
        return name
    if processor is not None:
        return type(processor).__name__
    return None


def _is_pipeline_sink_processor(processor: Any) -> bool:
    name = _processor_name(processor)
    return isinstance(name, str) and name.endswith("::Sink")


def _filter_llm_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in _ALLOWED_LLM_METADATA_FIELDS and value is not None}


def _llm_usage_metrics(usage: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    for key, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("tokens", total_tokens),
        ("cache_read_input_tokens", cache_read),
        ("cache_creation_input_tokens", cache_creation),
        ("reasoning_tokens", reasoning_tokens),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
    return metrics


def _tool_call_from_pipecat_call(call: Any) -> dict[str, Any] | None:
    name = getattr(call, "function_name", None)
    tool_call_id = getattr(call, "tool_call_id", None)
    if not isinstance(name, str):
        return None
    arguments = getattr(call, "arguments", None)
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments if arguments is not None else {}, sort_keys=True),
        },
    }


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in '[{"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
