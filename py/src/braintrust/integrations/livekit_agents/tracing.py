"""Direct tracing helpers for LiveKit Agents."""

import asyncio
import contextlib
import io
import json
import time
import wave
from contextvars import ContextVar
from typing import Any

from braintrust.logger import (
    NOOP_SPAN,
    Attachment,
    SpanTypeAttribute,
    _state,
    current_span,
    parent_context,
    start_span,
)


_SESSION_PARENT_ATTR = "__braintrust_livekit_session_parent__"
_SESSION_SPAN_ATTR = "__braintrust_livekit_session_span__"
_USER_SPEAKING_SPAN_ATTR = "__braintrust_livekit_user_speaking_span__"
_PLAYBACK_HANDLER_ATTACHED_ATTR = "__braintrust_livekit_playback_handler_attached__"
_PLAYBACK_START_ATTR = "__braintrust_livekit_playback_start__"
_PLAYBACK_AUDIO_ATTR = "__braintrust_livekit_playback_audio__"
_PLAYBACK_AUDIO_METADATA_ATTR = "__braintrust_livekit_playback_audio_metadata__"
_PLAYBACK_SESSION_STATE_ATTR = "__braintrust_livekit_playback_session_state__"
_METRICS_HANDLER_ATTACHED_ATTR = "__braintrust_livekit_metrics_handler_attached__"
_TTS_INPUT_TEXT_ATTR = "__braintrust_livekit_tts_input_text__"
_STT_SUPPRESS_METRICS_ATTR = "__braintrust_livekit_stt_suppress_metrics__"
_STT_LAST_METRICS_ATTR = "__braintrust_livekit_stt_last_metrics__"
_OUTER_PARENT_TO_SESSION_PARENT: dict[str, str] = {}
_OUTER_SPAN_ID_TO_SESSION_PARENT: dict[str, str] = {}
_LATEST_SESSION_PARENT: str | None = None
_ACTIVE_SESSION_PARENT: ContextVar[str | None] = ContextVar("braintrust_livekit_active_session_parent", default=None)


def attach_metrics_handler(obj: Any) -> bool:
    """Attach one Braintrust metrics listener to a LiveKit event emitter."""
    if getattr(obj, _METRICS_HANDLER_ATTACHED_ATTR, False):
        return True
    on = getattr(obj, "on", None)
    if not callable(on):
        return False

    def on_metrics_collected(event: Any) -> None:
        metrics_obj = getattr(event, "metrics", event)
        if metrics_obj is None:
            return
        metrics_type = getattr(metrics_obj, "type", None)
        parent = getattr(obj, _SESSION_PARENT_ATTR, None)
        with parent_context(parent if isinstance(parent, str) else None):
            if metrics_type == "llm_metrics":
                _log_metric_on_existing_span(obj, metrics_obj)
            elif metrics_type == "realtime_model_metrics":
                _log_metric_on_existing_span(obj, metrics_obj)
            elif metrics_type == "tts_metrics":
                _log_metric_span(
                    "tts_request",
                    metrics_obj,
                    SpanTypeAttribute.TASK,
                    parent=parent if isinstance(parent, str) else None,
                    input={"text": getattr(obj, _TTS_INPUT_TEXT_ATTR, None)},
                    first_response_kind="audio",
                )
            elif metrics_type == "stt_metrics":
                if getattr(obj, _STT_SUPPRESS_METRICS_ATTR, False):
                    setattr(obj, _STT_LAST_METRICS_ATTR, metrics_obj)
                    return
                _log_metric_span(
                    "stt_processing",
                    metrics_obj,
                    SpanTypeAttribute.TASK,
                    parent=parent if isinstance(parent, str) else None,
                )
            elif metrics_type == "eou_metrics":
                _log_vad_endpointing_span(
                    metrics_obj,
                    parent=parent if isinstance(parent, str) else None,
                )
                _log_metric_span(
                    "eou_detection",
                    metrics_obj,
                    SpanTypeAttribute.TASK,
                    parent=parent if isinstance(parent, str) else None,
                )

    on("metrics_collected", on_metrics_collected)
    setattr(obj, _METRICS_HANDLER_ATTACHED_ATTR, True)
    return True


async def traced_session_start(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    global _LATEST_SESSION_PARENT  # pylint: disable=global-statement
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    if not isinstance(parent, str):
        outer_parent = _current_span_export()
        outer_span_id = _current_span_id()
        span = start_span(name="livekit_agent_session", type=SpanTypeAttribute.TASK, set_current=False)
        parent = span.export()
        setattr(instance, _SESSION_SPAN_ATTR, span)
        setattr(instance, _SESSION_PARENT_ATTR, parent)
        if outer_parent is not None:
            _OUTER_PARENT_TO_SESSION_PARENT[outer_parent] = parent
        if outer_span_id is not None:
            _OUTER_SPAN_ID_TO_SESSION_PARENT[outer_span_id] = parent
        _LATEST_SESSION_PARENT = parent
        _propagate_session_parent_to_start_args(instance, args, kwargs)
    try:
        result = await wrapped(*args, **kwargs)
    except BaseException:
        _end_session_span(instance)
        raise
    _propagate_session_parent_to_components(instance)
    _attach_session_playback_handler(instance)
    return result


async def traced_session_aexit(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _end_session_span(instance)


async def traced_session_close(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _end_session_span(instance)


def traced_update_user_state(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    global _LATEST_SESSION_PARENT  # pylint: disable=global-statement
    state = args[0] if args else kwargs.get("state")
    session_parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    if isinstance(session_parent, str):
        _LATEST_SESSION_PARENT = session_parent
    if state == "speaking" and getattr(instance, _USER_SPEAKING_SPAN_ATTR, None) is None:
        parent = getattr(instance, _SESSION_PARENT_ATTR, None)
        span = start_span(
            name="user_speaking",
            type=SpanTypeAttribute.TASK,
            parent=parent if isinstance(parent, str) else None,
            set_current=False,
        )
        setattr(instance, _USER_SPEAKING_SPAN_ATTR, span)
    elif state != "speaking":
        _end_user_speaking_span(instance)
    try:
        return wrapped(*args, **kwargs)
    except BaseException:
        if state == "speaking":
            _end_user_speaking_span(instance)
        raise


async def traced_stt_recognize(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    setattr(instance, _STT_LAST_METRICS_ATTR, None)
    setattr(instance, _STT_SUPPRESS_METRICS_ATTR, True)
    try:
        result = await wrapped(*args, **kwargs)
    finally:
        setattr(instance, _STT_SUPPRESS_METRICS_ATTR, False)
    metrics_obj = getattr(instance, _STT_LAST_METRICS_ATTR, None)
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    if metrics_obj is not None:
        _log_metric_span(
            "stt_processing",
            metrics_obj,
            SpanTypeAttribute.TASK,
            parent=parent if isinstance(parent, str) else None,
            output=_speech_event_output(result),
        )
    return result


def traced_llm_chat(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    stream = wrapped(*args, **kwargs)
    parent = getattr(instance, _SESSION_PARENT_ATTR, None) or _LATEST_SESSION_PARENT or _state.current_parent.get()
    if isinstance(parent, str):
        _set_component_session(stream, parent, getattr(instance, _SESSION_SPAN_ATTR, None))
    return stream


def traced_llm_stream_init(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    result = wrapped(*args, **kwargs)
    if _LATEST_SESSION_PARENT is not None:
        _set_component_session(instance, _LATEST_SESSION_PARENT, None)
    return result


async def traced_llm_stream_run(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    llm = getattr(instance, "_llm", None)
    parent = (
        getattr(instance, _SESSION_PARENT_ATTR, None)
        or getattr(llm, _SESSION_PARENT_ATTR, None)
        or _ACTIVE_SESSION_PARENT.get()
        or _session_parent_for_current_span()
        or _LATEST_SESSION_PARENT
        or _state.current_parent.get()
    )
    if _LATEST_SESSION_PARENT is not None:
        parent = _LATEST_SESSION_PARENT
    if llm is not None:
        attach_metrics_handler(llm)
    parent_arg = parent if isinstance(parent, str) else None
    with parent_context(parent_arg):
        return await _traced_async_span(
            "llm_request_run",
            SpanTypeAttribute.TASK,
            wrapped,
            args,
            kwargs,
            parent=parent_arg,
        )


async def traced_execute_tools_task(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    session = kwargs.get("session")
    parent = getattr(session, _SESSION_PARENT_ATTR, None)
    result = await _traced_async_span(
        "function_tool",
        SpanTypeAttribute.TOOL,
        wrapped,
        args,
        kwargs,
        parent=parent if isinstance(parent, str) else None,
        event_builder=lambda: _tool_execution_event(kwargs.get("tool_output")),
    )
    return result


async def _traced_async_span(
    name: str,
    span_type: str,
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent: str | None = None,
    event_builder: Any | None = None,
    **event: Any,
) -> Any:
    with _without_current_span(parent):
        span = start_span(name=name, type=span_type, parent=parent, set_current=False)
    exported = span.export()
    try:
        with _without_current_span(parent), parent_context(exported):
            result = await wrapped(*args, **kwargs)
    except asyncio.CancelledError:
        span.end()
        raise
    except Exception as exc:
        span.log(error=exc, **event)
        span.end()
        raise
    except BaseException:
        span.end()
        raise
    if callable(event_builder):
        event.update(event_builder())
    span.log(**event)
    span.end()
    return result


@contextlib.contextmanager
def _without_current_span(parent: str | None):
    if parent is None:
        yield
        return
    token = _state.context_manager.set_current_span(None)
    legacy_token = _state.current_span.set(NOOP_SPAN)
    try:
        yield
    finally:
        _state.current_span.reset(legacy_token)
        _state.context_manager.unset_current_span(token)


@contextlib.contextmanager
def _current_span_context(span: Any):
    if span is None:
        yield
        return
    token = _state.context_manager.set_current_span(span)
    try:
        yield
    finally:
        _state.context_manager.unset_current_span(token)


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str) or not value or value[0] not in '[{"':
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_execution_event(tool_output: Any) -> dict[str, Any]:
    outputs = getattr(tool_output, "output", None)
    if not isinstance(outputs, list) or not outputs:
        return {}
    calls = []
    for item in outputs:
        fnc_call = getattr(item, "fnc_call", None)
        fnc_call_out = getattr(item, "fnc_call_out", None)
        call = {
            "name": getattr(fnc_call, "name", None),
            "arguments": _parse_jsonish(getattr(fnc_call, "arguments", None)),
            "call_id": getattr(fnc_call, "call_id", None),
            "output": getattr(fnc_call_out, "output", None),
            "is_error": getattr(fnc_call_out, "is_error", None),
        }
        calls.append({key: value for key, value in call.items() if value is not None})
    if len(calls) == 1:
        call = calls[0]
        return {
            "input": {"name": call.get("name"), "arguments": call.get("arguments")},
            "output": {"output": call.get("output"), "is_error": call.get("is_error")},
            "metadata": {"call_id": call.get("call_id")},
        }
    return {"output": {"tool_calls": calls}, "metadata": {"tool_call_count": len(calls)}}


async def traced_session_run(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    global _LATEST_SESSION_PARENT  # pylint: disable=global-statement
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    token = None
    if isinstance(parent, str):
        _LATEST_SESSION_PARENT = parent
        token = _ACTIVE_SESSION_PARENT.set(parent)
    try:
        with (
            _current_span_context(getattr(instance, _SESSION_SPAN_ATTR, None)),
            parent_context(parent if isinstance(parent, str) else None),
        ):
            result = await wrapped(*args, **kwargs)
    finally:
        if token is not None:
            _ACTIVE_SESSION_PARENT.reset(token)
    metadata = _agent_turn_metadata(result)
    if metadata:
        span = getattr(instance, _SESSION_SPAN_ATTR, None)
        if span is not None:
            span.log(metadata=metadata)
        else:
            with start_span(name="livekit_agent_session", type=SpanTypeAttribute.TASK) as fallback_span:
                fallback_span.log(metadata=metadata)
    return result


def traced_session_say(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return wrapped(*args, **kwargs)


def traced_metrics_emitter_call(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return wrapped(*args, **kwargs)


def traced_vad_stream(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return wrapped(*args, **kwargs)


async def traced_vad_stream_anext(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    event = await wrapped(*args, **kwargs)
    _log_vad_endpointing_event(getattr(instance, "_vad", None), event)
    return event


def _log_vad_endpointing_event(vad: Any, event: Any) -> None:
    event_type = getattr(event, "type", None)
    event_type_value = getattr(event_type, "value", event_type)
    if event_type_value != "end_of_speech":
        return
    silence_duration = getattr(event, "silence_duration", None)
    if not isinstance(silence_duration, (int, float)) or isinstance(silence_duration, bool) or silence_duration <= 0:
        return
    end_time = time.time()
    parent = getattr(vad, _SESSION_PARENT_ATTR, None)
    metadata = {
        "speech_duration": getattr(event, "speech_duration", None),
        "silence_duration": silence_duration,
        "samples_index": getattr(event, "samples_index", None),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    span = start_span(
        name="vad_endpointing",
        type=SpanTypeAttribute.TASK,
        start_time=end_time - silence_duration,
        set_current=False,
        parent=parent if isinstance(parent, str) else None,
    )
    span.log(metadata=metadata)
    span.end(end_time=end_time)


def traced_session_audio_output_changed(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    _propagate_session_parent_to_components(instance)
    _attach_session_playback_handler(instance)
    return wrapped(*args, **kwargs)


async def traced_audio_output_capture_frame(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if getattr(instance, _PLAYBACK_HANDLER_ATTACHED_ATTR, False):
        if getattr(instance, _PLAYBACK_START_ATTR, None) is None:
            setattr(instance, _PLAYBACK_START_ATTR, time.time())
        _capture_playback_audio(instance, args[0] if args else kwargs.get("frame"))
    return await wrapped(*args, **kwargs)


def traced_tts_synthesize(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    stream = wrapped(*args, **kwargs)
    text = args[0] if args else kwargs.get("text")
    try:
        setattr(instance, _TTS_INPUT_TEXT_ATTR, text)
    except Exception:
        pass
    return _TTSStreamProxy(stream, text)


class _TTSStreamProxy:
    def __init__(self, stream: Any, text: Any):
        self._stream = stream
        self._start = time.time()
        self._first_chunk_time = None
        self._ended = False
        self._span = start_span(name="tts_request", type=SpanTypeAttribute.TASK, input={"text": text})

    def __aiter__(self):
        return self

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        exit_ = getattr(self._stream, "__aexit__", None)
        try:
            if callable(exit_):
                return await exit_(exc_type, exc, tb)
            close = getattr(self._stream, "aclose", None)
            if callable(close):
                await close()
            return None
        finally:
            self._end()

    async def __anext__(self):
        try:
            item = await self._stream.__anext__()
        except StopAsyncIteration:
            self._end()
            raise
        except Exception as exc:
            self._span.log(error=exc)
            self._span.end()
            self._ended = True
            raise
        if self._first_chunk_time is None:
            self._first_chunk_time = time.time()
        return item

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        try:
            if callable(close):
                await close()
        finally:
            self._end()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _end(self) -> None:
        if self._ended:
            return
        self._ended = True
        metrics = {}
        if self._first_chunk_time is not None:
            ttfb = self._first_chunk_time - self._start
            metrics["time_to_first_token"] = ttfb
        self._span.log(metrics=metrics, metadata={"first_response_kind": "audio"})
        self._span.end()


def _propagate_session_parent_to_start_args(instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    if not isinstance(parent, str):
        return
    span = getattr(instance, _SESSION_SPAN_ATTR, None)
    event_parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    agent = args[0] if args else kwargs.get("agent")
    if agent is not None:
        _set_component_session(agent, parent, span)
        for attr in ("llm", "stt", "tts", "vad", "_llm", "_stt", "_tts", "_vad"):
            component = getattr(agent, attr, None)
            if component is not None:
                _set_component_session(component, parent, span)


def _propagate_session_parent_to_components(instance: Any) -> None:
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    if not isinstance(parent, str):
        return
    span = getattr(instance, _SESSION_SPAN_ATTR, None)
    event_parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    _set_component_session(instance, parent, span)
    for attr in ("llm", "stt", "tts", "vad", "_llm", "_stt", "_tts", "_vad", "_agent", "agent", "output"):
        component = getattr(instance, attr, None)
        if component is not None:
            _set_component_session(component, parent, span)
            for nested_attr in ("llm", "stt", "tts", "vad", "audio", "_llm", "_stt", "_tts", "_vad", "_audio"):
                nested_component = getattr(component, nested_attr, None)
                if nested_component is not None:
                    _set_component_session(nested_component, parent, span)


def _set_component_session(component: Any, parent: str, span: Any) -> None:
    try:
        setattr(component, _SESSION_PARENT_ATTR, parent)
        if span is not None:
            setattr(component, _SESSION_SPAN_ATTR, span)
    except Exception:
        pass


def _attach_session_playback_handler(instance: Any) -> None:
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    span = getattr(instance, _SESSION_SPAN_ATTR, None)
    output = getattr(instance, "output", None)
    audio_output = getattr(output, "audio", None)
    state = getattr(instance, _PLAYBACK_SESSION_STATE_ATTR, None)
    if state is None:
        state = {}
        setattr(instance, _PLAYBACK_SESSION_STATE_ATTR, state)
    for sink in _iter_audio_outputs(audio_output):
        if isinstance(parent, str):
            _set_component_session(sink, parent, span)
        setattr(sink, _PLAYBACK_SESSION_STATE_ATTR, state)
    if audio_output is not None:
        attach_playback_handler(audio_output)


def _iter_audio_outputs(audio_output: Any):
    current = audio_output
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "next_in_chain", None)


def attach_playback_handler(obj: Any) -> bool:
    if getattr(obj, _PLAYBACK_HANDLER_ATTACHED_ATTR, False):
        return True
    on = getattr(obj, "on", None)
    if not callable(on):
        return False

    def on_playback_finished(event: Any) -> None:
        playback_position = getattr(event, "playback_position", None)
        if _is_duplicate_playback_event(obj, playback_position, getattr(event, "interrupted", None)):
            _clear_playback_audio(obj)
            return
        end = time.time()
        start = getattr(obj, _PLAYBACK_START_ATTR, None)
        if not isinstance(start, (int, float)):
            start = end - playback_position if isinstance(playback_position, (int, float)) else end
        metadata = {
            "audio_duration": playback_position,
            "interrupted": getattr(event, "interrupted", None),
            "output_label": getattr(obj, "label", None),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        output = {}
        audio = _pop_playback_audio(obj)
        if audio is not None:
            output["audio"] = audio
        synchronized_transcript = getattr(event, "synchronized_transcript", None)
        if synchronized_transcript:
            output["text"] = synchronized_transcript
        parent = getattr(obj, _SESSION_PARENT_ATTR, None)
        span = start_span(
            name="agent_speaking",
            type=SpanTypeAttribute.TASK,
            start_time=start,
            set_current=False,
            parent=parent if isinstance(parent, str) else None,
        )
        span.log(metadata=metadata, output=output or None)
        span.end()
        setattr(obj, _PLAYBACK_START_ATTR, None)

    on("playback_finished", on_playback_finished)
    setattr(obj, _PLAYBACK_HANDLER_ATTACHED_ATTR, True)
    return True


def _is_duplicate_playback_event(obj: Any, playback_position: Any, interrupted: Any) -> bool:
    state = getattr(obj, _PLAYBACK_SESSION_STATE_ATTR, None)
    if not isinstance(state, dict):
        return False
    now = time.time()
    key = (playback_position, interrupted)
    last_key = state.get("last_key")
    last_time = state.get("last_time")
    state["last_key"] = key
    state["last_time"] = now
    return key == last_key and isinstance(last_time, (int, float)) and now - last_time < 1.0


def _capture_playback_audio(obj: Any, frame: Any) -> None:
    data = getattr(frame, "data", None)
    if data is None:
        return
    try:
        chunk = bytes(data)
    except Exception:
        return
    if not chunk:
        return
    audio = getattr(obj, _PLAYBACK_AUDIO_ATTR, None)
    if audio is None:
        audio = bytearray()
        setattr(obj, _PLAYBACK_AUDIO_ATTR, audio)
        metadata = {
            "sample_rate": getattr(frame, "sample_rate", None),
            "num_channels": getattr(frame, "num_channels", None),
            "samples_per_channel": getattr(frame, "samples_per_channel", None),
        }
        setattr(obj, _PLAYBACK_AUDIO_METADATA_ATTR, metadata)
    audio.extend(chunk)


def _clear_playback_audio(obj: Any) -> None:
    setattr(obj, _PLAYBACK_AUDIO_ATTR, None)
    setattr(obj, _PLAYBACK_AUDIO_METADATA_ATTR, None)


def _pop_playback_audio(obj: Any) -> Attachment | None:
    audio = getattr(obj, _PLAYBACK_AUDIO_ATTR, None)
    metadata = getattr(obj, _PLAYBACK_AUDIO_METADATA_ATTR, None) or {}
    _clear_playback_audio(obj)
    if not audio:
        return None
    sample_rate = metadata.get("sample_rate")
    num_channels = metadata.get("num_channels")
    suffix = f"_{sample_rate}hz_{num_channels}ch" if sample_rate and num_channels else ""
    if isinstance(sample_rate, int) and isinstance(num_channels, int) and sample_rate > 0 and num_channels > 0:
        return Attachment(
            data=_pcm_to_wav(bytes(audio), sample_rate=sample_rate, num_channels=num_channels),
            filename=f"agent_speaking{suffix}.wav",
            content_type="audio/wav",
        )
    return Attachment(
        data=bytes(audio),
        filename=f"agent_speaking{suffix}.pcm",
        content_type="audio/pcm",
    )


def _pcm_to_wav(audio: bytes, *, sample_rate: int, num_channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        writer: Any = wav_file
        writer.setnchannels(num_channels)  # pylint: disable=no-member
        writer.setsampwidth(2)  # pylint: disable=no-member
        writer.setframerate(sample_rate)  # pylint: disable=no-member
        writer.writeframes(audio)  # pylint: disable=no-member
    return buffer.getvalue()


_AGENT_TURN_METADATA_FIELDS = (
    "generation_id",
    "speech_id",
    "interrupted",
    "function_calls",
    "function_tools",
)


def _agent_turn_metadata(result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in _iter_agent_turn_metadata_sources(result):
        for field in _AGENT_TURN_METADATA_FIELDS:
            value = _get_metadata_value(source, field)
            if value is not None and field not in metadata:
                metadata[field] = value
    return metadata


def _iter_agent_turn_metadata_sources(result: Any) -> list[Any]:
    sources = [result]
    events = getattr(result, "events", None)
    if isinstance(events, list):
        sources.extend(events)
    return sources


def _get_metadata_value(source: Any, field: str) -> Any:
    for key in (field, f"lk.{field}", f"lk.response.{field}"):
        if isinstance(source, dict) and key in source:
            return source[key]
        value = getattr(source, key, None)
        if value is not None:
            return value
    return None


def _end_user_speaking_span(instance: Any) -> None:
    span = getattr(instance, _USER_SPEAKING_SPAN_ATTR, None)
    if span is None:
        return
    span.end()
    setattr(instance, _USER_SPEAKING_SPAN_ATTR, None)


def _end_session_span(instance: Any) -> None:
    parent = getattr(instance, _SESSION_PARENT_ATTR, None)
    span = getattr(instance, _SESSION_SPAN_ATTR, None)
    if span is not None:
        span.end()
        setattr(instance, _SESSION_SPAN_ATTR, None)
    setattr(instance, _SESSION_PARENT_ATTR, None)


def _current_span_export() -> str | None:
    span = current_span()
    if span == NOOP_SPAN:
        return None
    try:
        return span.export()
    except Exception:
        return None


def _current_span_id() -> str | None:
    span = current_span()
    if span == NOOP_SPAN:
        return None
    span_id = getattr(span, "span_id", None)
    return span_id if isinstance(span_id, str) else None


def _session_parent_for_current_span() -> str | None:
    outer_span_id = _current_span_id()
    if outer_span_id is not None:
        parent = _OUTER_SPAN_ID_TO_SESSION_PARENT.get(outer_span_id)
        if parent is not None:
            return parent
    outer_parent = _current_span_export()
    if outer_parent is None:
        return None
    return _OUTER_PARENT_TO_SESSION_PARENT.get(outer_parent)


def _log_metric_span(
    name: str,
    metrics_obj: Any,
    span_type: str,
    parent: str | None = None,
    input: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    metrics_payload = _metrics_from_object(metrics_obj)
    event = _metric_event(metrics_payload, **metadata)
    if name == "eou_detection":
        event.update(_eou_detection_io(metrics_payload))
    if output is not None:
        event["output"] = output
    start_time, end_time = _metric_span_times(metrics_payload)
    span = start_span(name=name, type=span_type, start_time=start_time, set_current=False, parent=parent, input=input)
    span.log(**event)
    span.end(end_time=end_time)


def _log_vad_endpointing_span(metrics_obj: Any, parent: str | None = None) -> None:
    metrics_payload = _metrics_from_object(metrics_obj)
    endpointing_delay = _first_numeric_metric(
        metrics_payload,
        "endpointing_delay",
        "end_of_turn_delay",
        "end_of_utterance_delay",
        "eou_delay",
    )
    if endpointing_delay is None or endpointing_delay <= 0:
        return
    timestamp = metrics_payload.get("timestamp")
    end_time = timestamp if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) else None
    start_time = end_time - endpointing_delay if end_time is not None else None
    metadata = _promoted_metadata(metrics_payload)
    metadata["livekit_metrics"] = {
        key: value
        for key, value in metrics_payload.items()
        if key
        not in {"metadata", "type", "endpointing_delay", "end_of_turn_delay", "end_of_utterance_delay", "eou_delay"}
    }
    if not metadata["livekit_metrics"]:
        metadata.pop("livekit_metrics")
    span = start_span(
        name="vad_endpointing",
        type=SpanTypeAttribute.TASK,
        start_time=start_time,
        set_current=False,
        parent=parent,
    )
    span.log(metadata=metadata)
    span.end(end_time=end_time)


def _first_numeric_metric(metrics_payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics_payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _eou_detection_io(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    event: dict[str, Any] = {}
    text = _latest_user_text(metrics_payload)
    if text:
        event["input"] = {"text": text}
    output = {
        "probability": metrics_payload.get("probability"),
        "language": metrics_payload.get("language"),
        "unlikely_threshold": metrics_payload.get("unlikely_threshold"),
    }
    probability = output["probability"]
    threshold = output["unlikely_threshold"]
    if isinstance(probability, (int, float)) and isinstance(threshold, (int, float)):
        output["is_end_of_turn"] = probability >= threshold
    output = {key: value for key, value in output.items() if value is not None}
    if output:
        event["output"] = output
    return event


def _latest_user_text(metrics_payload: dict[str, Any]) -> str | None:
    for key in ("text", "user_text", "transcript", "user_transcript"):
        value = metrics_payload.get(key)
        if isinstance(value, str) and value:
            return value
    chat_ctx = metrics_payload.get("chat_ctx")
    parsed = _parse_jsonish(chat_ctx)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return None
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text = " ".join(part for part in content if isinstance(part, str) and part)
            if text:
                return text
    return None


def _speech_event_output(event: Any) -> dict[str, Any]:
    alternatives = getattr(event, "alternatives", None)
    if not alternatives:
        return {}
    alternative = alternatives[0]
    output = {
        "text": getattr(alternative, "text", None),
        "language": getattr(alternative, "language", None),
        "confidence": getattr(alternative, "confidence", None),
    }
    return {key: value for key, value in output.items() if value is not None}


def _metric_span_times(metrics_payload: dict[str, Any]) -> tuple[float | None, float | None]:
    timestamp = metrics_payload.get("timestamp")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        return None, None
    duration = metrics_payload.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return timestamp - duration, timestamp
    audio_duration = metrics_payload.get("audio_duration")
    if isinstance(audio_duration, (int, float)) and not isinstance(audio_duration, bool) and audio_duration > 0:
        return timestamp - audio_duration, timestamp
    return timestamp, timestamp


def _log_metric_on_existing_span(obj: Any, metrics_obj: Any) -> None:
    span = getattr(obj, _SESSION_SPAN_ATTR, None)
    if span is None:
        return
    metrics_payload = _metrics_from_object(metrics_obj)
    event = _metric_event(metrics_payload)
    _drop_llm_token_metrics(event, metrics_payload)
    span.log(**event)


def _drop_llm_token_metrics(event: dict[str, Any], metrics_payload: dict[str, Any]) -> None:
    metrics = event.get("metrics")
    token_fields = (
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "tokens",
        "total_tokens",
    )
    if isinstance(metrics, dict):
        for key in token_fields:
            metrics.pop(key, None)
    metadata = event.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    livekit_metrics = metadata.setdefault("livekit_metrics", {})
    if not isinstance(livekit_metrics, dict):
        return
    for key in token_fields:
        value = metrics_payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            livekit_metrics.setdefault(key, value)


def _metric_event(metrics_payload: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    bt_metrics = _promoted_metrics(metrics_payload)
    promoted_metadata = _promoted_metadata(metrics_payload)
    metadata_payload = {**promoted_metadata, **metadata}
    livekit_metrics = _compact_livekit_metrics(metrics_payload, bt_metrics, promoted_metadata)
    if livekit_metrics:
        metadata_payload["livekit_metrics"] = livekit_metrics
    return {"metrics": bt_metrics, "metadata": metadata_payload}


def _metrics_from_object(metrics_obj: Any) -> dict[str, Any]:
    to_dict = getattr(metrics_obj, "model_dump", None)
    if callable(to_dict):
        try:
            ret = to_dict(mode="python")
            if isinstance(ret, dict):
                return ret
        except Exception:
            pass
    return dict(getattr(metrics_obj, "__dict__", {}) or {})


def _compact_livekit_metrics(
    metrics_payload: dict[str, Any], promoted_metrics: dict[str, Any], promoted_metadata: dict[str, Any]
) -> dict[str, Any]:
    promoted_metric_sources = {
        source
        for source, dest in _PROMOTED_METRIC_FIELDS
        if dest in promoted_metrics and metrics_payload.get(source) == promoted_metrics[dest]
    }
    if promoted_metrics.get("time_to_first_token") == metrics_payload.get("ttft"):
        promoted_metric_sources.add("ttft")
    if promoted_metrics.get("time_to_first_token") == metrics_payload.get("ttfb"):
        promoted_metric_sources.add("ttfb")
    promoted_metadata_sources = {
        source
        for source in (
            "label",
            "request_id",
            "speech_id",
            "segment_id",
            "generation_id",
            "interrupted",
            "function_calls",
            "function_tools",
        )
        if source in promoted_metadata and metrics_payload.get(source) == promoted_metadata[source]
    }
    compact = {
        key: value
        for key, value in metrics_payload.items()
        if key not in promoted_metric_sources
        and key not in promoted_metadata_sources
        and key not in {"metadata", "type"}
    }
    return compact


def _promoted_metadata(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    nested_metadata = metrics_payload.get("metadata")
    if isinstance(nested_metadata, dict):
        model = nested_metadata.get("model_name")
        provider = nested_metadata.get("model_provider")
        if isinstance(model, str):
            ret["model"] = model
        if isinstance(provider, str):
            ret["provider"] = provider
    for source, dest in (
        ("label", "label"),
        ("request_id", "request_id"),
        ("speech_id", "speech_id"),
        ("segment_id", "segment_id"),
        ("generation_id", "generation_id"),
        ("function_calls", "function_calls"),
        ("function_tools", "function_tools"),
    ):
        value = metrics_payload.get(source)
        if isinstance(value, str):
            ret[dest] = value
    interrupted = metrics_payload.get("interrupted")
    if isinstance(interrupted, bool):
        ret["interrupted"] = interrupted
    return ret


_PROMOTED_METRIC_FIELDS = (
    ("duration", "duration"),
    ("prompt_tokens", "prompt_tokens"),
    ("completion_tokens", "completion_tokens"),
    ("total_tokens", "tokens"),
    ("input_tokens", "prompt_tokens"),
    ("output_tokens", "completion_tokens"),
)


def _promoted_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    if isinstance(metrics_payload.get("ttft"), (int, float)):
        ret["time_to_first_token"] = metrics_payload["ttft"]
    if isinstance(metrics_payload.get("ttfb"), (int, float)):
        ret["time_to_first_token"] = metrics_payload["ttfb"]
    for source, dest in _PROMOTED_METRIC_FIELDS:
        value = metrics_payload.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                source in {"input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"}
                and value == 0
            ):
                continue
            ret[dest] = value
    return ret
