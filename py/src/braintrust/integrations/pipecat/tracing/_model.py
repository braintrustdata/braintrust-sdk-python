"""Client ownership and operation-scoped model telemetry.

Ownership is resolved before execution, from clients referenced by the actual
service implementation. It never depends on setup state or observed children.
"""

import dis
import inspect
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from types import CodeType
from typing import Any
from weakref import WeakKeyDictionary, WeakValueDictionary

from braintrust.integrations.utils import _is_not_given, _parse_openai_usage_metrics, _pcm_to_wav
from braintrust.logger import Attachment, Span, SpanTypeAttribute

from ._common import LOG, SpanFactory, metadata_from_processor


# Known instrumented model entry points on actual client objects. Merely
# reading an SDK client's configuration (or uploading a file) does not give
# that SDK ownership of a model request made through another transport.
_CLIENT_MODEL_PATHS = {
    "openai": {
        "llm_response": ("chat.completions.create", "chat.completions.parse", "responses.create", "responses.parse"),
        "stt": ("audio.transcriptions.create", "audio.translations.create"),
        "tts": ("audio.speech.create",),
    },
    "anthropic": {"llm_response": ("messages.create", "messages.stream")},
    "google.genai": {"llm_response": ("models.generate_content", "models.generate_content_stream")},
    "mistralai": {
        "llm_response": ("chat.complete", "chat.complete_async", "chat.stream", "chat.stream_async"),
        "stt": ("audio.transcriptions.complete", "audio.transcriptions.complete_async"),
        "tts": ("audio.speech.complete", "audio.speech.complete_async"),
    },
    "cohere": {"llm_response": ("chat", "chat_stream"), "stt": ("audio.transcriptions.create",)},
    "huggingface_hub": {"llm_response": ("chat_completion", "text_generation", "chat.completions.create")},
}
# Operation identity follows the executing task, including metrics emitted
# before queued start events and after queued completion events.
_ACTIVE_MODEL: ContextVar[Any] = ContextVar("braintrust_pipecat_model", default=None)
# Pipecat LLMTokenUsage field -> Braintrust metric.
_USAGE_METRIC_NAMES = {
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "total_tokens": "tokens",
    "cache_read_input_tokens": "prompt_cached_tokens",
    "cache_creation_input_tokens": "prompt_cache_creation_tokens",
    "reasoning_tokens": "completion_reasoning_tokens",
    "input_audio_tokens": "prompt_audio_tokens",
    "output_audio_tokens": "completion_audio_tokens",
}
_OPERATION_METHODS = {"llm_response": "_process_context", "stt": "run_stt", "tts": "run_tts"}


@lru_cache(maxsize=256)
def _self_references(code: CodeType) -> frozenset[tuple[str, ...]]:
    instructions = list(dis.get_instructions(code))
    paths = set()
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_FAST", "LOAD_DEREF"} or instruction.argval != "self":
            continue
        path = []
        for following in instructions[index + 1 :]:
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            path.append(following.argval)
        if path:
            paths.add(tuple(path))
    for child in code.co_consts:
        if isinstance(child, CodeType):
            paths.update(_self_references(child))
    return frozenset(paths)


def _sdk_model_reference(client: Any, path: tuple[str, ...], operation: str) -> bool:
    for cls in type(client).__mro__:
        for namespace, operations in _CLIENT_MODEL_PATHS.items():
            if cls.__module__ != namespace and not cls.__module__.startswith(namespace + "."):
                continue
            # A service may hold an SDK resource rather than the root client.
            # OpenAI resource modules repeat the final segment, e.g.
            # openai.resources.chat.completions.completions.AsyncCompletions.
            prefix = cls.__module__.removeprefix(namespace + ".resources.")
            resource = prefix.split(".") if prefix != cls.__module__ else []
            parts = [*resource, *path]
            normalized = []
            for part in parts:
                if part in {"beta", "aio", "v2", "with_raw_response", "with_streaming_response"}:
                    continue
                if not normalized or normalized[-1] != part:
                    normalized.append(part)
            if ".".join(normalized) in operations.get(operation, ()):
                return True
    return False


def provider_client_expected(service: Any, operation: str) -> bool:
    """Follow the service's bound execution methods to their client references.

    Require a reference to a known model entry point on an actual SDK client.
    References can be invoked directly or delegated to a service helper. Follow
    service methods without evaluating arbitrary properties; configuration-only
    reads and inherited unused clients do not establish model ownership.
    """
    pending = [getattr(service, _OPERATION_METHODS[operation], None)]
    visited: set[Any] = set()
    while pending:
        method = inspect.unwrap(pending.pop())
        function = getattr(method, "__func__", method)
        code = getattr(function, "__code__", None)
        if code is None or code in visited:
            continue
        visited.add(code)
        if "super" in code.co_names:
            # Resolve super() against the implementation's defining class,
            # while ordinary self.method calls retain dynamic dispatch.
            mro = type(service).__mro__
            owner = next(
                (
                    index
                    for index, cls in enumerate(mro)
                    if getattr(inspect.unwrap(vars(cls).get(function.__name__)), "__code__", None) is code
                ),
                None,
            )
            if owner is not None:
                for name in code.co_names:
                    for base in mro[owner + 1 :]:
                        candidate = vars(base).get(name)
                        if inspect.isfunction(candidate):
                            pending.append(candidate)
                            break
        for path in _self_references(code):
            name = path[0]
            client = vars(service).get(name)
            if client is not None and _sdk_model_reference(client, path[1:], operation):
                return True
            # Avoid evaluating properties and following framework routing,
            # metrics, or lifecycle methods back into unrelated model work.
            if name in {"push_frame", "process_frame", "process_generator", "tts_process_generator"}:
                continue
            candidate = inspect.getattr_static(type(service), name, None)
            if inspect.isfunction(candidate) and not candidate.__module__.startswith(
                ("pipecat.processors.", "pipecat.services.ai_service")
            ):
                pending.append(candidate)
    return False


def request_metadata(params: dict[str, Any]) -> dict[str, Any]:
    """The request settings a model child records; the rest of the request is not kept."""
    return {
        key: value
        for key in ("tools", "tool_choice", "parallel_tool_calls", "max_tool_calls")
        if (value := params.get(key)) is not None and not _is_not_given(value) and (key != "tools" or value)
    }


# Service -> operation -> covered. Ownership follows the service's code and
# clients, which are fixed once it executes; walk them once per service.
_COVERED: "WeakKeyDictionary[Any, dict[str, bool]]" = WeakKeyDictionary()


def _client_covered(service: Any, operation: str) -> bool:
    try:
        cached = _COVERED.setdefault(service, {})
    except TypeError:  # Not weak-referenceable or hashable: do not cache.
        cached = {}
    covered = cached.get(operation)
    if covered is None:
        covered = cached[operation] = provider_client_expected(service, operation)
    return covered


def _wav_part(data: bytes, filename: str) -> dict[str, Any]:
    attachment = Attachment(data=data, filename=filename, content_type="audio/wav")
    return {"type": "file", "file": {"filename": filename, "file_data": attachment, "byte_size": len(data)}}


@dataclass
class ModelOperation:
    """One optional model child; also the identity carried by queued metrics."""

    span: Span | None
    processor: Any
    parent: Span | None = None
    started: float = field(default_factory=time.monotonic)
    ended: bool = False
    first_content: bool = False
    usage: dict[str, int | float] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    audio: bytearray = field(default_factory=bytearray)
    audio_format: tuple[int, int] | None = None
    deferred: bool = False
    complete: bool = False
    error: Any = None
    input_audio: bytearray = field(default_factory=bytearray)
    input_rate: int = 16000
    audio_frames: WeakValueDictionary = field(default_factory=WeakValueDictionary)

    @classmethod
    def start(cls, spans: SpanFactory, parent: Span, processor: Any, operation: str, **kwargs: Any):
        span = None
        try:
            covered = _client_covered(processor, operation)
        except Exception:
            LOG.warning("Unable to determine Pipecat client ownership", exc_info=True)
            # Uncertain ownership must not create duplicate billable spans.
            covered = True
        if not covered:
            span = spans.start(
                "pipecat." + operation,
                parent=parent,
                type=SpanTypeAttribute.LLM,
                metadata=metadata_from_processor(processor),
                **kwargs,
            )
        return cls(span, processor, parent=parent)

    def request(self, params: dict[str, Any]) -> None:
        if self.span is None:
            return
        # Each observation is a complete request snapshot. Defer metadata until
        # completion so a later request builder can remove/replace adapter tools.
        self.request_metadata = request_metadata(params)

    def content(self) -> None:
        if self.span is not None and not self.first_content:
            self.first_content = True
            self.span.log(metrics={"time_to_first_token": time.monotonic() - self.started})

    def metrics(self, frame: Any) -> None:
        if self.span is None:
            return
        from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData  # pylint: disable=import-error

        for metric in frame.data or []:
            if metric.processor != self.processor.name:
                continue
            if isinstance(metric, LLMUsageMetricsData):
                names = _USAGE_METRIC_NAMES
                values = {name: getattr(metric.value, name, None) for name in names}
                # Pipecat defines total_tokens as gross of prompt-cache usage,
                # while prompt_tokens can be net (Anthropic) or gross (OpenAI).
                total, completion = values["total_tokens"], values["completion_tokens"]
                if total is not None and completion is not None and total >= completion:
                    values["prompt_tokens"] = total - completion
                self.report_usage(_parse_openai_usage_metrics(values, token_name_map=names, token_prefix_map={}))
                if metric.model:
                    self.span.log(metadata={"model": metric.model})
            elif isinstance(metric, TTFBMetricsData) and math.isfinite(metric.value) and metric.value >= 0:
                self.span.log(metrics={"time_to_first_token": metric.value})
            # TTS character counts are not a standardized media metric.

    def report_usage(self, usage: dict[str, int | float]) -> None:
        if self.span is None:
            return
        # Reports are snapshots. Repeated reports replace, never sum.
        self.usage.update({key: value for key, value in usage.items() if value >= 0})
        self.span.log(metrics=self.usage)

    def audio_chunk(self, frame: Any, *, capture: bool) -> None:
        if frame.id in self.audio_frames:
            return
        self.audio_frames[frame.id] = frame
        self.content()
        if self.span is None or not capture:
            return
        fmt = (frame.sample_rate, frame.num_channels)
        if self.audio_format is not None and self.audio_format != fmt:
            return
        self.audio_format = fmt
        self.audio.extend(frame.audio)

    def finish(self, *, end_time: float, output: Any = None, error: Any = None) -> None:
        if self.ended:
            return
        self.ended = True
        if self.span is None:
            return
        if self.audio and self.audio_format and self.complete and error is None:
            rate, channels = self.audio_format
            data = _pcm_to_wav(self.audio, sample_rate=rate, num_channels=channels)
            output = {"content": [_wav_part(data, "synthesized.wav")]}
        if self.input_audio:
            audio = self.input_audio
            data = (
                bytes(audio)
                if audio.startswith(b"RIFF")
                else _pcm_to_wav(audio, sample_rate=self.input_rate, num_channels=1)
            )
            self.span.log(input={"operation": "transcribe", "content": [_wav_part(data, "recognition.wav")]})
            self.input_audio.clear()
        if self.request_metadata:
            self.span.log(metadata=self.request_metadata)
            self.request_metadata.clear()
        if output is not None:
            self.span.log(output=output)
        if error is not None:
            self.span.log(error=error)
        self.span.end(end_time=end_time)
        self.audio.clear()
