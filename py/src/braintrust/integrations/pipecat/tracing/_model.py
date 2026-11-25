"""Client ownership and operation-scoped model telemetry.

Ownership is resolved before execution, from a known native execution route
or an explicit application client binding. It never depends on setup state or observed children.
"""

import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakValueDictionary

from braintrust.integrations.utils import _is_not_given, _parse_openai_usage_metrics, _pcm_to_wav
from braintrust.logger import Attachment, Span, SpanTypeAttribute

from .._model_clients import provider_client_expected
from ._common import LOG, SpanFactory, metadata_from_processor


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


def client_covered(processor: Any, operation: str) -> bool:
    """Resolve ownership before execution, independently of integration setup."""
    try:
        return provider_client_expected(processor, operation)
    except Exception:
        LOG.warning("Unable to determine Pipecat client ownership", exc_info=True)
        # Uncertain ownership must not create duplicate billable spans.
        return True


def request_metadata(params: dict[str, Any]) -> dict[str, Any]:
    """The request settings a model child records; the rest of the request is not kept."""
    return {
        key: value
        for key in ("tools", "tool_choice", "parallel_tool_calls", "max_tool_calls")
        if (value := params.get(key)) is not None and not _is_not_given(value) and (key != "tools" or value)
    }


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
    # TTS text sent for this operation, in order.
    prompt: str | None = None
    input_audio: bytearray = field(default_factory=bytearray)
    input_rate: int = 16000
    audio_frames: WeakValueDictionary = field(default_factory=WeakValueDictionary)

    @classmethod
    def start(
        cls,
        spans: SpanFactory,
        parent: Span,
        processor: Any,
        operation: str,
        *,
        covered: bool | None = None,
        **kwargs: Any,
    ):
        span = None
        if covered is None:
            covered = client_covered(processor, operation)
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
