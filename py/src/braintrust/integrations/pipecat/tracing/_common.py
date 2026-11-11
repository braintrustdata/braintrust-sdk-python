"""Span and processor helpers shared by the Pipecat trackers."""

import logging
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from braintrust._audio.clock import SessionClock
from braintrust._audio.schema import TURN_ID, span_id
from braintrust.logger import Span, SpanTypeAttribute
from braintrust.logger import start_span as _bt_start_span


if TYPE_CHECKING:
    from pipecat.frames.frames import TTSTextFrame
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.utils.string import TextPartForConcatenation


LOG = logging.getLogger("braintrust")
INSTRUMENTATION = "pipecat-auto"
# Correlation is best effort when a pipeline never consumes what it produces
# (e.g. never submits context). Bound pending references, never trace payloads.
MAX_PENDING = 256

ProcessorKind = Literal["stt", "tts", "llm", "output"]


@dataclass
class SpeechText:
    """Join TTS text with the same spacing rules as Pipecat's aggregator."""

    parts: list["TextPartForConcatenation"] = field(default_factory=list)

    def append(self, frame: "TTSTextFrame") -> None:
        from pipecat.utils.string import TextPartForConcatenation  # pylint: disable=import-error

        self.parts.append(TextPartForConcatenation(frame.text, frame.includes_inter_frame_spaces))

    def __str__(self) -> str:
        from pipecat.utils.string import concatenate_aggregated_text  # pylint: disable=import-error

        return concatenate_aggregated_text(self.parts)


def start_span(*args: Any, **kwargs: Any) -> Span:
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


class SpanFactory:
    """Creates session spans; a span within a turn records that turn's ID."""

    def __init__(self, session: Span, clock: SessionClock) -> None:
        self.session = session
        self._clock = clock

    def start(
        self,
        name: str,
        *,
        turn: Span | None = None,
        parent: Span | None = None,
        type: SpanTypeAttribute = SpanTypeAttribute.TASK,
        start_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        can_set_current: bool = False,
        **kwargs: Any,
    ) -> Span:
        metadata = dict(metadata or {})
        if turn is not None:
            metadata[TURN_ID] = span_id(turn)
        owner = parent or turn or self.session
        # can_set_current only permits an explicit set_current(); the span is
        # never made current implicitly.
        return start_span(
            name=name,
            type=type,
            parent=owner.export(),
            set_current=can_set_current,
            start_time=self._clock.wall(start_ms),
            metadata=metadata,
            **kwargs,
        )

    def turn(self, name: str) -> Span:
        span = self.start(name)
        span.log(metadata={TURN_ID: span_id(span)})
        return span


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    return ""


@cache
def processor_kind(cls: type) -> ProcessorKind | None:
    kinds: dict[str, ProcessorKind] = {
        "STTService": "stt",
        "TTSService": "tts",
        "LLMService": "llm",
        "BaseOutputTransport": "output",
    }
    return next((kinds[base.__name__] for base in cls.__mro__ if base.__name__ in kinds), None)


def is_pipeline_sink(processor: "FrameProcessor | None") -> bool:
    # Nested Pipeline sinks can see EndFrame before downstream services finish.
    handler = getattr(processor, "_downstream_push_frame", None)
    owner = getattr(handler, "__self__", None)
    return any(base.__name__ == "PipelineWorker" for base in type(owner).__mro__)


def metadata_from_processor(processor: "FrameProcessor | None") -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    settings = getattr(processor, "_settings", None)
    model = getattr(settings, "model", None) or getattr(processor, "model", None)
    if isinstance(model, str):
        metadata["model"] = model
    provider = _provider_from_processor(processor)
    if provider:
        metadata["provider"] = provider
    return metadata


def _provider_from_processor(processor: "FrameProcessor | None") -> str | None:
    for base in type(processor).__mro__:
        module = base.__module__
        if module.startswith("pipecat.services."):
            provider = module.split(".")[2]
            if provider not in {"llm_service", "stt_service", "tts_service", "ai_service"}:
                return provider_from_name(provider) or provider
    return None


# Request parameters a model call was made with. System instructions are
# logged in response input; provider-specific values live in ``extra``.
_REQUEST_SETTINGS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


def request_settings_from_processor(processor: "FrameProcessor | None") -> dict[str, Any]:
    settings = getattr(processor, "_settings", None)
    candidates = {name: getattr(settings, name, None) for name in _REQUEST_SETTINGS}
    extra = getattr(settings, "extra", None)
    if isinstance(extra, dict):
        candidates.update(extra)
    return {name: value for name, value in candidates.items() if isinstance(value, (bool, int, float, str))}


def provider_from_name(name: str) -> str | None:
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


def tools_schema_to_openai_tools(tools: Any) -> list[dict[str, Any]]:
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
