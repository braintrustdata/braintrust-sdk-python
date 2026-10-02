"""Discover a single native voice path without integration-specific setup."""

from types import SimpleNamespace


def _is(processor, name):
    return any(cls.__name__ == name and cls.__module__.startswith("pipecat.") for cls in type(processor).__mro__)


def discover(pipeline):
    # Parallel/nested paths need explicit ownership; don't guess between them.
    processors = list(getattr(pipeline, "processors", ()))
    if any(getattr(processor, "processors", ()) for processor in processors):
        return None
    names = ("BaseInputTransport", "BaseOutputTransport", "LLMUserAggregator", "LLMAssistantAggregator")
    matches = [[processor for processor in processors if _is(processor, name)] for name in names]
    if any(len(match) != 1 for match in matches):
        return None
    source, destination, user, assistant = (match[0] for match in matches)
    stts = [processor for processor in processors if _is(processor, "SegmentedSTTService")]
    realtime = [processor for processor in processors if _is(processor, "OpenAIRealtimeLLMService")]
    if len(stts) + len(realtime) != 1:
        return None
    return dict(
        transport=SimpleNamespace(input=lambda: source, output=lambda: destination),
        user_aggregator=user,
        assistant_aggregator=assistant,
        stt=stts[0] if stts else None,
        realtime_service=realtime[0] if realtime else None,
    )
