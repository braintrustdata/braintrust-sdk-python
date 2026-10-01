"""Internal native voice tracing installed by the Pipecat integration."""

from .instrumentation import NativeObserver as VoiceObserver


__all__ = ["VoiceObserver"]
