"""Braintrust integration for ElevenLabs."""

from .integration import ElevenLabsIntegration
from .patchers import wrap_elevenlabs


def setup_elevenlabs() -> bool:
    """Instrument the ElevenLabs Python SDK."""
    return ElevenLabsIntegration.setup()


__all__ = ["ElevenLabsIntegration", "setup_elevenlabs", "wrap_elevenlabs"]
