"""ElevenLabs integration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import MediaPatcher, TextToSpeechPatcher


class ElevenLabsIntegration(BaseIntegration):
    name = "elevenlabs"
    import_names = ("elevenlabs",)
    distribution_names = ("elevenlabs",)
    patchers = (TextToSpeechPatcher, MediaPatcher)
