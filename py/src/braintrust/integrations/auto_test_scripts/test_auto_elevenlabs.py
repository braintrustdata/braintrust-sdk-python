"""Test auto_instrument for ElevenLabs."""

# pylint: disable=import-error

import inspect

from braintrust.auto import auto_instrument
from wrapt import FunctionWrapper


def _is_braintrust_wrapped(target, attr: str) -> bool:
    return isinstance(inspect.getattr_static(target, attr, None), FunctionWrapper)


from elevenlabs.speech_to_speech.client import AsyncSpeechToSpeechClient, SpeechToSpeechClient  # noqa: E402
from elevenlabs.speech_to_text.client import AsyncSpeechToTextClient, SpeechToTextClient  # noqa: E402
from elevenlabs.text_to_sound_effects.client import (  # noqa: E402
    AsyncTextToSoundEffectsClient,
    TextToSoundEffectsClient,
)
from elevenlabs.text_to_speech.client import AsyncTextToSpeechClient, TextToSpeechClient  # noqa: E402


assert not _is_braintrust_wrapped(TextToSpeechClient, "convert")
assert not _is_braintrust_wrapped(TextToSpeechClient, "stream")
assert not _is_braintrust_wrapped(AsyncTextToSpeechClient, "convert")
assert not _is_braintrust_wrapped(AsyncTextToSpeechClient, "stream")
assert not _is_braintrust_wrapped(SpeechToTextClient, "convert")
assert not _is_braintrust_wrapped(SpeechToSpeechClient, "convert")
assert not _is_braintrust_wrapped(TextToSoundEffectsClient, "convert")
assert not _is_braintrust_wrapped(AsyncSpeechToTextClient, "convert")
assert not _is_braintrust_wrapped(AsyncSpeechToSpeechClient, "convert")
assert not _is_braintrust_wrapped(AsyncTextToSoundEffectsClient, "convert")

results = auto_instrument()
assert results.get("elevenlabs") is True
assert _is_braintrust_wrapped(TextToSpeechClient, "convert")
assert _is_braintrust_wrapped(TextToSpeechClient, "stream")
assert _is_braintrust_wrapped(AsyncTextToSpeechClient, "convert")
assert _is_braintrust_wrapped(AsyncTextToSpeechClient, "stream")
assert _is_braintrust_wrapped(SpeechToTextClient, "convert")
assert _is_braintrust_wrapped(SpeechToSpeechClient, "convert")
assert _is_braintrust_wrapped(TextToSoundEffectsClient, "convert")
assert _is_braintrust_wrapped(AsyncSpeechToTextClient, "convert")
assert _is_braintrust_wrapped(AsyncSpeechToSpeechClient, "convert")
assert _is_braintrust_wrapped(AsyncTextToSoundEffectsClient, "convert")

results2 = auto_instrument()
assert results2.get("elevenlabs") is True

print("SUCCESS")
