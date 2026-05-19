"""ElevenLabs SDK patchers."""

from typing import Any

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    traced_async_sound_effects,
    traced_async_speech_to_speech,
    traced_async_speech_to_text,
    traced_async_tts,
    traced_async_tts_with_timestamps,
    traced_sound_effects,
    traced_speech_to_speech,
    traced_speech_to_text,
    traced_tts,
    traced_tts_with_timestamps,
)


class TextToSpeechConvertPatcher(FunctionWrapperPatcher):
    name = "text_to_speech_convert"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "TextToSpeechClient.convert"
    wrapper = staticmethod(traced_tts)


class TextToSpeechStreamPatcher(FunctionWrapperPatcher):
    name = "text_to_speech_stream"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "TextToSpeechClient.stream"
    wrapper = staticmethod(traced_tts)


class TextToSpeechConvertWithTimestampsPatcher(FunctionWrapperPatcher):
    name = "text_to_speech_convert_with_timestamps"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "TextToSpeechClient.convert_with_timestamps"
    wrapper = staticmethod(traced_tts_with_timestamps)


class TextToSpeechStreamWithTimestampsPatcher(FunctionWrapperPatcher):
    name = "text_to_speech_stream_with_timestamps"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "TextToSpeechClient.stream_with_timestamps"
    wrapper = staticmethod(traced_tts)


class AsyncTextToSpeechConvertPatcher(FunctionWrapperPatcher):
    name = "async_text_to_speech_convert"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "AsyncTextToSpeechClient.convert"
    wrapper = staticmethod(traced_async_tts)


class AsyncTextToSpeechStreamPatcher(FunctionWrapperPatcher):
    name = "async_text_to_speech_stream"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "AsyncTextToSpeechClient.stream"
    wrapper = staticmethod(traced_async_tts)


class AsyncTextToSpeechConvertWithTimestampsPatcher(FunctionWrapperPatcher):
    name = "async_text_to_speech_convert_with_timestamps"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "AsyncTextToSpeechClient.convert_with_timestamps"
    wrapper = staticmethod(traced_async_tts_with_timestamps)


class AsyncTextToSpeechStreamWithTimestampsPatcher(FunctionWrapperPatcher):
    name = "async_text_to_speech_stream_with_timestamps"
    target_module = "elevenlabs.text_to_speech.client"
    target_path = "AsyncTextToSpeechClient.stream_with_timestamps"
    wrapper = staticmethod(traced_async_tts)


class TextToSpeechPatcher(CompositeFunctionWrapperPatcher):
    name = "text_to_speech"
    sub_patchers = (
        TextToSpeechConvertPatcher,
        TextToSpeechStreamPatcher,
        TextToSpeechConvertWithTimestampsPatcher,
        TextToSpeechStreamWithTimestampsPatcher,
        AsyncTextToSpeechConvertPatcher,
        AsyncTextToSpeechStreamPatcher,
        AsyncTextToSpeechConvertWithTimestampsPatcher,
        AsyncTextToSpeechStreamWithTimestampsPatcher,
    )


class SpeechToTextPatcher(FunctionWrapperPatcher):
    name = "speech_to_text"
    target_module = "elevenlabs.speech_to_text.client"
    target_path = "SpeechToTextClient.convert"
    wrapper = staticmethod(traced_speech_to_text)


class AsyncSpeechToTextPatcher(FunctionWrapperPatcher):
    name = "async_speech_to_text"
    target_module = "elevenlabs.speech_to_text.client"
    target_path = "AsyncSpeechToTextClient.convert"
    wrapper = staticmethod(traced_async_speech_to_text)


class SpeechToSpeechPatcher(FunctionWrapperPatcher):
    name = "speech_to_speech"
    target_module = "elevenlabs.speech_to_speech.client"
    target_path = "SpeechToSpeechClient.convert"
    wrapper = staticmethod(traced_speech_to_speech)


class SpeechToSpeechStreamPatcher(FunctionWrapperPatcher):
    name = "speech_to_speech_stream"
    target_module = "elevenlabs.speech_to_speech.client"
    target_path = "SpeechToSpeechClient.stream"
    wrapper = staticmethod(traced_speech_to_speech)


class AsyncSpeechToSpeechPatcher(FunctionWrapperPatcher):
    name = "async_speech_to_speech"
    target_module = "elevenlabs.speech_to_speech.client"
    target_path = "AsyncSpeechToSpeechClient.convert"
    wrapper = staticmethod(traced_async_speech_to_speech)


class AsyncSpeechToSpeechStreamPatcher(FunctionWrapperPatcher):
    name = "async_speech_to_speech_stream"
    target_module = "elevenlabs.speech_to_speech.client"
    target_path = "AsyncSpeechToSpeechClient.stream"
    wrapper = staticmethod(traced_async_speech_to_speech)


class TextToSoundEffectsPatcher(FunctionWrapperPatcher):
    name = "text_to_sound_effects"
    target_module = "elevenlabs.text_to_sound_effects.client"
    target_path = "TextToSoundEffectsClient.convert"
    wrapper = staticmethod(traced_sound_effects)


class AsyncTextToSoundEffectsPatcher(FunctionWrapperPatcher):
    name = "async_text_to_sound_effects"
    target_module = "elevenlabs.text_to_sound_effects.client"
    target_path = "AsyncTextToSoundEffectsClient.convert"
    wrapper = staticmethod(traced_async_sound_effects)


class MediaPatcher(CompositeFunctionWrapperPatcher):
    name = "media"
    sub_patchers = (
        SpeechToTextPatcher,
        AsyncSpeechToTextPatcher,
        SpeechToSpeechPatcher,
        SpeechToSpeechStreamPatcher,
        AsyncSpeechToSpeechPatcher,
        AsyncSpeechToSpeechStreamPatcher,
        TextToSoundEffectsPatcher,
        AsyncTextToSoundEffectsPatcher,
    )


def wrap_elevenlabs(target: Any) -> Any:
    """Instrument an ElevenLabs SDK class or instance directly."""
    TextToSpeechPatcher.wrap_target(target)
    MediaPatcher.wrap_target(target)
    return target
