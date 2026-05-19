# pylint: disable=import-error

from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.elevenlabs import ElevenLabsIntegration, setup_elevenlabs
from braintrust.integrations.elevenlabs.patchers import MediaPatcher, TextToSpeechPatcher
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-project-elevenlabs-py-tracing")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_name(log):
    return log.get("span_attributes", {}).get("name")


def _single_span(logs, name):
    matches = [log for log in logs if _span_name(log) == name]
    assert len(matches) == 1, matches
    return matches[0]


def _audio_file():
    data = (Path(__file__).parents[2] / "fixtures" / "test_audio.wav").read_bytes()
    return ("test_audio.wav", data, "audio/wav")


@pytest.mark.vcr
def test_text_to_speech_convert_traces_audio(memory_logger):
    from elevenlabs.client import ElevenLabs

    assert setup_elevenlabs()
    client = ElevenLabs()

    audio = b"".join(
        client.text_to_speech.convert(
            "JBFqnCBsd6RMkjVDRZzb",
            text="Braintrust instrumentation test.",
            model_id="eleven_multilingual_v2",
        )
    )

    assert audio
    span = _single_span(memory_logger.pop(), "elevenlabs_text_to_speech")
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["voice_id"] == "JBFqnCBsd6RMkjVDRZzb"
    assert span["metadata"]["model"] == "eleven_multilingual_v2"
    assert span["input"]["text"] == "Braintrust instrumentation test."
    assert span["output"]["audio_bytes"] == len(audio)
    assert span["metrics"]["chunk_count"] > 0
    assert span["metrics"]["audio_bytes"] == len(audio)
    assert span["metrics"]["input_characters"] == len("Braintrust instrumentation test.")
    assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_text_to_speech_convert_traces_audio(memory_logger):
    from elevenlabs.client import AsyncElevenLabs

    assert setup_elevenlabs()
    client = AsyncElevenLabs()

    chunks = []
    async for chunk in client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="Braintrust async instrumentation test.",
        model_id="eleven_multilingual_v2",
    ):
        chunks.append(chunk)
    audio = b"".join(chunks)

    assert audio
    span = _single_span(memory_logger.pop(), "elevenlabs_text_to_speech")
    assert span["input"]["text"] == "Braintrust async instrumentation test."
    assert span["metadata"]["model"] == "eleven_multilingual_v2"
    assert span["metrics"]["audio_bytes"] == len(audio)
    assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.vcr
def test_text_to_speech_with_timestamps_traces_response(memory_logger):
    from elevenlabs.client import ElevenLabs

    assert setup_elevenlabs()
    client = ElevenLabs()

    response = client.text_to_speech.convert_with_timestamps(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="Timing test.",
        model_id="eleven_multilingual_v2",
    )

    assert response.audio_base_64
    span = _single_span(memory_logger.pop(), "elevenlabs_text_to_speech_with_timestamps")
    assert span["input"]["text"] == "Timing test."
    assert span["metadata"]["model"] == "eleven_multilingual_v2"
    assert "audio_base_64" in span["output"]
    assert "time_to_first_token" not in span["metrics"]


@pytest.mark.vcr
def test_speech_to_text_convert_traces_response(memory_logger):
    from elevenlabs.client import ElevenLabs

    assert setup_elevenlabs()
    client = ElevenLabs()

    response = client.speech_to_text.convert(model_id="scribe_v1", file=_audio_file())

    assert response.transcription_id
    span = _single_span(memory_logger.pop(), "elevenlabs_speech_to_text")
    assert span["metadata"]["model"] == "scribe_v1"
    assert "model_id" not in span["input"]
    assert "time_to_first_token" not in span["metrics"]
    assert span["input"]["file"]["filename"] == "test_audio.wav"
    assert span["output"]["text"] == response.text


@pytest.mark.vcr
def test_speech_to_speech_convert_traces_audio(memory_logger):
    from elevenlabs.client import ElevenLabs

    assert setup_elevenlabs()
    client = ElevenLabs()

    audio = b"".join(
        client.speech_to_speech.convert(
            "JBFqnCBsd6RMkjVDRZzb",
            audio=_audio_file(),
            model_id="eleven_multilingual_sts_v2",
        )
    )

    assert audio
    span = _single_span(memory_logger.pop(), "elevenlabs_speech_to_speech")
    assert span["metadata"]["model"] == "eleven_multilingual_sts_v2"
    assert span["input"]["audio"]["filename"] == "test_audio.wav"
    assert span["metrics"]["audio_bytes"] == len(audio)
    assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.vcr
def test_text_to_sound_effects_convert_traces_audio(memory_logger):
    from elevenlabs.client import ElevenLabs

    assert setup_elevenlabs()
    client = ElevenLabs()

    audio = b"".join(client.text_to_sound_effects.convert(text="A short soft chime", duration_seconds=0.5))

    assert audio
    span = _single_span(memory_logger.pop(), "elevenlabs_text_to_sound_effects")
    assert span["input"]["text"] == "A short soft chime"
    assert span["metrics"]["audio_bytes"] == len(audio)
    assert span["metrics"]["input_characters"] == len("A short soft chime")
    assert span["metrics"]["time_to_first_token"] >= 0


def test_elevenlabs_integration_exports_patcher():
    assert ElevenLabsIntegration.name == "elevenlabs"
    assert ElevenLabsIntegration.patchers == (TextToSpeechPatcher, MediaPatcher)


def test_auto_instrument_elevenlabs_subprocess():
    pytest.importorskip("elevenlabs")
    verify_autoinstrument_script("test_auto_elevenlabs.py")
