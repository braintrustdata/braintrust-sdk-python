"""Bounded compatibility with native Pipecat execution implementations.

Pipecat 1.3/1.16 have no public executing-client accessor. Only inspect the
legacy client store after matching the complete known execution route. An
inherited unused client or an overridden request method proves nothing.
"""

import inspect
import sys
from typing import Any


# operation, implementation module/class, method route[, client store].
# Identity checks allow inherited implementations, never application overrides.
_EXECUTIONS = (
    (
        "llm_response",
        "openai.base_llm",
        "BaseOpenAILLMService",
        ("process_frame", "_process_context", "get_chat_completions"),
    ),
    (
        "llm_response",
        "anthropic.llm",
        "AnthropicLLMService",
        ("process_frame", "_process_context", "_create_message_stream"),
    ),
    ("llm_response", "google.llm", "GoogleLLMService", ("process_frame", "_process_context", "_stream_content")),
    # Bedrock creates its runtime client per request from this session.
    ("llm_response", "aws.llm", "AWSBedrockLLMService", ("process_frame", "_process_context"), "_aws_session"),
    ("tts", "openai.tts", "OpenAITTSService", ("run_tts",)),
    ("tts", "google.tts", "GeminiTTSService", ("run_tts",)),
)

# These native operations execute transports without a covered model SDK.
# Deliberately list concrete execution methods, not provider labels.
_TRANSPORTS = (
    ("tts", "cartesia.tts", "CartesiaTTSService"),
    ("tts", "cartesia.tts", "CartesiaHttpTTSService"),
    ("tts", "elevenlabs.tts", "ElevenLabsTTSService"),
    ("tts", "elevenlabs.tts", "ElevenLabsHttpTTSService"),
    ("tts", "google.tts", "GoogleTTSService"),
    ("tts", "google.tts", "GoogleHttpTTSService"),
    ("stt", "deepgram.stt", "DeepgramSTTService"),
    ("stt", "elevenlabs.stt", "ElevenLabsSTTService"),
    ("stt", "elevenlabs.stt", "ElevenLabsRealtimeSTTService"),
    ("stt", "google.stt", "GoogleSTTService"),
    ("stt", "openai.stt", "OpenAIRealtimeSTTService"),
    # The Realtime LLM service talks to its websocket directly, not the SDK.
    (
        "llm_response",
        "openai.realtime.llm",
        "OpenAIRealtimeLLMService",
        ("process_frame", "_handle_evt_response_done"),
    ),
)


def _implementation(module: str, name: str) -> Any:
    # The service's actual class has already loaded its defining module. Do not
    # import optional providers just to classify another service.
    return getattr(sys.modules.get("pipecat.services." + module), name, None)


def _function(method: Any) -> Any:
    method = inspect.unwrap(method)
    return getattr(method, "__func__", method)


def _matches(service: Any, cls: Any, methods: tuple[str, ...]) -> bool:
    return cls is not None and all(
        callable(expected := getattr(cls, name, None))
        and _function(getattr(service, name, None)) is _function(expected)
        for name in methods
    )


def _tts_dispatch_matches(service: Any, cls: Any) -> bool:
    # The public dispatchers and their known version-specific execution path
    # must agree too: an inherited but unused run_tts proves no ownership.
    methods = tuple(
        name
        for name in (
            "process_frame",
            "tts_process_generator",
            "_handle_tts_speak",
            "_handle_text",
            "_handle_aggregated_text",
            "_speak_text",
            "_synthesize_text",
            "_push_tts_frames",
        )
        if callable(getattr(cls, name, None))
    )
    return cls is not None and _matches(service, cls, methods)


def executing_client(service: Any, operation: str) -> tuple[bool, Any]:
    for kind, module, name, methods, *store in _EXECUTIONS:
        cls = _implementation(module, name)
        if (
            operation == kind
            and _matches(service, cls, methods)
            and (kind != "tts" or _tts_dispatch_matches(service, cls))
        ):
            return True, vars(service).get(store[0] if store else "_client")
    if operation == "stt" and all(
        _matches(service, _implementation(module, name), methods)
        for module, name, methods in (
            ("stt_service", "SegmentedSTTService", ("process_frame",)),
            ("whisper.base_stt", "BaseWhisperSTTService", ("run_stt",)),
            ("openai.stt", "OpenAISTTService", ("_transcribe",)),
        )
    ):
        return True, vars(service).get("_client")
    for kind, module, name, *methods in _TRANSPORTS:
        cls = _implementation(module, name)
        if (
            operation == kind
            and _matches(service, cls, methods[0] if methods else ("process_frame", "run_" + kind))
            and (kind != "tts" or _tts_dispatch_matches(service, cls))
        ):
            return True, None
    return False, None
