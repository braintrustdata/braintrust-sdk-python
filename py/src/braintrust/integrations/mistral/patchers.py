"""Mistral patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agents_complete_async_wrapper,
    _agents_complete_wrapper,
    _agents_stream_async_wrapper,
    _agents_stream_wrapper,
    _chat_complete_async_wrapper,
    _chat_complete_wrapper,
    _chat_stream_async_wrapper,
    _chat_stream_wrapper,
    _embeddings_create_async_wrapper,
    _embeddings_create_wrapper,
    _fim_complete_async_wrapper,
    _fim_complete_wrapper,
    _fim_stream_async_wrapper,
    _fim_stream_wrapper,
    _ocr_process_async_wrapper,
    _ocr_process_wrapper,
    _speech_complete_async_wrapper,
    _speech_complete_wrapper,
    _transcriptions_complete_async_wrapper,
    _transcriptions_complete_wrapper,
    _transcriptions_stream_async_wrapper,
    _transcriptions_stream_wrapper,
)


class _ChatCompleteV2Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.complete.v2"
    target_module = "mistralai.client.chat"
    target_path = "Chat.complete"
    wrapper = _chat_complete_wrapper


class _ChatCompleteV1Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.complete.v1"
    target_module = "mistralai.chat"
    target_path = "Chat.complete"
    wrapper = _chat_complete_wrapper
    superseded_by = (_ChatCompleteV2Patcher,)


class _ChatCompleteAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.complete_async.v2"
    target_module = "mistralai.client.chat"
    target_path = "Chat.complete_async"
    wrapper = _chat_complete_async_wrapper


class _ChatCompleteAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.complete_async.v1"
    target_module = "mistralai.chat"
    target_path = "Chat.complete_async"
    wrapper = _chat_complete_async_wrapper
    superseded_by = (_ChatCompleteAsyncV2Patcher,)


class _ChatStreamV2Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.stream.v2"
    target_module = "mistralai.client.chat"
    target_path = "Chat.stream"
    wrapper = _chat_stream_wrapper


class _ChatStreamV1Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.stream.v1"
    target_module = "mistralai.chat"
    target_path = "Chat.stream"
    wrapper = _chat_stream_wrapper
    superseded_by = (_ChatStreamV2Patcher,)


class _ChatStreamAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.stream_async.v2"
    target_module = "mistralai.client.chat"
    target_path = "Chat.stream_async"
    wrapper = _chat_stream_async_wrapper


class _ChatStreamAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.chat.stream_async.v1"
    target_module = "mistralai.chat"
    target_path = "Chat.stream_async"
    wrapper = _chat_stream_async_wrapper
    superseded_by = (_ChatStreamAsyncV2Patcher,)


class ChatPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.chat"
    sub_patchers = (
        _ChatCompleteV2Patcher,
        _ChatCompleteV1Patcher,
        _ChatCompleteAsyncV2Patcher,
        _ChatCompleteAsyncV1Patcher,
        _ChatStreamV2Patcher,
        _ChatStreamV1Patcher,
        _ChatStreamAsyncV2Patcher,
        _ChatStreamAsyncV1Patcher,
    )


class _EmbeddingsCreateV2Patcher(FunctionWrapperPatcher):
    name = "mistral.embeddings.create.v2"
    target_module = "mistralai.client.embeddings"
    target_path = "Embeddings.create"
    wrapper = _embeddings_create_wrapper


class _EmbeddingsCreateV1Patcher(FunctionWrapperPatcher):
    name = "mistral.embeddings.create.v1"
    target_module = "mistralai.embeddings"
    target_path = "Embeddings.create"
    wrapper = _embeddings_create_wrapper
    superseded_by = (_EmbeddingsCreateV2Patcher,)


class _EmbeddingsCreateAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.embeddings.create_async.v2"
    target_module = "mistralai.client.embeddings"
    target_path = "Embeddings.create_async"
    wrapper = _embeddings_create_async_wrapper


class _EmbeddingsCreateAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.embeddings.create_async.v1"
    target_module = "mistralai.embeddings"
    target_path = "Embeddings.create_async"
    wrapper = _embeddings_create_async_wrapper
    superseded_by = (_EmbeddingsCreateAsyncV2Patcher,)


class EmbeddingsPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.embeddings"
    sub_patchers = (
        _EmbeddingsCreateV2Patcher,
        _EmbeddingsCreateV1Patcher,
        _EmbeddingsCreateAsyncV2Patcher,
        _EmbeddingsCreateAsyncV1Patcher,
    )


class _TranscriptionsCompleteV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.complete.v2"
    target_module = "mistralai.client.transcriptions"
    target_path = "Transcriptions.complete"
    wrapper = _transcriptions_complete_wrapper


class _TranscriptionsCompleteV1Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.complete.v1"
    target_module = "mistralai.transcriptions"
    target_path = "Transcriptions.complete"
    wrapper = _transcriptions_complete_wrapper
    superseded_by = (_TranscriptionsCompleteV2Patcher,)


class _TranscriptionsCompleteAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.complete_async.v2"
    target_module = "mistralai.client.transcriptions"
    target_path = "Transcriptions.complete_async"
    wrapper = _transcriptions_complete_async_wrapper


class _TranscriptionsCompleteAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.complete_async.v1"
    target_module = "mistralai.transcriptions"
    target_path = "Transcriptions.complete_async"
    wrapper = _transcriptions_complete_async_wrapper
    superseded_by = (_TranscriptionsCompleteAsyncV2Patcher,)


class _TranscriptionsStreamV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.stream.v2"
    target_module = "mistralai.client.transcriptions"
    target_path = "Transcriptions.stream"
    wrapper = _transcriptions_stream_wrapper


class _TranscriptionsStreamV1Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.stream.v1"
    target_module = "mistralai.transcriptions"
    target_path = "Transcriptions.stream"
    wrapper = _transcriptions_stream_wrapper
    superseded_by = (_TranscriptionsStreamV2Patcher,)


class _TranscriptionsStreamAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.stream_async.v2"
    target_module = "mistralai.client.transcriptions"
    target_path = "Transcriptions.stream_async"
    wrapper = _transcriptions_stream_async_wrapper


class _TranscriptionsStreamAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.transcriptions.stream_async.v1"
    target_module = "mistralai.transcriptions"
    target_path = "Transcriptions.stream_async"
    wrapper = _transcriptions_stream_async_wrapper
    superseded_by = (_TranscriptionsStreamAsyncV2Patcher,)


class TranscriptionsPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.audio.transcriptions"
    sub_patchers = (
        _TranscriptionsCompleteV2Patcher,
        _TranscriptionsCompleteV1Patcher,
        _TranscriptionsCompleteAsyncV2Patcher,
        _TranscriptionsCompleteAsyncV1Patcher,
        _TranscriptionsStreamV2Patcher,
        _TranscriptionsStreamV1Patcher,
        _TranscriptionsStreamAsyncV2Patcher,
        _TranscriptionsStreamAsyncV1Patcher,
    )


class _SpeechCompleteV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.speech.complete.v2"
    target_module = "mistralai.client.speech"
    target_path = "Speech.complete"
    wrapper = _speech_complete_wrapper


class _SpeechCompleteAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.audio.speech.complete_async.v2"
    target_module = "mistralai.client.speech"
    target_path = "Speech.complete_async"
    wrapper = _speech_complete_async_wrapper


class SpeechPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.audio.speech"
    sub_patchers = (
        _SpeechCompleteV2Patcher,
        _SpeechCompleteAsyncV2Patcher,
    )


class _FimCompleteV2Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.complete.v2"
    target_module = "mistralai.client.fim"
    target_path = "Fim.complete"
    wrapper = _fim_complete_wrapper


class _FimCompleteV1Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.complete.v1"
    target_module = "mistralai.fim"
    target_path = "Fim.complete"
    wrapper = _fim_complete_wrapper
    superseded_by = (_FimCompleteV2Patcher,)


class _FimCompleteAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.complete_async.v2"
    target_module = "mistralai.client.fim"
    target_path = "Fim.complete_async"
    wrapper = _fim_complete_async_wrapper


class _FimCompleteAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.complete_async.v1"
    target_module = "mistralai.fim"
    target_path = "Fim.complete_async"
    wrapper = _fim_complete_async_wrapper
    superseded_by = (_FimCompleteAsyncV2Patcher,)


class _FimStreamV2Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.stream.v2"
    target_module = "mistralai.client.fim"
    target_path = "Fim.stream"
    wrapper = _fim_stream_wrapper


class _FimStreamV1Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.stream.v1"
    target_module = "mistralai.fim"
    target_path = "Fim.stream"
    wrapper = _fim_stream_wrapper
    superseded_by = (_FimStreamV2Patcher,)


class _FimStreamAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.stream_async.v2"
    target_module = "mistralai.client.fim"
    target_path = "Fim.stream_async"
    wrapper = _fim_stream_async_wrapper


class _FimStreamAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.fim.stream_async.v1"
    target_module = "mistralai.fim"
    target_path = "Fim.stream_async"
    wrapper = _fim_stream_async_wrapper
    superseded_by = (_FimStreamAsyncV2Patcher,)


class FimPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.fim"
    sub_patchers = (
        _FimCompleteV2Patcher,
        _FimCompleteV1Patcher,
        _FimCompleteAsyncV2Patcher,
        _FimCompleteAsyncV1Patcher,
        _FimStreamV2Patcher,
        _FimStreamV1Patcher,
        _FimStreamAsyncV2Patcher,
        _FimStreamAsyncV1Patcher,
    )


class _AgentsCompleteV2Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.complete.v2"
    target_module = "mistralai.client.agents"
    target_path = "Agents.complete"
    wrapper = _agents_complete_wrapper


class _AgentsCompleteV1Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.complete.v1"
    target_module = "mistralai.agents"
    target_path = "Agents.complete"
    wrapper = _agents_complete_wrapper
    superseded_by = (_AgentsCompleteV2Patcher,)


class _AgentsCompleteAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.complete_async.v2"
    target_module = "mistralai.client.agents"
    target_path = "Agents.complete_async"
    wrapper = _agents_complete_async_wrapper


class _AgentsCompleteAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.complete_async.v1"
    target_module = "mistralai.agents"
    target_path = "Agents.complete_async"
    wrapper = _agents_complete_async_wrapper
    superseded_by = (_AgentsCompleteAsyncV2Patcher,)


class _AgentsStreamV2Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.stream.v2"
    target_module = "mistralai.client.agents"
    target_path = "Agents.stream"
    wrapper = _agents_stream_wrapper


class _AgentsStreamV1Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.stream.v1"
    target_module = "mistralai.agents"
    target_path = "Agents.stream"
    wrapper = _agents_stream_wrapper
    superseded_by = (_AgentsStreamV2Patcher,)


class _AgentsStreamAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.stream_async.v2"
    target_module = "mistralai.client.agents"
    target_path = "Agents.stream_async"
    wrapper = _agents_stream_async_wrapper


class _AgentsStreamAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.agents.stream_async.v1"
    target_module = "mistralai.agents"
    target_path = "Agents.stream_async"
    wrapper = _agents_stream_async_wrapper
    superseded_by = (_AgentsStreamAsyncV2Patcher,)


class AgentsPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.agents"
    sub_patchers = (
        _AgentsCompleteV2Patcher,
        _AgentsCompleteV1Patcher,
        _AgentsCompleteAsyncV2Patcher,
        _AgentsCompleteAsyncV1Patcher,
        _AgentsStreamV2Patcher,
        _AgentsStreamV1Patcher,
        _AgentsStreamAsyncV2Patcher,
        _AgentsStreamAsyncV1Patcher,
    )


class _OcrProcessV2Patcher(FunctionWrapperPatcher):
    name = "mistral.ocr.process.v2"
    target_module = "mistralai.client.ocr"
    target_path = "Ocr.process"
    wrapper = _ocr_process_wrapper


class _OcrProcessV1Patcher(FunctionWrapperPatcher):
    name = "mistral.ocr.process.v1"
    target_module = "mistralai.ocr"
    target_path = "Ocr.process"
    wrapper = _ocr_process_wrapper
    superseded_by = (_OcrProcessV2Patcher,)


class _OcrProcessAsyncV2Patcher(FunctionWrapperPatcher):
    name = "mistral.ocr.process_async.v2"
    target_module = "mistralai.client.ocr"
    target_path = "Ocr.process_async"
    wrapper = _ocr_process_async_wrapper


class _OcrProcessAsyncV1Patcher(FunctionWrapperPatcher):
    name = "mistral.ocr.process_async.v1"
    target_module = "mistralai.ocr"
    target_path = "Ocr.process_async"
    wrapper = _ocr_process_async_wrapper
    superseded_by = (_OcrProcessAsyncV2Patcher,)


class OcrPatcher(CompositeFunctionWrapperPatcher):
    name = "mistral.ocr"
    sub_patchers = (
        _OcrProcessV2Patcher,
        _OcrProcessV1Patcher,
        _OcrProcessAsyncV2Patcher,
        _OcrProcessAsyncV1Patcher,
    )
