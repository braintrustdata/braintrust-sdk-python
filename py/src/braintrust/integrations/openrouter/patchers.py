"""OpenRouter patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _chat_send_async_wrapper,
    _chat_send_wrapper,
    _embeddings_generate_async_wrapper,
    _embeddings_generate_wrapper,
    _responses_send_async_wrapper,
    _responses_send_wrapper,
)


class ChatSendPatcher(FunctionWrapperPatcher):
    name = "openrouter.chat.send"
    target_module = "openrouter.chat"
    target_path = "Chat.send"
    wrapper = _chat_send_wrapper


class ChatSendAsyncPatcher(FunctionWrapperPatcher):
    name = "openrouter.chat.send_async"
    target_module = "openrouter.chat"
    target_path = "Chat.send_async"
    wrapper = _chat_send_async_wrapper


class ChatPatcher(CompositeFunctionWrapperPatcher):
    name = "openrouter.chat"
    sub_patchers = (ChatSendPatcher, ChatSendAsyncPatcher)


class EmbeddingsGeneratePatcher(FunctionWrapperPatcher):
    name = "openrouter.embeddings.generate"
    target_module = "openrouter.embeddings"
    target_path = "Embeddings.generate"
    wrapper = _embeddings_generate_wrapper


class EmbeddingsGenerateAsyncPatcher(FunctionWrapperPatcher):
    name = "openrouter.embeddings.generate_async"
    target_module = "openrouter.embeddings"
    target_path = "Embeddings.generate_async"
    wrapper = _embeddings_generate_async_wrapper


class EmbeddingsPatcher(CompositeFunctionWrapperPatcher):
    name = "openrouter.embeddings"
    sub_patchers = (EmbeddingsGeneratePatcher, EmbeddingsGenerateAsyncPatcher)


class ResponsesSendPatcher(FunctionWrapperPatcher):
    name = "openrouter.beta.responses.send"
    target_module = "openrouter.responses"
    target_path = "Responses.send"
    wrapper = _responses_send_wrapper


class ResponsesSendAsyncPatcher(FunctionWrapperPatcher):
    name = "openrouter.beta.responses.send_async"
    target_module = "openrouter.responses"
    target_path = "Responses.send_async"
    wrapper = _responses_send_async_wrapper


class ResponsesPatcher(CompositeFunctionWrapperPatcher):
    name = "openrouter.beta.responses"
    sub_patchers = (ResponsesSendPatcher, ResponsesSendAsyncPatcher)
