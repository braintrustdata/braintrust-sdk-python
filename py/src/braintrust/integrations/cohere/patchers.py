"""Cohere patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _v2_chat_async_wrapper,
    _v2_chat_stream_async_wrapper,
    _v2_chat_stream_wrapper,
    _v2_chat_wrapper,
    _v2_embed_async_wrapper,
    _v2_embed_wrapper,
    _v2_rerank_async_wrapper,
    _v2_rerank_wrapper,
)


class _V2ChatPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat"
    target_module = "cohere.v2.client"
    target_path = "V2Client.chat"
    wrapper = _v2_chat_wrapper


class _V2ChatAsyncPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat_async"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.chat"
    wrapper = _v2_chat_async_wrapper


class _V2ChatStreamPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat_stream"
    target_module = "cohere.v2.client"
    target_path = "V2Client.chat_stream"
    wrapper = _v2_chat_stream_wrapper


class _V2ChatStreamAsyncPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat_stream_async"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.chat_stream"
    wrapper = _v2_chat_stream_async_wrapper


class V2ChatPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.v2_chat"
    sub_patchers = (
        _V2ChatPatcher,
        _V2ChatAsyncPatcher,
        _V2ChatStreamPatcher,
        _V2ChatStreamAsyncPatcher,
    )


class _V2EmbedPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.embed"
    target_module = "cohere.v2.client"
    target_path = "V2Client.embed"
    wrapper = _v2_embed_wrapper


class _V2EmbedAsyncPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.embed_async"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.embed"
    wrapper = _v2_embed_async_wrapper


class V2EmbedPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.v2_embed"
    sub_patchers = (_V2EmbedPatcher, _V2EmbedAsyncPatcher)


class _V2RerankPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.rerank"
    target_module = "cohere.v2.client"
    target_path = "V2Client.rerank"
    wrapper = _v2_rerank_wrapper


class _V2RerankAsyncPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.rerank_async"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.rerank"
    wrapper = _v2_rerank_async_wrapper


class V2RerankPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.v2_rerank"
    sub_patchers = (_V2RerankPatcher, _V2RerankAsyncPatcher)
