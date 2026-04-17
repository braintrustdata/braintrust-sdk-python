"""Cohere patchers.

Patches live on the two Cohere base clients:

- v1 methods on ``cohere.base_client.BaseCohere`` (sync) and
  ``cohere.base_client.AsyncBaseCohere`` (async). ``cohere.Client`` and
  ``cohere.AsyncClient`` both extend these, so patching the base class covers
  ``cohere.Client()`` and friends.
- v2 methods on ``cohere.v2.client.V2Client`` / ``AsyncV2Client``.

Each patcher also supports ``wrap_target(instance)`` so ``wrap_cohere(client)``
can instrument a single instance without touching global class state.
"""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _async_chat_stream_wrapper,
    _async_chat_wrapper,
    _async_embed_wrapper,
    _async_rerank_wrapper,
    _chat_stream_wrapper,
    _chat_wrapper,
    _embed_wrapper,
    _rerank_wrapper,
)


# ---------------------------------------------------------------------------
# v1 — sync
# ---------------------------------------------------------------------------


class ChatPatcher(FunctionWrapperPatcher):
    name = "cohere.chat"
    target_module = "cohere.base_client"
    target_path = "BaseCohere.chat"
    wrapper = _chat_wrapper


class ChatStreamPatcher(FunctionWrapperPatcher):
    name = "cohere.chat_stream"
    target_module = "cohere.base_client"
    target_path = "BaseCohere.chat_stream"
    wrapper = _chat_stream_wrapper


class EmbedPatcher(FunctionWrapperPatcher):
    name = "cohere.embed"
    target_module = "cohere.base_client"
    target_path = "BaseCohere.embed"
    wrapper = _embed_wrapper


class RerankPatcher(FunctionWrapperPatcher):
    name = "cohere.rerank"
    target_module = "cohere.base_client"
    target_path = "BaseCohere.rerank"
    wrapper = _rerank_wrapper


# ---------------------------------------------------------------------------
# v1 — async
# ---------------------------------------------------------------------------


class AsyncChatPatcher(FunctionWrapperPatcher):
    name = "cohere.async.chat"
    target_module = "cohere.base_client"
    target_path = "AsyncBaseCohere.chat"
    wrapper = _async_chat_wrapper


class AsyncChatStreamPatcher(FunctionWrapperPatcher):
    name = "cohere.async.chat_stream"
    target_module = "cohere.base_client"
    target_path = "AsyncBaseCohere.chat_stream"
    wrapper = _async_chat_stream_wrapper


class AsyncEmbedPatcher(FunctionWrapperPatcher):
    name = "cohere.async.embed"
    target_module = "cohere.base_client"
    target_path = "AsyncBaseCohere.embed"
    wrapper = _async_embed_wrapper


class AsyncRerankPatcher(FunctionWrapperPatcher):
    name = "cohere.async.rerank"
    target_module = "cohere.base_client"
    target_path = "AsyncBaseCohere.rerank"
    wrapper = _async_rerank_wrapper


# ---------------------------------------------------------------------------
# v2 — sync
# ---------------------------------------------------------------------------


class V2ChatPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat"
    target_module = "cohere.v2.client"
    target_path = "V2Client.chat"
    wrapper = _chat_wrapper


class V2ChatStreamPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.chat_stream"
    target_module = "cohere.v2.client"
    target_path = "V2Client.chat_stream"
    wrapper = _chat_stream_wrapper


class V2EmbedPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.embed"
    target_module = "cohere.v2.client"
    target_path = "V2Client.embed"
    wrapper = _embed_wrapper


class V2RerankPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.rerank"
    target_module = "cohere.v2.client"
    target_path = "V2Client.rerank"
    wrapper = _rerank_wrapper


# ---------------------------------------------------------------------------
# v2 — async
# ---------------------------------------------------------------------------


class AsyncV2ChatPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.async.chat"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.chat"
    wrapper = _async_chat_wrapper


class AsyncV2ChatStreamPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.async.chat_stream"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.chat_stream"
    wrapper = _async_chat_stream_wrapper


class AsyncV2EmbedPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.async.embed"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.embed"
    wrapper = _async_embed_wrapper


class AsyncV2RerankPatcher(FunctionWrapperPatcher):
    name = "cohere.v2.async.rerank"
    target_module = "cohere.v2.client"
    target_path = "AsyncV2Client.rerank"
    wrapper = _async_rerank_wrapper


# ---------------------------------------------------------------------------
# Composite patchers — group all sync/async variants by execution surface.
# ---------------------------------------------------------------------------


class CohereChatPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.chat.all"
    sub_patchers = (
        ChatPatcher,
        AsyncChatPatcher,
        V2ChatPatcher,
        AsyncV2ChatPatcher,
    )


class CohereChatStreamPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.chat_stream.all"
    sub_patchers = (
        ChatStreamPatcher,
        AsyncChatStreamPatcher,
        V2ChatStreamPatcher,
        AsyncV2ChatStreamPatcher,
    )


class CohereEmbedPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.embed.all"
    sub_patchers = (
        EmbedPatcher,
        AsyncEmbedPatcher,
        V2EmbedPatcher,
        AsyncV2EmbedPatcher,
    )


class CohereRerankPatcher(CompositeFunctionWrapperPatcher):
    name = "cohere.rerank.all"
    sub_patchers = (
        RerankPatcher,
        AsyncRerankPatcher,
        V2RerankPatcher,
        AsyncV2RerankPatcher,
    )
