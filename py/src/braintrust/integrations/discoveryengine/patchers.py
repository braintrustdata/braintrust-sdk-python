"""Exact v1 GAPIC targets. No transport, constructor, or v1alpha/v1beta patches."""

from functools import partial

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import _async_call, _call


class AnswerQueryPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.answer_query"
    target_path = "ConversationalSearchServiceClient.answer_query"
    wrapper = partial(_call, "answer_query")


class AsyncAnswerQueryPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.async.answer_query"
    target_path = "ConversationalSearchServiceAsyncClient.answer_query"
    wrapper = partial(_async_call, "answer_query")


class StreamAnswerQueryPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.stream_answer_query"
    target_path = "ConversationalSearchServiceClient.stream_answer_query"
    wrapper = partial(_call, "stream_answer_query")


class AsyncStreamAnswerQueryPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.async.stream_answer_query"
    target_path = "ConversationalSearchServiceAsyncClient.stream_answer_query"
    wrapper = partial(_async_call, "stream_answer_query")


class ConverseConversationPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.converse_conversation"
    target_path = "ConversationalSearchServiceClient.converse_conversation"
    wrapper = partial(_call, "converse_conversation")


class AsyncConverseConversationPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.async.converse_conversation"
    target_path = "ConversationalSearchServiceAsyncClient.converse_conversation"
    wrapper = partial(_async_call, "converse_conversation")


class CheckGroundingPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.check_grounding"
    target_path = "GroundedGenerationServiceClient.check_grounding"
    wrapper = partial(_call, "check_grounding")


class AsyncCheckGroundingPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.async.check_grounding"
    target_path = "GroundedGenerationServiceAsyncClient.check_grounding"
    wrapper = partial(_async_call, "check_grounding")


class RankPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.rank"
    target_path = "RankServiceClient.rank"
    wrapper = partial(_call, "rank")


class AsyncRankPatcher(FunctionWrapperPatcher):
    name = "discoveryengine.async.rank"
    target_path = "RankServiceAsyncClient.rank"
    wrapper = partial(_async_call, "rank")


PATCHERS = (
    AnswerQueryPatcher,
    AsyncAnswerQueryPatcher,
    StreamAnswerQueryPatcher,
    AsyncStreamAnswerQueryPatcher,
    ConverseConversationPatcher,
    AsyncConverseConversationPatcher,
    CheckGroundingPatcher,
    AsyncCheckGroundingPatcher,
    RankPatcher,
    AsyncRankPatcher,
)


def wrap_discoveryengine(client):
    """Instrument one v1 client instance, returning the same client."""
    from google.cloud import discoveryengine_v1

    for patcher in PATCHERS:
        client_type = getattr(discoveryengine_v1, patcher.target_path.split(".")[0])
        if isinstance(client, client_type) and not patcher.is_patched(discoveryengine_v1, None):
            patcher.wrap_target(client)
    return client
