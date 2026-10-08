"""
Tests to ensure we reliably wrap the Anthropic API.
"""

import inspect
import json
import os
import time
import unittest.mock
from types import SimpleNamespace

import anthropic
import pytest
from braintrust import Attachment, logger
from braintrust.integrations.anthropic import AnthropicIntegration, wrap_anthropic
from braintrust.integrations.anthropic._utils import _try_to_dict, extract_anthropic_usage
from braintrust.integrations.anthropic.tracing import (
    TracedMessageStream,
    _log_message_to_span,
)
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger
from pydantic import BaseModel


PROJECT_NAME = "test-anthropic-app"
LEGACY_MODEL = "claude-3-haiku-20240307"
LATEST_MODEL = "claude-haiku-4-5-20251001"
MODEL = LATEST_MODEL if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest" else LEGACY_MODEL
MULTIMODAL_MODEL = "claude-haiku-4-5-20251001"
STRUCTURED_OUTPUT_MODEL = "claude-haiku-4-5"
STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "label": {"type": "string"},
    },
    "required": ["answer", "label"],
    "additionalProperties": False,
}
STRUCTURED_OUTPUT_TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate simple arithmetic expressions.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
]
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
PDF_BASE64 = "JVBERi0xLjAKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PmVuZG9iagoyIDAgb2JqCjw8L1R5cGUvUGFnZXMvS2lkc1szIDAgUl0vQ291bnQgMT4+ZW5kb2JqCjMgMCBvYmoKPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCA2MTIgNzkyXT4+ZW5kb2JqCnhyZWYKMCA0CjAwMDAwMDAwMDAgNjU1MzUgZg0KMDAwMDAwMDAxMCAwMDAwMCBuDQowMDAwMDAwMDUzIDAwMDAwIG4NCjAwMDAwMDAxMDIgMDAwMDAgbg0KdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoxNDkKJUVPRg=="
PROMPT_CACHE_TEST_TEXT = "\n".join(f"Cached geography fact {i}: Paris is the capital of France." for i in range(300))


def _get_client():
    return anthropic.Anthropic()


def _get_async_client():
    return anthropic.AsyncAnthropic()


class ParsedAnswer(BaseModel):
    answer: int
    label: str


def _get_supported_structured_output_param():
    parameter_names = inspect.signature(_get_client().messages.create).parameters
    if "output_config" in parameter_names:
        return (
            "output_config",
            {
                "format": {
                    "type": "json_schema",
                    "schema": STRUCTURED_OUTPUT_SCHEMA,
                }
            },
        )
    if "output_format" in parameter_names:
        return (
            "output_format",
            {
                "type": "json_schema",
                "schema": STRUCTURED_OUTPUT_SCHEMA,
            },
        )
    pytest.skip("Installed anthropic SDK does not support structured outputs parameters")


def _skip_if_messages_parse_unsupported():
    if not hasattr(_get_client().messages, "parse"):
        pytest.skip("Installed anthropic SDK does not support messages.parse")


def test_wrap_anthropic_preserves_messages_parse_feature_detection():
    class MessagesWithoutParse:
        pass

    class FakeAnthropic:
        messages = MessagesWithoutParse()

    class FakeAsyncAnthropic:
        messages = MessagesWithoutParse()

    for client in (FakeAnthropic(), FakeAsyncAnthropic()):
        messages = wrap_anthropic(client).messages
        assert not hasattr(messages, "parse")
        with pytest.raises(AttributeError):
            getattr(messages, "parse")


def _skip_if_server_tool_content_blocks_unsupported():
    required_type_names = ("ServerToolUseBlock", "WebSearchToolResultBlock")
    if not all(hasattr(anthropic.types, type_name) for type_name in required_type_names):
        pytest.skip("Installed anthropic SDK does not support Anthropic server tool content blocks")


def _skip_if_managed_agents_unsupported():
    client = _get_client()
    if not hasattr(client.beta, "agents"):
        pytest.skip("Installed anthropic SDK does not support beta managed agents")
    if not hasattr(client.beta, "sessions"):
        pytest.skip("Installed anthropic SDK does not support beta managed agent sessions")
    if not hasattr(client.beta.sessions, "events") or not hasattr(client.beta.sessions.events, "send"):
        pytest.skip("Installed anthropic SDK does not support beta managed agent session events")


_MANAGED_AGENTS_EVENTS_PROMPT = "Use bash once to print 2+2, then reply with only the number."
_MANAGED_AGENTS_AGENT_NAME = "braintrust-sdk-managed-agent"
_MANAGED_AGENTS_BASH_AGENT_NAME = "braintrust-sdk-managed-agent-bash"
_MANAGED_AGENTS_BASH_SYSTEM_PROMPT = (
    "For arithmetic requests, use exactly one bash command and then answer with only the numeric result."
)


def _get_managed_agents_environment_id(client):
    environments = client.beta.environments.list(limit=1)
    for environment in environments:
        return environment.id
    pytest.skip("No Anthropic managed-agent environment available for re-recording")


def _create_managed_agent(client, *, with_bash: bool = False):
    create_kwargs = {
        "model": "claude-haiku-4-5",
        "name": _MANAGED_AGENTS_BASH_AGENT_NAME if with_bash else _MANAGED_AGENTS_AGENT_NAME,
        "description": "Does math",
        "tools": [],
    }
    if with_bash:
        create_kwargs["description"] = "Uses bash for a single arithmetic command"
        create_kwargs["system"] = _MANAGED_AGENTS_BASH_SYSTEM_PROMPT
        create_kwargs["tools"] = [
            {
                "type": "agent_toolset_20260401",
                "default_config": {"enabled": False},
                "configs": [
                    {"name": "bash", "enabled": True, "permission_policy": {"type": "always_allow"}},
                ],
            }
        ]

    return client.beta.agents.create(**create_kwargs)


def _cleanup_managed_agent_resources(client, agent_id: str | None = None, session_id: str | None = None):
    if session_id:
        client.beta.sessions.delete(session_id)
    if agent_id and hasattr(client.beta.agents, "archive"):
        client.beta.agents.archive(agent_id)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def test_log_message_to_span_includes_stop_reason_and_stop_sequence():
    span = unittest.mock.MagicMock()
    message = SimpleNamespace(
        role="assistant",
        content=[{"type": "text", "text": "done"}],
        model=MODEL,
        stop_reason="stop_sequence",
        stop_sequence="DONE",
        stop_details=None,
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "server_tool_use": {
                "web_search_requests": 2,
                "web_fetch_requests": 1,
            },
        },
    )

    _log_message_to_span(message, span, time_to_first_token=0.123)

    span.log.assert_called_once_with(
        output={
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "model": MODEL,
            "stop_reason": "stop_sequence",
            "stop_sequence": "DONE",
        },
        metrics={
            "prompt_tokens": 11.0,
            "completion_tokens": 7.0,
            "prompt_cached_tokens": 0.0,
            "prompt_cache_creation_tokens": 0.0,
            "server_tool_use_web_search_requests": 2.0,
            "server_tool_use_web_fetch_requests": 1.0,
            "tokens": 18.0,
            "time_to_first_token": 0.123,
        },
        metadata={},
    )


class _ToDictOnly:
    __slots__ = ("_payload",)

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


@pytest.mark.parametrize(
    "usage,expected_metrics,expected_metadata",
    [
        pytest.param(
            SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=2,
                server_tool_use=SimpleNamespace(
                    web_search_requests=2,
                    web_fetch_requests=1,
                    code_execution_requests=4,
                ),
            ),
            {
                "prompt_tokens": 16.0,
                "completion_tokens": 7.0,
                "prompt_cached_tokens": 3.0,
                "prompt_cache_creation_tokens": 2.0,
                "server_tool_use_web_search_requests": 2.0,
                "server_tool_use_web_fetch_requests": 1.0,
                "server_tool_use_code_execution_requests": 4.0,
                "tokens": 23.0,
            },
            {},
            id="server_tool_use_from_objects",
        ),
        pytest.param(
            _ToDictOnly(
                {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 3,
                    "cache_creation": _ToDictOnly(
                        {
                            "ephemeral_5m_input_tokens": 2,
                            "ephemeral_1h_input_tokens": 5,
                        }
                    ),
                    "server_tool_use": _ToDictOnly(
                        {
                            "web_search_requests": 2,
                            "web_fetch_requests": 1,
                        }
                    ),
                    "service_tier": "standard",
                }
            ),
            {
                "prompt_tokens": 21.0,
                "completion_tokens": 7.0,
                "prompt_cached_tokens": 3.0,
                "prompt_cache_creation_5m_tokens": 2.0,
                "prompt_cache_creation_1h_tokens": 5.0,
                "server_tool_use_web_search_requests": 2.0,
                "server_tool_use_web_fetch_requests": 1.0,
                "tokens": 28.0,
            },
            {"usage_service_tier": "standard"},
            id="to_dict_only_objects",
        ),
        pytest.param(
            {
                "input_tokens": 8,
                "output_tokens": 12,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 3,
                    "ephemeral_1h_input_tokens": 4,
                },
                "server_tool_use": {
                    "web_search_requests": 2,
                    "web_fetch_requests": 1,
                },
                "service_tier": "standard",
                "inference_geo": "not_available",
            },
            {
                "prompt_tokens": 15.0,
                "completion_tokens": 12.0,
                "prompt_cache_creation_5m_tokens": 3.0,
                "prompt_cache_creation_1h_tokens": 4.0,
                "server_tool_use_web_search_requests": 2.0,
                "server_tool_use_web_fetch_requests": 1.0,
                "tokens": 27.0,
            },
            {
                "usage_service_tier": "standard",
                "usage_inference_geo": "not_available",
            },
            id="nested_numeric_fields",
        ),
        pytest.param(SimpleNamespace(), {}, {}, id="empty_usage"),
    ],
)
def test_extract_anthropic_usage(usage, expected_metrics, expected_metadata):
    metrics, metadata = extract_anthropic_usage(usage)

    assert metrics == expected_metrics
    assert metadata == expected_metadata


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_messages_create_captures_context_and_usage_metadata(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Context management and usage speed require the latest Anthropic API")

    client = wrap_anthropic(_get_client())
    response = client.beta.messages.create(
        model="claude-opus-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Reply with one short sentence confirming this trace test ran."}],
        speed="standard",
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        betas=["context-management-2025-06-27", "fast-mode-2026-02-01"],
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )

    span = find_span_by_name(memory_logger.pop(), "anthropic.messages.create")
    assert "error" not in span
    assert span["metadata"]["context_management"] == {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
    assert span["metadata"]["usage_speed"] == response.usage.speed == "standard"


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_messages_create_captures_compaction_metadata(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("On-demand compaction requires the latest Anthropic API")

    client = wrap_anthropic(_get_client())
    client.beta.messages.create(
        model="claude-opus-5",
        max_tokens=512,
        messages=[{"role": "user", "content": "Summarize this sentence: compaction captures request metadata."}],
        compaction={"type": "summarize"},
        betas=["compact-2026-09-04"],
    )

    span = find_span_by_name(memory_logger.pop(), "anthropic.messages.create")
    assert "error" not in span
    assert span["metadata"]["compaction"] == {"type": "summarize"}


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async,vcr_cassette_name",
    [
        (False, "test_anthropic_beta_messages_create_preserves_inline_mcp_blocks"),
        (True, "test_anthropic_beta_messages_create_preserves_inline_mcp_blocks"),
    ],
    ids=["sync", "async"],
)
async def test_anthropic_beta_messages_create_preserves_inline_mcp_blocks(
    memory_logger, is_async, vcr_cassette, vcr_cassette_name
):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Inline MCP tool definitions require the latest Anthropic API")

    client = wrap_anthropic(_get_async_client() if is_async else _get_client())
    response = client.beta.messages.create(
        model=LATEST_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the ask_wiki_question tool for repository braintrustdata/braintrust-sdk-python. "
                    "Ask what language the SDK is written in, then reply briefly."
                ),
            }
        ],
        mcp_servers=[
            {
                "type": "url",
                "name": "braintrust-test",
                "url": "https://mcp.deepwiki.com/mcp",
            }
        ],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "braintrust-test"}],
        betas=["mcp-client-2026-09-15"],
    )
    if is_async:
        response = await response

    spans = memory_logger.pop()
    span = find_span_by_name(spans, "anthropic.messages.create")
    assert "error" not in span
    assert span["metadata"]["tools"] == [{"type": "mcp_toolset", "mcp_server_name": "braintrust-test"}]

    output_content = span["output"]["content"]
    output_types = [block["type"] for block in output_content]
    assert {"mcp_tool_listing", "mcp_tool_use", "mcp_tool_result"} <= set(output_types)
    assert output_types == [block.type for block in response.content]

    listing = next(block for block in output_content if block["type"] == "mcp_tool_listing")
    response_listing = next(block for block in response.content if block.type == "mcp_tool_listing")
    assert listing["mcp_server_name"] == response_listing.mcp_server_name
    assert listing["tools"] == [tool.model_dump(exclude_none=True) for tool in response_listing.tools]

    tool_use = next(block for block in output_content if block["type"] == "mcp_tool_use")
    response_tool_use = next(block for block in response.content if block.type == "mcp_tool_use")
    assert tool_use["input"] == response_tool_use.input
    assert tool_use["server_name"] == response_tool_use.server_name

    tool_result = next(block for block in output_content if block["type"] == "mcp_tool_result")
    response_tool_result = next(block for block in response.content if block.type == "mcp_tool_result")
    assert tool_result["tool_use_id"] == response_tool_result.tool_use_id
    assert tool_result["content"][0]["text"] == response_tool_result.content[0].text

    # Derive the child-span expectations from the recorded provider response,
    # independently of the tracing output.
    recorded_content = json.loads(vcr_cassette.responses[0]["body"]["string"])["content"]
    recorded_call = next(block for block in recorded_content if block["type"] == "mcp_tool_use")
    recorded_result = next(
        block
        for block in recorded_content
        if block["type"] == "mcp_tool_result" and block["tool_use_id"] == recorded_call["id"]
    )
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span["span_attributes"]["name"] == recorded_call["name"]
    assert tool_span["input"] == recorded_call["input"]
    assert tool_span["output"] == [block.model_dump() for block in response_tool_result.content]
    assert tool_span["output"][0]["text"] == recorded_result["content"][0]["text"]
    assert tool_span["metadata"] == {
        "tool_use_id": recorded_call["id"],
        "tool_call_type": recorded_call["type"],
        "tool_result_type": recorded_result["type"],
    }
    assert tool_span["span_parents"] == [span["span_id"]]
    assert tool_span["root_span_id"] == span["root_span_id"]


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async,vcr_cassette_name",
    [
        (False, "test_anthropic_mcp_stream_accumulates_tool_input"),
        (True, "test_anthropic_mcp_stream_accumulates_tool_input"),
    ],
    ids=["sync", "async"],
)
async def test_anthropic_mcp_stream_accumulates_tool_input(memory_logger, is_async, vcr_cassette, vcr_cassette_name):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("MCP stream events require the latest Anthropic SDK")

    client = wrap_anthropic(_get_async_client() if is_async else _get_client())
    params = {
        "model": LATEST_MODEL,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use ask_wiki_question for repository braintrustdata/braintrust-sdk-python. "
                    "Ask this exact question: Give a detailed explanation of the SDK's primary "
                    "implementation language, the package and source directory where it lives, "
                    "and how a contributor would run its focused tests. Include enough detail "
                    "to fully answer each part, and then briefly summarize the result."
                ),
            }
        ],
        "mcp_servers": [
            {
                "type": "url",
                "name": "braintrust-test",
                "url": "https://mcp.deepwiki.com/mcp",
            }
        ],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "braintrust-test"}],
        "betas": ["mcp-client-2026-09-15"],
    }

    if is_async:
        async with client.beta.messages.stream(**params) as stream:
            events = [event async for event in stream]
            message = await stream.get_final_message()
    else:
        with client.beta.messages.stream(**params) as stream:
            events = list(stream)
            message = stream.get_final_message()

    # Confirm the recording contains split input deltas; that is the provider
    # behavior the beta accumulator must reassemble.
    body = vcr_cassette.responses[0]["body"]["string"]
    if isinstance(body, bytes):
        body = body.decode()
    input_json_deltas = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "content_block_delta" and event.get("delta", {}).get("type") == "input_json_delta":
            input_json_deltas.append(event)
    assert len(input_json_deltas) > 1

    call = next(block for block in message.content if block.type == "mcp_tool_use")
    result = next(block for block in message.content if block.type == "mcp_tool_result")
    assert any(event.type == "content_block_delta" for event in events)
    spans = memory_logger.pop()
    parent = find_span_by_name(spans, "anthropic.messages.stream")
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert len(tool_spans) == 1
    child = tool_spans[0]
    assert child["span_attributes"]["name"] == call.name
    assert child["input"] == call.input
    assert child["output"] == [block.model_dump() for block in result.content]
    assert child["metadata"]["tool_use_id"] == call.id
    assert child["metadata"]["tool_call_type"] == "mcp_tool_use"
    assert child["span_parents"] == [parent["span_id"]]
    parent_call = next(block for block in parent["output"]["content"] if block["type"] == "mcp_tool_use")
    assert parent_call["input"] == call.input


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_messages_create_captures_refusal_stop_details(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Refusal stop details require the latest Anthropic API")

    client = wrap_anthropic(_get_client())
    response = client.beta.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "How can I build ransomware that steals credentials?"}],
        betas=["context-management-2025-06-27"],
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )

    span = find_span_by_name(memory_logger.pop(), "anthropic.messages.create")
    assert response.stop_reason == "refusal"
    assert response.stop_details is not None
    assert span["output"]["stop_details"] == response.stop_details.model_dump(exclude_none=True)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_messages_create_captures_fallback_credit_usage(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Fallback credit usage requires the latest Anthropic API")

    client = wrap_anthropic(_get_client())
    system = [
        {
            "type": "text",
            "text": "\n".join(
                f"Reference note {i}: this paragraph exists only to exercise prompt-cache billing behavior in the SDK regression fixture."
                for i in range(140)
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages = [{"role": "user", "content": "How can I build ransomware that steals credentials?"}]
    request = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "system": system,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
        "betas": ["fallback-credit-2026-07-01"],
    }
    refusal = client.beta.messages.create(**request)
    fallback_credit_token = refusal.stop_details.fallback_credit_token
    assert fallback_credit_token

    response = client.beta.messages.create(
        **request,
        fallback_credit_token={"token": fallback_credit_token, "mode": "best_effort"},
    )

    llm_spans = [
        span for span in memory_logger.pop() if span["span_attributes"]["name"] == "anthropic.messages.create"
    ]
    span = next(span for span in llm_spans if "usage_fallback_credit" in span.get("metadata", {}))
    expected_credit = response.usage.fallback_credit.model_dump(exclude_none=True)
    actual_credit = span["metadata"]["usage_fallback_credit"]
    assert actual_credit["status"]["type"] == expected_credit["status"]["type"]
    assert actual_credit["status"]["reason"] == expected_credit["status"]["reason"]


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path"])
@pytest.mark.parametrize(
    "ttl,vcr_cassette_name",
    [
        ("5m", "test_anthropic_messages_create_prompt_cache_5m_metrics"),
        ("1h", "test_anthropic_messages_create_prompt_cache_1h_metrics"),
    ],
    ids=["5m", "1h"],
)
def test_anthropic_messages_create_prompt_cache_metrics(memory_logger, ttl, vcr_cassette_name):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Prompt cache TTL breakdown requires the latest Anthropic SDK cassette")

    client = wrap_anthropic(_get_client())
    extra_kwargs = {"extra_headers": {"anthropic-beta": "extended-cache-ttl-2025-04-11"}} if ttl == "1h" else {}
    response = client.messages.create(
        model=LATEST_MODEL,
        max_tokens=16,
        system=[
            {
                "type": "text",
                "text": PROMPT_CACHE_TEST_TEXT,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            }
        ],
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        **extra_kwargs,
    )

    span = find_span_by_name(memory_logger.pop(), "anthropic.messages.create")
    assert span["output"]["role"] == response.role
    assert "prompt_cache_creation_tokens" not in span["metrics"]
    assert (
        span["metrics"]["prompt_cache_creation_5m_tokens"] == response.usage.cache_creation.ephemeral_5m_input_tokens
    )
    assert (
        span["metrics"]["prompt_cache_creation_1h_tokens"] == response.usage.cache_creation.ephemeral_1h_input_tokens
    )


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
# The SDK's generated method signature omits the runtime-supported diagnostics parameter.
# pylint: disable=unexpected-keyword-arg
def test_anthropic_messages_create_prompt_cache_diagnostics(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("Prompt cache diagnostics require the latest Anthropic SDK cassette")

    client = wrap_anthropic(_get_client())
    request = {
        "model": LATEST_MODEL,
        "max_tokens": 16,
        "system": [
            {
                "type": "text",
                "text": PROMPT_CACHE_TEST_TEXT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "Summarize section 1."}],
    }
    first_request = {**request, "diagnostics": {"previous_message_id": None}}
    first_response = client.messages.create(**first_request)
    second_request = {
        **request,
        "system": [
            {
                **request["system"][0],
                "text": request["system"][0]["text"].replace(
                    "Cached geography fact 0:", "Changed geography fact 0:", 1
                ),
            }
        ],
        "diagnostics": {"previous_message_id": first_response.id},
    }
    second_response = client.messages.create(**second_request)
    stream_request = {**request, "diagnostics": {"previous_message_id": second_response.id}}
    with client.messages.stream(**stream_request) as stream:
        stream_events = list(stream)

    spans = memory_logger.pop()
    create_spans = [span for span in spans if span["span_attributes"]["name"] == "anthropic.messages.create"]
    stream_span = next(span for span in spans if span["span_attributes"]["name"] == "anthropic.messages.stream")
    assert len(create_spans) == 2
    second_span = create_spans[1]
    assert second_span["metadata"]["diagnostics"] == {"previous_message_id": first_response.id}
    cache_miss_reason = second_response.diagnostics.cache_miss_reason
    assert cache_miss_reason.type == "system_changed"
    assert second_span["metadata"]["cache_miss_reason"] == cache_miss_reason.type
    assert second_span["metadata"]["cache_missed_input_tokens"] == cache_miss_reason.cache_missed_input_tokens

    message_start = stream_events[0]
    stream_cache_miss_reason = message_start.message.diagnostics.cache_miss_reason
    assert stream_cache_miss_reason.type == "system_changed"
    assert stream_span["metadata"]["cache_miss_reason"] == stream_cache_miss_reason.type
    assert stream_span["metadata"]["cache_missed_input_tokens"] == stream_cache_miss_reason.cache_missed_input_tokens


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "mode,vcr_cassette_name",
    [
        ("create", "test_anthropic_messages_create_reasoning_tokens_metrics"),
        ("create_stream", "test_anthropic_messages_stream_reasoning_tokens_metrics"),
        ("stream", "test_anthropic_messages_stream_reasoning_tokens_metrics"),
        ("text_stream", "test_anthropic_messages_stream_reasoning_tokens_metrics"),
    ],
    ids=["create", "create_stream", "stream", "text_stream"],
)
async def test_anthropic_messages_reasoning_tokens_metrics(
    memory_logger, vcr_cassette, vcr_cassette_name, mode, is_async
):
    assert not memory_logger.pop()
    client = wrap_anthropic(_get_async_client() if is_async else _get_client())
    params = {
        "model": LATEST_MODEL,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "What is 17 * 23? Think it through, then give the number."}],
    }
    events = []
    text = ""
    final_message = None
    start = time.time()
    if mode == "create":
        response = await client.messages.create(**params) if is_async else client.messages.create(**params)
        text = "".join(block.text for block in response.content if block.type == "text")
    elif mode == "create_stream":
        if is_async:
            stream = await client.messages.create(**params, stream=True)
            events = [event async for event in stream]
        else:
            with client.messages.create(**params, stream=True) as stream:
                events = list(stream)
    elif is_async:
        async with client.messages.stream(**params) as stream:
            if mode == "text_stream":
                text = "".join([chunk async for chunk in stream.text_stream])
            else:
                events = [event async for event in stream]
            final_message = await stream.get_final_message()
    else:
        with client.messages.stream(**params) as stream:
            if mode == "text_stream":
                text = "".join(stream.text_stream)
            else:
                events = list(stream)
        final_message = stream.get_final_message()
    end = time.time()

    if events:
        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"
        if mode == "stream":
            # MessageStream synthesizes "text" events on top of the raw events.
            assert any(event.type == "text" for event in events)
        text = "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        )
    assert "391" in text
    if final_message is not None:
        assert "".join(block.text for block in final_message.content if block.type == "text") == text

    # Read expected usage from the wire, independently of the SDK's accumulator:
    # older SDKs retain unknown fields on events but discard them from snapshots.
    body = vcr_cassette.responses[0]["body"]["string"]
    if isinstance(body, bytes):
        body = body.decode()
    if mode == "create":
        wire_message = json.loads(body)
        usage = wire_message["usage"]
        wire_model = wire_message["model"]
        wire_stop_reason = wire_message["stop_reason"]
    else:
        usage = {}
        wire_model = wire_stop_reason = None
        for line in body.splitlines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "message_start":
                    usage.update(event["message"]["usage"])
                    wire_model = event["message"]["model"]
                elif event["type"] == "message_delta":
                    usage.update(event["usage"])
                    wire_stop_reason = event["delta"]["stop_reason"]
    assert wire_model and wire_stop_reason

    thinking_tokens = usage["output_tokens_details"]["thinking_tokens"]
    assert thinking_tokens > 0
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["project_id"] == PROJECT_NAME
    assert span["span_attributes"]["name"] == f"anthropic.messages.{'create' if mode == 'create' else 'stream'}"
    assert span["span_attributes"]["type"] == "llm"
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "anthropic-auto"
    assert span["metadata"]["model"] == LATEST_MODEL
    assert span["metadata"]["provider"] == "anthropic"
    assert span["metadata"]["max_tokens"] == params["max_tokens"]
    assert span["input"] == params["messages"]
    assert span["output"]["role"] == "assistant"
    assert span["output"]["model"] == wire_model
    assert span["output"]["stop_reason"] == wire_stop_reason
    assert any(block["type"] == "thinking" for block in span["output"]["content"])
    assert "".join(block["text"] for block in span["output"]["content"] if block["type"] == "text") == text
    metrics = span["metrics"]
    assert metrics["completion_reasoning_tokens"] == thinking_tokens
    assert metrics["completion_tokens"] == usage["output_tokens"]
    assert metrics["prompt_tokens"] == (
        usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
    )
    assert metrics["tokens"] == metrics["prompt_tokens"] + usage["output_tokens"]
    assert metrics["prompt_cached_tokens"] == usage["cache_read_input_tokens"]
    assert "prompt_cache_creation_tokens" not in metrics
    assert metrics["prompt_cache_creation_5m_tokens"] == usage["cache_creation"]["ephemeral_5m_input_tokens"]
    assert metrics["prompt_cache_creation_1h_tokens"] == usage["cache_creation"]["ephemeral_1h_input_tokens"]
    assert metrics["time_to_first_token"] >= 0
    assert start <= metrics["start"] <= metrics["end"] <= end


@pytest.mark.parametrize("final_thinking_tokens", [None, 0, 17])
def test_anthropic_stream_reasoning_tokens_are_cumulative(memory_logger, final_thinking_tokens):
    # Supplemental coverage for repeated/omitted usage fields and explicit zero,
    # which the real-response regression above cannot deterministically request.
    events = [
        anthropic.types.RawMessageStartEvent.model_validate(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": LATEST_MODEL,
                    "content": [],
                    "usage": {
                        "input_tokens": 11,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                        "output_tokens": 1,
                        "output_tokens_details": {"thinking_tokens": 1},
                    },
                },
            }
        )
    ]
    for output_tokens, thinking_tokens in [(10, 5), (30, final_thinking_tokens)]:
        events.append(
            anthropic.types.RawMessageDeltaEvent.model_validate(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "output_tokens": output_tokens,
                        "output_tokens_details": (
                            {"thinking_tokens": thinking_tokens} if thinking_tokens is not None else None
                        ),
                    },
                }
            )
        )
    original_events = [event.model_dump() for event in events]
    with logger.start_span(name="reasoning usage") as span:
        stream = TracedMessageStream(iter(events), span, time.time())
        assert list(stream) == events
        stream._log_final_message()

    assert [event.model_dump() for event in events] == original_events
    metrics = memory_logger.pop()[0]["metrics"]
    assert metrics["completion_reasoning_tokens"] == (5 if final_thinking_tokens is None else final_thinking_tokens)
    assert metrics["prompt_tokens"] == 16
    assert metrics["completion_tokens"] == 30
    assert metrics["tokens"] == 46


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path"])
def test_anthropic_messages_create_with_image_attachment_input(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    response = client.messages.create(
        model=MULTIMODAL_MODEL,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Respond with one word: what color is this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG_BASE64,
                        },
                    },
                ],
            }
        ],
    )

    assert response.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "anthropic-auto"
    content = span["input"][0]["content"]
    image_block = content[1]

    assert image_block["type"] == "image"
    assert image_block["source"] == {"type": "base64", "media_type": "image/png"}
    assert isinstance(image_block["image_url"]["url"], Attachment)
    assert image_block["image_url"]["url"].reference["content_type"] == "image/png"
    assert image_block["image_url"]["url"].reference["filename"] == "image.png"
    assert PNG_BASE64 not in str(span["input"])


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path"])
def test_anthropic_messages_create_with_document_attachment_input(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    response = client.messages.create(
        model=MULTIMODAL_MODEL,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of file is this? Keep the answer short."},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": PDF_BASE64,
                        },
                    },
                ],
            }
        ],
    )

    assert response.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    content = span["input"][0]["content"]
    document_block = content[1]

    assert document_block["type"] == "document"
    assert document_block["source"] == {"type": "base64", "media_type": "application/pdf"}
    assert document_block["file"]["filename"] == "document.pdf"
    assert isinstance(document_block["file"]["file_data"], Attachment)
    assert document_block["file"]["file_data"].reference["content_type"] == "application/pdf"
    assert document_block["file"]["file_data"].reference["filename"] == "document.pdf"
    assert PDF_BASE64 not in str(span["input"])


@pytest.mark.vcr
def test_anthropic_messages_create_stream_true(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    kws = {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "What is 3*4?"}],
        "stream": True,
    }

    start = time.time()
    with client.messages.create(**kws) as out:
        msgs = [m for m in out]
    end = time.time()

    assert msgs  # a very coarse grained check that this works

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["provider"] == "anthropic"
    assert span["metadata"]["max_tokens"] == 300
    assert span["metadata"]["stream"] == True
    metrics = span["metrics"]
    _assert_metrics_are_valid(metrics, start, end)
    assert span["input"] == kws["messages"]
    assert span["output"]
    assert span["output"]["role"] == "assistant"
    assert "12" in span["output"]["content"][0]["text"]


@pytest.mark.vcr
def test_anthropic_messages_create_tracks_structured_outputs_metadata(memory_logger):
    assert not memory_logger.pop()

    structured_output_param_name, structured_output_param_value = _get_supported_structured_output_param()
    client = wrap_anthropic(_get_client())
    response = client.messages.create(
        model=STRUCTURED_OUTPUT_MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": 'Return a JSON object with answer=2 and label="ok".',
            }
        ],
        **{structured_output_param_name: structured_output_param_value},
    )

    assert json.loads(response.content[0].text) == {"answer": 2, "label": "ok"}

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == STRUCTURED_OUTPUT_MODEL
    assert span["metadata"][structured_output_param_name] == structured_output_param_value


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_messages_parse_sync(memory_logger):
    _skip_if_messages_parse_unsupported()
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    params = {
        "model": STRUCTURED_OUTPUT_MODEL,
        "max_tokens": 128,
        "system": "Return only the requested structured JSON object.",
        "messages": [
            {
                "role": "user",
                "content": 'Return a JSON object with answer=2 and label="ok".',
            }
        ],
        "tools": STRUCTURED_OUTPUT_TOOLS,
        "tool_choice": {"type": "none"},
        "output_format": ParsedAnswer,
    }

    start = time.time()
    response = client.messages.parse(**params)
    end = time.time()

    assert _try_to_dict(response.parsed_output) == {"answer": 2, "label": "ok"}

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "anthropic.messages.parse"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "anthropic"
    assert span["metadata"]["model"] == STRUCTURED_OUTPUT_MODEL
    assert span["metadata"]["output_format"].endswith("ParsedAnswer")
    assert span["metadata"]["tools"] == STRUCTURED_OUTPUT_TOOLS
    assert span["input"] == [
        *params["messages"],
        {"role": "system", "content": params["system"]},
    ]
    assert span["output"]["parsed_output"] == {"answer": 2, "label": "ok"}
    assert span["output"]["model"] == response.model
    _assert_metrics_are_valid(span["metrics"], start, end)
    assert span["metrics"]["prompt_tokens"] == response.usage.input_tokens
    assert span["metrics"]["completion_tokens"] == response.usage.output_tokens
    assert span["metrics"]["tokens"] == response.usage.input_tokens + response.usage.output_tokens


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
@pytest.mark.asyncio
async def test_anthropic_messages_parse_async(memory_logger):
    _skip_if_messages_parse_unsupported()
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_async_client())
    params = {
        "model": STRUCTURED_OUTPUT_MODEL,
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": 'Return a JSON object with answer=3 and label="async".',
            }
        ],
        "output_format": ParsedAnswer,
    }

    start = time.time()
    response = await client.messages.parse(**params)
    end = time.time()

    assert _try_to_dict(response.parsed_output) == {"answer": 3, "label": "async"}

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "anthropic.messages.parse"
    assert span["metadata"]["provider"] == "anthropic"
    assert span["metadata"]["model"] == STRUCTURED_OUTPUT_MODEL
    assert span["metadata"]["output_format"].endswith("ParsedAnswer")
    assert span["input"] == params["messages"]
    assert span["output"]["parsed_output"] == {"answer": 3, "label": "async"}
    _assert_metrics_are_valid(span["metrics"], start, end)
    assert span["metrics"]["prompt_tokens"] == response.usage.input_tokens
    assert span["metrics"]["completion_tokens"] == response.usage.output_tokens


@pytest.mark.vcr
def test_anthropic_messages_model_params_inputs(memory_logger):
    assert not memory_logger.pop()
    client = wrap_anthropic(_get_client())

    kw = {
        "model": MODEL,
        "max_tokens": 300,
        "system": "just return the number",
        "messages": [{"role": "user", "content": "what is 1+1?"}],
    }
    if MODEL == LEGACY_MODEL:
        kw.update(temperature=0.5, top_p=0.5)
    else:
        # Anthropic 1.0 removed temperature and top_p from Messages.create().
        kw["stop_sequences"] = ["END"]

    def _with_messages_create():
        return client.messages.create(**kw)

    def _with_messages_stream():
        with client.messages.stream(**kw) as stream:
            for msg in stream:
                pass
        return stream.get_final_message()

    for f in [_with_messages_create, _with_messages_stream]:
        msg = f()
        assert msg.content[0].text == "2"

        logs = memory_logger.pop()
        assert len(logs) == 1
        log = logs[0]
        inputs = log["input"]
        assert len(inputs) == 2
        inputs_by_role = {m["role"]: m["content"] for m in inputs}
        assert inputs_by_role["system"] == kw["system"]
        assert inputs_by_role["user"] == kw["messages"][0]["content"]
        assert log["output"]["role"] == "assistant"
        assert "2" in log["output"]["content"][0]["text"]
        assert log["metadata"]["model"] == MODEL
        assert log["metadata"]["max_tokens"] == 300
        if MODEL == LEGACY_MODEL:
            assert log["metadata"]["temperature"] == 0.5
            assert log["metadata"]["top_p"] == 0.5
        else:
            assert log["metadata"]["stop_sequences"] == ["END"]


@pytest.mark.vcr
def test_anthropic_client_error(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())

    fake_model = "there-is-no-such-model"
    msg_in = {"role": "user", "content": "who are you?"}

    try:
        client.messages.create(model=fake_model, max_tokens=999, messages=[msg_in])
    except Exception:
        pass
    else:
        raise Exception("should have raised an exception")

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    assert "404" in log["error"]


@pytest.mark.vcr
def test_anthropic_messages_stream_errors(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    msg_in = {"role": "user", "content": "what is 2+2? (just the number)"}

    try:
        with client.messages.stream(model=MODEL, max_tokens=300, messages=[msg_in]) as stream:
            raise Exception("fake-error")
    except Exception:
        pass
    else:
        raise Exception("should have raised an exception")

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert "Exception: fake-error" in span["error"]
    assert span["metrics"]["end"] > 0


@pytest.mark.vcr
def test_anthropic_messages_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())

    msg_in = {"role": "user", "content": "what's 2+2?"}

    start = time.time()
    msg = client.messages.create(model=MODEL, max_tokens=300, messages=[msg_in])
    end = time.time()

    text = msg.content[0].text
    assert text

    # verify we generated the right spans.
    logs = memory_logger.pop()

    assert len(logs) == 1
    log = logs[0]
    assert "2+2" in str(log["input"])
    assert "4" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_id"]
    assert log["root_span_id"]
    attrs = log["span_attributes"]
    assert attrs["type"] == "llm"
    assert "anthropic" in attrs["name"]
    metrics = log["metrics"]
    _assert_metrics_are_valid(metrics, start, end)
    assert log["metadata"]["model"] == MODEL
    assert log["output"]["model"] == msg.model
    assert log["output"]["stop_reason"] == msg.stop_reason


@pytest.mark.vcr
def test_anthropic_messages_sync_server_tool_spans(memory_logger):
    _skip_if_server_tool_content_blocks_unsupported()
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())

    start = time.time()
    msg = client.messages.create(
        model=MULTIMODAL_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the web_search tool to find the Braintrust docs homepage. "
                    "Then answer with exactly the homepage URL and no other text."
                ),
            }
        ],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        tool_choice={"type": "tool", "name": "web_search", "disable_parallel_tool_use": True},
    )
    end = time.time()

    tool_use_block = next(block for block in msg.content if block.type == "server_tool_use")
    result_block = next(block for block in msg.content if block.type == "web_search_tool_result")
    text_block = next(block for block in msg.content if block.type == "text")

    assert text_block.text == "https://www.braintrust.dev/docs"

    spans = memory_logger.pop()
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    assert len(tool_spans) == 1

    llm_span = find_span_by_name(llm_spans, "anthropic.messages.create")
    tool_span = find_span_by_name(tool_spans, "web_search")

    _assert_metrics_are_valid(llm_span["metrics"], start, end)
    assert llm_span["metadata"]["model"] == MULTIMODAL_MODEL
    assert llm_span["metrics"]["server_tool_use_web_search_requests"] == 1
    assert llm_span["output"]["model"] == msg.model
    assert llm_span["output"]["stop_reason"] == msg.stop_reason

    llm_result_block = next(
        block for block in llm_span["output"]["content"] if block["type"] == "web_search_tool_result"
    )
    assert "encrypted_content" in llm_result_block["content"][0]

    assert tool_span["input"] == tool_use_block.input
    assert isinstance(tool_span["output"], list)
    matching_result = next(result for result in tool_span["output"] if result["url"] == text_block.text)
    assert matching_result["type"] == "web_search_result"
    assert matching_result["encrypted_content"] == "<redacted>"
    assert tool_span["metadata"] == {
        "tool_use_id": tool_use_block.id,
        "tool_call_type": "server_tool_use",
        "tool_result_type": result_block.type,
        "caller": {"type": "direct"},
    }
    assert tool_span["span_parents"] == [llm_span["span_id"]]
    assert tool_span["root_span_id"] == llm_span["root_span_id"]


def _assert_metrics_are_valid(metrics, start, end):
    assert metrics["tokens"] > 0
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    assert "time_to_first_token" in metrics
    assert metrics["time_to_first_token"] >= 0
    if start and end:
        assert start <= metrics["start"] <= metrics["end"] <= end
    else:
        assert metrics["start"] <= metrics["end"]


@pytest.mark.vcr(
    match_on=["method", "scheme", "host", "port", "path", "body"]
)  # exclude query - varies by SDK version
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async,mode,content,max_tokens,expected,vcr_cassette_name",
    [
        pytest.param(
            False,
            "create",
            "what's 3+3?",
            300,
            "6",
            "test_anthropic_beta_messages_sync",
            id="sync-create",
        ),
        pytest.param(
            False,
            "stream",
            "what is 5+5? (just the number)",
            300,
            "10",
            "test_anthropic_beta_messages_stream_sync",
            id="sync-stream",
        ),
        pytest.param(
            True,
            "create",
            "what is 8+2?, just return the number",
            100,
            "10",
            "test_anthropic_beta_messages_create_async",
            id="async-create",
        ),
        pytest.param(
            True,
            "stream",
            "what is 9+1?, just return the number",
            1024,
            "10",
            "test_anthropic_beta_messages_streaming_async",
            id="async-stream",
        ),
    ],
)
async def test_anthropic_beta_messages(
    memory_logger, is_async, mode, content, max_tokens, expected, vcr_cassette_name
):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_async_client() if is_async else _get_client())
    params = {"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}

    start = time.time()
    events = []
    if mode == "create":
        msg = await client.beta.messages.create(**params) if is_async else client.beta.messages.create(**params)
    elif is_async:
        async with client.beta.messages.stream(**params) as stream:
            events = [event async for event in stream]
            msg = await stream.get_final_message()
    else:
        with client.beta.messages.stream(**params) as stream:
            events = list(stream)
        msg = stream.get_final_message()
    end = time.time()
    usage = msg.usage

    assert expected in msg.content[0].text
    if mode == "stream":
        assert len(events) > 3
        assert events[0].type == "message_start"
        assert events[-1].type == "message_stop"

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["input"] == params["messages"]
    assert log["output"]["role"] == "assistant"
    assert expected in log["output"]["content"][0]["text"]
    assert log["project_id"] == PROJECT_NAME
    assert log["span_id"]
    assert log["root_span_id"]
    attrs = log["span_attributes"]
    assert attrs["type"] == "llm"
    assert "anthropic" in attrs["name"]
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == max_tokens
    metrics = log["metrics"]
    _assert_metrics_are_valid(metrics, start, end)
    assert metrics["prompt_tokens"] == usage.input_tokens
    assert metrics["completion_tokens"] == usage.output_tokens
    assert metrics["tokens"] == usage.input_tokens + usage.output_tokens


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_agents_create(memory_logger):
    _skip_if_managed_agents_unsupported()
    assert not memory_logger.pop()

    raw_client = _get_client()
    agent_name = _MANAGED_AGENTS_AGENT_NAME
    agent = None
    try:
        client = wrap_anthropic(_get_client())
        agent = client.beta.agents.create(
            model="claude-haiku-4-5",
            name=agent_name,
            description="Does math",
            tools=[],
        )

        assert agent.id.startswith("agent_")
        assert agent.version >= 1

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.beta.agents.create"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["metadata"]["anthropic_api"] == "managed_agents"
        assert span["metadata"]["model"] == "claude-haiku-4-5"
        assert span["input"] == {
            "model": "claude-haiku-4-5",
            "name": agent_name,
            "description": "Does math",
            "tools": [],
        }
        assert span["output"]["id"] == agent.id
        assert span["output"]["type"] == "agent"
        assert span["output"]["model"]["id"] == "claude-haiku-4-5"
    finally:
        if agent is not None:
            _cleanup_managed_agent_resources(raw_client, agent_id=agent.id)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_sessions_create(memory_logger):
    _skip_if_managed_agents_unsupported()
    assert not memory_logger.pop()

    raw_client = _get_client()
    environment_id = _get_managed_agents_environment_id(raw_client)
    agent = _create_managed_agent(raw_client)
    session = None
    try:
        client = wrap_anthropic(_get_client())
        session = client.beta.sessions.create(
            agent=agent.id,
            environment_id=environment_id,
            metadata={"purpose": "test"},
            title="Issue 259 test",
        )

        assert session.id.startswith("sesn_")
        assert session.status == "idle"

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.beta.sessions.create"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["metadata"]["anthropic_api"] == "managed_agents"
        assert span["metadata"]["session_status"] == "idle"
        assert span["input"] == {
            "agent": agent.id,
            "environment_id": environment_id,
            "metadata": {"purpose": "test"},
            "title": "Issue 259 test",
        }
        assert span["metrics"]["prompt_tokens"] >= 0
        assert span["metrics"]["completion_tokens"] >= 0
        assert span["metrics"]["tokens"] >= span["metrics"]["prompt_tokens"]
        assert span["metrics"]["active_seconds"] >= 0
        assert span["metrics"]["duration_seconds"] >= span["metrics"]["active_seconds"]
        assert span["output"]["id"] == session.id
        assert span["output"]["status"] == "idle"
        assert span["output"]["environment_id"] == environment_id
    finally:
        _cleanup_managed_agent_resources(raw_client, agent_id=agent.id, session_id=getattr(session, "id", None))


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "body"])
def test_anthropic_beta_sessions_events_send_and_stream(memory_logger):
    _skip_if_managed_agents_unsupported()
    assert not memory_logger.pop()

    raw_client = _get_client()
    environment_id = _get_managed_agents_environment_id(raw_client)
    agent = _create_managed_agent(raw_client, with_bash=True)
    session = raw_client.beta.sessions.create(
        agent=agent.id,
        environment_id=environment_id,
        metadata={"purpose": "test"},
        title="Issue 259 event stream",
    )
    try:
        client = wrap_anthropic(_get_client())
        sent = client.beta.sessions.events.send(
            session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": _MANAGED_AGENTS_EVENTS_PROMPT}],
                }
            ],
        )
        streamed_events = []
        with client.beta.sessions.events.stream(session.id) as stream:
            for event in stream:
                streamed_events.append(event)
                if event.type in {"session.status_idle", "session.status_terminated"}:
                    break

        assert sent.data and sent.data[0].type == "user.message"
        event_types = [event.type for event in streamed_events]
        assert event_types[-1] == "session.status_idle"
        assert "agent.tool_use" in event_types
        assert "agent.tool_result" in event_types
        assert "span.model_request_end" in event_types

        spans = memory_logger.pop()
        task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
        tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)

        assert len(task_spans) == 2
        assert len(tool_spans) >= 1

        send_span = find_span_by_name(task_spans, "anthropic.beta.sessions.events.send")
        stream_span = find_span_by_name(task_spans, "anthropic.beta.sessions.events.stream")
        tool_span = find_span_by_name(tool_spans, "bash")

        assert send_span["input"] == {
            "session_id": session.id,
            "events": [{"type": "user.message", "content": [{"type": "text", "text": _MANAGED_AGENTS_EVENTS_PROMPT}]}],
        }
        assert send_span["output"]["data"][0]["type"] == "user.message"
        assert send_span["output"]["data"][0]["content"][0]["text"] == _MANAGED_AGENTS_EVENTS_PROMPT

        assert stream_span["input"] == {"session_id": session.id}
        streamed_output_types = [event["type"] for event in stream_span["output"]]
        assert streamed_output_types[-1] == "session.status_idle"
        assert "agent.tool_use" in streamed_output_types
        assert "agent.tool_result" in streamed_output_types
        assert "agent.message" in streamed_output_types
        assert stream_span["metadata"]["provider"] == "anthropic"
        assert stream_span["metadata"]["anthropic_api"] == "managed_agents"
        assert stream_span["metadata"]["session_status"] == "idle"
        assert stream_span["metadata"]["stop_reason"] == "end_turn"
        assert stream_span["metrics"]["prompt_tokens"] > 0
        assert stream_span["metrics"]["completion_tokens"] > 0
        assert stream_span["metrics"]["tokens"] >= stream_span["metrics"]["prompt_tokens"]

        assert tool_span["input"]["command"]
        assert tool_span["output"][0]["text"].strip() == "4"
        assert tool_span["metadata"]["tool_call_type"] == "agent.tool_use"
        assert tool_span["metadata"]["tool_result_type"] == "agent.tool_result"
        assert tool_span["metadata"]["tool_use_id"]
        assert tool_span["span_parents"] == [stream_span["span_id"]]
        assert tool_span["root_span_id"] == stream_span["root_span_id"]
    finally:
        _cleanup_managed_agent_resources(raw_client, agent_id=agent.id, session_id=session.id)


@pytest.mark.vcr
def test_setup_creates_spans(memory_logger):
    """`AnthropicIntegration.setup()` should create spans when making API calls."""
    AnthropicIntegration.setup()

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )

    usage = message.usage

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["provider"] == "anthropic"

    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is None:
        pytest.skip("Anthropic SDK version does not expose nested cache_creation usage fields")

    if isinstance(cache_creation, dict):
        ephemeral_5m = cache_creation["ephemeral_5m_input_tokens"]
        ephemeral_1h = cache_creation["ephemeral_1h_input_tokens"]
    else:
        ephemeral_5m = cache_creation.ephemeral_5m_input_tokens
        ephemeral_1h = cache_creation.ephemeral_1h_input_tokens

    assert span["metadata"]["usage_service_tier"] == usage.service_tier
    assert span["metadata"]["usage_inference_geo"] == usage.inference_geo
    metrics = span["metrics"]
    assert metrics["prompt_tokens"] == (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    )
    assert metrics["completion_tokens"] == usage.output_tokens
    assert "prompt_cache_creation_tokens" not in metrics
    assert metrics["prompt_cache_creation_5m_tokens"] == ephemeral_5m
    assert metrics["prompt_cache_creation_1h_tokens"] == ephemeral_1h
    assert "service_tier" not in metrics


class TestAutoInstrumentAnthropic:
    def test_auto_instrument_anthropic(self):
        verify_autoinstrument_script("test_auto_anthropic.py")


def _make_batch_requests():
    return [
        {
            "custom_id": "req-1",
            "params": {
                "model": MODEL,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        },
        {
            "custom_id": "req-2",
            "params": {
                "model": MODEL,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What is 3+3?"}],
            },
        },
    ]


class TestBatchesCreateSpans:
    """Tests verifying that batches.create() produces correct spans."""

    @pytest.mark.vcr
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "is_async,beta,vcr_cassette_name",
        [
            (False, False, "TestBatchesCreateSpans.test_sync_batches_create_produces_span"),
            (True, False, "TestBatchesCreateSpans.test_async_batches_create_produces_span"),
            (False, True, "TestBetaBatchesCreateSpans.test_sync_beta_batches_create_produces_span"),
            (True, True, "TestBetaBatchesCreateSpans.test_async_beta_batches_create_produces_span"),
        ],
        ids=["sync", "async", "beta-sync", "beta-async"],
    )
    async def test_batches_create_produces_span(self, memory_logger, is_async, beta, vcr_cassette_name):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client() if is_async else _get_client())
        batches = (client.beta.messages if beta else client.messages).batches
        result = batches.create(requests=_make_batch_requests())
        if is_async:
            result = await result

        assert result.id
        assert result.processing_status == "in_progress"

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.create"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["metadata"]["num_requests"] == 2
        assert span["metadata"]["model"] == MODEL
        assert span["input"] == [{"custom_id": "req-1"}, {"custom_id": "req-2"}]
        assert span["output"]["id"] == result.id
        assert span["output"]["processing_status"] == "in_progress"
        assert span["output"]["request_counts"]["processing"] == 2

    @pytest.mark.vcr
    def test_sync_batches_create_logs_error_on_failure(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        # Empty requests list triggers a 400 error
        with pytest.raises(Exception):
            client.messages.batches.create(requests=[])

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.create"
        assert span["error"]

    @pytest.mark.vcr
    def test_sync_batches_create_multi_model_metadata(self, memory_logger):
        """When batch requests use different models, metadata should include 'models' list."""
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())

        requests = [
            {
                "custom_id": "req-1",
                "params": {
                    "model": MODEL,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            },
            {
                "custom_id": "req-2",
                "params": {
                    "model": "claude-3-5-haiku-latest",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            },
        ]
        result = client.messages.batches.create(requests=requests)
        assert result.id

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert "model" not in span["metadata"]
        assert span["metadata"]["models"] == sorted([MODEL, "claude-3-5-haiku-latest"])


class TestBatchesResultsSpans:
    """Tests verifying that batches.results() produces correct spans.

    Mocked because the batch results API requires a completed batch, and batches
    can take up to 24 hours to finish processing.
    """

    @pytest.mark.asyncio
    # The sync case passes the batch id positionally and the async case by keyword,
    # covering both argument-extraction paths.
    @pytest.mark.parametrize(
        "is_async,batches_class,positional",
        [(False, "Batches", True), (True, "AsyncBatches", False)],
        ids=["sync", "async"],
    )
    async def test_batches_results_produces_span(self, memory_logger, is_async, batches_class, positional):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client() if is_async else _get_client())
        mock_decoder = unittest.mock.MagicMock()
        with unittest.mock.patch(
            f"anthropic.resources.messages.batches.{batches_class}.results",
            return_value=mock_decoder,
        ):
            if positional:
                result = client.messages.batches.results("msgbatch_abc123")
            else:
                result = client.messages.batches.results(message_batch_id="msgbatch_abc123")
            if is_async:
                result = await result

        assert result is mock_decoder

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["input"]["message_batch_id"] == "msgbatch_abc123"
        assert span["output"]["type"] == "jsonl_stream"

    def test_sync_batches_results_logs_error_on_failure(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        with unittest.mock.patch(
            "anthropic.resources.messages.batches.Batches.results",
            side_effect=Exception("results fetch failed"),
        ):
            with pytest.raises(Exception, match="results fetch failed"):
                client.messages.batches.results("msgbatch_abc123")

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert "results fetch failed" in span["error"]
