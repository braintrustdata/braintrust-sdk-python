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
from braintrust.integrations.anthropic._utils import extract_anthropic_usage
from braintrust.integrations.anthropic.tracing import (
    _get_input_from_kwargs,
    _get_metadata_from_kwargs,
    _log_message_to_span,
)
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger


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
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
PDF_BASE64 = "JVBERi0xLjAKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PmVuZG9iagoyIDAgb2JqCjw8L1R5cGUvUGFnZXMvS2lkc1szIDAgUl0vQ291bnQgMT4+ZW5kb2JqCjMgMCBvYmoKPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCA2MTIgNzkyXT4+ZW5kb2JqCnhyZWYKMCA0CjAwMDAwMDAwMDAgNjU1MzUgZg0KMDAwMDAwMDAxMCAwMDAwMCBuDQowMDAwMDAwMDUzIDAwMDAwIG4NCjAwMDAwMDAxMDIgMDAwMDAgbg0KdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoxNDkKJUVPRg=="


def _get_client():
    return anthropic.Anthropic()


def _get_async_client():
    return anthropic.AsyncAnthropic()


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


def test_get_input_from_kwargs_converts_multimodal_base64_blocks_to_attachments():
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these files."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG_BASE64,
                        },
                    },
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
        ]
    }

    processed_input = _get_input_from_kwargs(kwargs)

    content = processed_input[0]["content"]
    image_block = content[1]
    document_block = content[2]

    assert image_block["type"] == "image"
    assert image_block["source"] == {"type": "base64", "media_type": "image/png"}
    assert isinstance(image_block["image_url"]["url"], Attachment)
    assert image_block["image_url"]["url"].reference["content_type"] == "image/png"
    assert image_block["image_url"]["url"].reference["filename"] == "image.png"

    assert document_block["type"] == "document"
    assert document_block["source"] == {"type": "base64", "media_type": "application/pdf"}
    assert document_block["file"]["filename"] == "document.pdf"
    assert isinstance(document_block["file"]["file_data"], Attachment)
    assert document_block["file"]["file_data"].reference["content_type"] == "application/pdf"
    assert document_block["file"]["file_data"].reference["filename"] == "document.pdf"

    serialized = str(processed_input)
    assert PNG_BASE64 not in serialized
    assert PDF_BASE64 not in serialized


def test_get_metadata_from_kwargs_includes_structured_output_params():
    metadata = _get_metadata_from_kwargs(
        {
            "model": MODEL,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": STRUCTURED_OUTPUT_SCHEMA,
                }
            },
            "output_format": {
                "type": "json_schema",
                "schema": STRUCTURED_OUTPUT_SCHEMA,
            },
        }
    )

    assert metadata == {
        "provider": "anthropic",
        "model": MODEL,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": STRUCTURED_OUTPUT_SCHEMA,
            }
        },
        "output_format": {
            "type": "json_schema",
            "schema": STRUCTURED_OUTPUT_SCHEMA,
        },
    }


def test_log_message_to_span_includes_stop_reason_and_stop_sequence():
    span = unittest.mock.MagicMock()
    message = SimpleNamespace(
        role="assistant",
        content=[{"type": "text", "text": "done"}],
        model=MODEL,
        stop_reason="stop_sequence",
        stop_sequence="DONE",
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


def test_extract_anthropic_usage_includes_server_tool_use_metrics_from_objects():
    usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=2,
        server_tool_use=SimpleNamespace(
            web_search_requests=2,
            web_fetch_requests=1,
            code_execution_requests=4,
        ),
    )

    metrics, metadata = extract_anthropic_usage(usage)

    assert metrics == {
        "prompt_tokens": 16.0,
        "completion_tokens": 7.0,
        "prompt_cached_tokens": 3.0,
        "prompt_cache_creation_tokens": 2.0,
        "server_tool_use_web_search_requests": 2.0,
        "server_tool_use_web_fetch_requests": 1.0,
        "server_tool_use_code_execution_requests": 4.0,
        "tokens": 23.0,
    }
    assert metadata == {}


def test_extract_anthropic_usage_supports_to_dict_only_objects():
    class ToDictOnly:
        __slots__ = ("_payload",)

        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

    usage = ToDictOnly(
        {
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_input_tokens": 3,
            "cache_creation": ToDictOnly(
                {
                    "ephemeral_5m_input_tokens": 2,
                    "ephemeral_1h_input_tokens": 5,
                }
            ),
            "server_tool_use": ToDictOnly(
                {
                    "web_search_requests": 2,
                    "web_fetch_requests": 1,
                }
            ),
            "service_tier": "standard",
        }
    )

    metrics, metadata = extract_anthropic_usage(usage)

    assert metrics == {
        "prompt_tokens": 21.0,
        "completion_tokens": 7.0,
        "prompt_cached_tokens": 3.0,
        "prompt_cache_creation_tokens": 7.0,
        "server_tool_use_web_search_requests": 2.0,
        "server_tool_use_web_fetch_requests": 1.0,
        "tokens": 28.0,
    }
    assert metadata == {
        "cache_creation_ephemeral_5m_input_tokens": 2,
        "cache_creation_ephemeral_1h_input_tokens": 5,
        "usage_service_tier": "standard",
    }


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


@pytest.mark.vcr
def test_anthropic_messages_model_params_inputs(memory_logger):
    assert not memory_logger.pop()
    client = wrap_anthropic(_get_client())

    kw = {
        "model": MODEL,
        "max_tokens": 300,
        "system": "just return the number",
        "messages": [{"role": "user", "content": "what is 1+1?"}],
        "top_p": 0.5,
    }
    if MODEL == LEGACY_MODEL:
        kw["temperature"] = 0.5

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
        assert log["output"]["role"] == "assistant"
        assert "2" in log["output"]["content"][0]["text"]
        assert log["metadata"]["model"] == MODEL
        assert log["metadata"]["max_tokens"] == 300
        if MODEL == LEGACY_MODEL:
            assert log["metadata"]["temperature"] == 0.5
        assert log["metadata"]["top_p"] == 0.5


@pytest.mark.vcr
def test_anthropic_messages_system_prompt_inputs(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    system = "Today's date is 2024-03-26. Only return the date"
    q = [{"role": "user", "content": "what is tomorrow's date? only return the date"}]

    args = {
        "messages": q,
        "temperature": 0,
        "max_tokens": 300,
        "system": system,
        "model": MODEL,
    }

    def _with_messages_create():
        return client.messages.create(**args)

    def _with_messages_stream():
        with client.messages.stream(**args) as stream:
            for msg in stream:
                pass
        return stream.get_final_message()

    for f in [_with_messages_create, _with_messages_stream]:
        msg = f()
        assert "2024-03-27" in msg.content[0].text

        logs = memory_logger.pop()
        assert len(logs) == 1
        log = logs[0]
        inputs = log["input"]
        assert len(inputs) == 2
        inputs_by_role = {m["role"]: m["content"] for m in inputs}
        assert inputs_by_role["system"] == system
        assert inputs_by_role["user"] == q[0]["content"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_messages_create_async(memory_logger):
    assert not memory_logger.pop()

    params = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "what is 6+1?, just return the number"}],
    }

    client = wrap_anthropic(anthropic.AsyncAnthropic())
    msg = await client.messages.create(**params)
    assert "7" in msg.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["max_tokens"] == 100
    assert span["input"] == params["messages"]
    assert span["output"]["role"] == "assistant"
    assert "7" in span["output"]["content"][0]["text"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_messages_create_async_stream_true(memory_logger):
    assert not memory_logger.pop()

    params = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "what is 6+1?, just return the number"}],
        "stream": True,
    }

    client = wrap_anthropic(anthropic.AsyncAnthropic())
    stream = await client.messages.create(**params)
    async for event in stream:
        pass

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["max_tokens"] == 100
    assert span["input"] == params["messages"]
    assert span["output"]["role"] == "assistant"
    assert "7" in span["output"]["content"][0]["text"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_messages_streaming_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_async_client())
    msgs_in = [{"role": "user", "content": "what is 1+1?, just return the number"}]

    start = time.time()
    msg_out = None

    async with client.messages.stream(max_tokens=1024, messages=msgs_in, model=MODEL) as stream:
        async for event in stream:
            pass
        msg_out = await stream.get_final_message()
        assert msg_out.content[0].text == "2"
        usage = msg_out.usage
    end = time.time()

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "1+1" in str(log["input"])
    assert "2" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == 1024
    _assert_metrics_are_valid(log["metrics"], start, end)
    metrics = log["metrics"]
    assert metrics["prompt_tokens"] == usage.input_tokens
    assert metrics["completion_tokens"] == usage.output_tokens
    assert metrics["tokens"] == usage.input_tokens + usage.output_tokens
    assert metrics["prompt_cached_tokens"] == usage.cache_read_input_tokens
    assert metrics["prompt_cache_creation_tokens"] == usage.cache_creation_input_tokens
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == 1024


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
def test_anthropic_messages_streaming_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    msg_in = {"role": "user", "content": "what is 2+2? (just the number)"}

    start = time.time()
    with client.messages.stream(model=MODEL, max_tokens=300, messages=[msg_in]) as stream:
        msgs_out = [m for m in stream]
    end = time.time()
    msg_out = stream.get_final_message()
    usage = msg_out.usage
    # crudely check that the stream is valid
    assert len(msgs_out) > 3
    assert 1 <= len([m for m in msgs_out if m.type == "text"])
    assert msgs_out[0].type == "message_start"
    assert msgs_out[-1].type == "message_stop"

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "2+2" in str(log["input"])
    assert "4" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    _assert_metrics_are_valid(log["metrics"], start, end)
    assert log["metrics"]["prompt_tokens"] == usage.input_tokens
    assert log["metrics"]["completion_tokens"] == usage.output_tokens
    assert log["metrics"]["tokens"] == usage.input_tokens + usage.output_tokens
    assert log["metrics"]["prompt_cached_tokens"] == usage.cache_read_input_tokens
    assert log["metrics"]["prompt_cache_creation_tokens"] == usage.cache_creation_input_tokens


@pytest.mark.vcr
def test_anthropic_messages_streaming_sync_text_stream(memory_logger):
    """time_to_first_token is captured when iterating via .text_stream (BT-4702)."""
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    msg_in = {"role": "user", "content": "what is 2+2? (just the number)"}

    start = time.time()
    with client.messages.stream(model=MODEL, max_tokens=300, messages=[msg_in]) as stream:
        texts = list(stream.text_stream)
    end = time.time()
    msg_out = stream.get_final_message()
    usage = msg_out.usage

    text = "".join(texts)
    assert "4" in text
    assert "4" in msg_out.content[0].text

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "2+2" in str(log["input"])
    assert "4" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == 300
    assert log["output"]["role"] == "assistant"
    assert log["output"]["model"] == msg_out.model
    assert log["output"]["stop_reason"] == msg_out.stop_reason
    _assert_metrics_are_valid(log["metrics"], start, end)
    assert log["metrics"]["prompt_tokens"] == usage.input_tokens
    assert log["metrics"]["completion_tokens"] == usage.output_tokens
    assert log["metrics"]["tokens"] == usage.input_tokens + usage.output_tokens
    assert log["metrics"]["prompt_cached_tokens"] == usage.cache_read_input_tokens
    assert log["metrics"]["prompt_cache_creation_tokens"] == usage.cache_creation_input_tokens


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_messages_streaming_async_text_stream(memory_logger):
    """time_to_first_token is captured when iterating via .text_stream on async streams (BT-4702)."""
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_async_client())
    msgs_in = [{"role": "user", "content": "what is 1+1?, just return the number"}]

    start = time.time()
    async with client.messages.stream(max_tokens=1024, messages=msgs_in, model=MODEL) as stream:
        texts = [t async for t in stream.text_stream]
        msg_out = await stream.get_final_message()
        usage = msg_out.usage
    end = time.time()

    text = "".join(texts)
    assert "2" in text
    assert msg_out.content[0].text == "2"

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "1+1" in str(log["input"])
    assert "2" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == 1024
    assert log["output"]["role"] == "assistant"
    assert log["output"]["model"] == msg_out.model
    assert log["output"]["stop_reason"] == msg_out.stop_reason
    _assert_metrics_are_valid(log["metrics"], start, end)
    assert log["metrics"]["prompt_tokens"] == usage.input_tokens
    assert log["metrics"]["completion_tokens"] == usage.output_tokens
    assert log["metrics"]["tokens"] == usage.input_tokens + usage.output_tokens
    assert log["metrics"]["prompt_cached_tokens"] == usage.cache_read_input_tokens
    assert log["metrics"]["prompt_cache_creation_tokens"] == usage.cache_creation_input_tokens


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
    assert tool_span["output"][0]["type"] == "web_search_result"
    assert tool_span["output"][0]["url"] == text_block.text
    assert tool_span["output"][0]["encrypted_content"] == "<redacted>"
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


@pytest.mark.vcr
def test_anthropic_beta_messages_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    msg_in = {"role": "user", "content": "what's 3+3?"}

    start = time.time()
    msg = client.beta.messages.create(model=MODEL, max_tokens=300, messages=[msg_in])
    end = time.time()

    text = msg.content[0].text
    assert text
    assert "6" in text

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "3+3" in str(log["input"])
    assert "6" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_id"]
    assert log["root_span_id"]
    attrs = log["span_attributes"]
    assert attrs["type"] == "llm"
    assert "anthropic" in attrs["name"]
    metrics = log["metrics"]
    _assert_metrics_are_valid(metrics, start, end)
    assert log["metadata"]["model"] == MODEL


@pytest.mark.vcr
def test_anthropic_beta_messages_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_client())
    msg_in = {"role": "user", "content": "what is 5+5? (just the number)"}

    start = time.time()
    with client.beta.messages.stream(model=MODEL, max_tokens=300, messages=[msg_in]) as stream:
        msgs_out = [m for m in stream]
    end = time.time()
    msg_out = stream.get_final_message()
    usage = msg_out.usage

    assert len(msgs_out) > 3
    assert msgs_out[0].type == "message_start"
    assert msgs_out[-1].type == "message_stop"
    assert "10" in msg_out.content[0].text

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "5+5" in str(log["input"])
    assert "10" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    _assert_metrics_are_valid(log["metrics"], start, end)
    assert log["metrics"]["prompt_tokens"] == usage.input_tokens
    assert log["metrics"]["completion_tokens"] == usage.output_tokens
    assert log["metrics"]["tokens"] == usage.input_tokens + usage.output_tokens


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_beta_messages_create_async(memory_logger):
    assert not memory_logger.pop()

    params = {
        "model": MODEL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "what is 8+2?, just return the number"}],
    }

    client = wrap_anthropic(anthropic.AsyncAnthropic())
    msg = await client.beta.messages.create(**params)
    assert "10" in msg.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["max_tokens"] == 100
    assert span["input"] == params["messages"]
    assert span["output"]["role"] == "assistant"
    assert "10" in span["output"]["content"][0]["text"]


@pytest.mark.vcr(
    match_on=["method", "scheme", "host", "port", "path", "body"]
)  # exclude query - varies by SDK version
@pytest.mark.asyncio
async def test_anthropic_beta_messages_streaming_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_anthropic(_get_async_client())
    msgs_in = [{"role": "user", "content": "what is 9+1?, just return the number"}]

    start = time.time()
    msg_out = None

    async with client.beta.messages.stream(max_tokens=1024, messages=msgs_in, model=MODEL) as stream:
        async for event in stream:
            pass
        msg_out = await stream.get_final_message()
        assert "10" in msg_out.content[0].text
        usage = msg_out.usage
    end = time.time()

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert "user" in str(log["input"])
    assert "9+1" in str(log["input"])
    assert "10" in str(log["output"])
    assert log["project_id"] == PROJECT_NAME
    assert log["span_attributes"]["type"] == "llm"
    assert log["metadata"]["model"] == MODEL
    assert log["metadata"]["max_tokens"] == 1024
    _assert_metrics_are_valid(log["metrics"], start, end)
    metrics = log["metrics"]
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
        assert event_types[0] == "session.status_running"
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
        assert streamed_output_types[0] == "session.status_running"
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
    assert metrics["prompt_cache_creation_tokens"] == usage.cache_creation_input_tokens
    assert span["metadata"]["cache_creation_ephemeral_5m_input_tokens"] == ephemeral_5m
    assert span["metadata"]["cache_creation_ephemeral_1h_input_tokens"] == ephemeral_1h
    assert "service_tier" not in metrics


class TestAutoInstrumentAnthropic:
    def test_auto_instrument_anthropic(self):
        verify_autoinstrument_script("test_auto_anthropic.py")


def test_extract_anthropic_usage_preserves_nested_numeric_fields():
    usage = {
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
    }
    metrics, metadata = extract_anthropic_usage(usage)

    assert metrics["prompt_tokens"] == 15
    assert metrics["completion_tokens"] == 12
    assert metrics["tokens"] == 27
    assert metrics["prompt_cache_creation_tokens"] == 7
    assert metadata["cache_creation_ephemeral_5m_input_tokens"] == 3
    assert metadata["cache_creation_ephemeral_1h_input_tokens"] == 4
    assert metrics["server_tool_use_web_search_requests"] == 2
    assert metrics["server_tool_use_web_fetch_requests"] == 1
    assert "service_tier" not in metrics
    assert metadata == {
        "cache_creation_ephemeral_5m_input_tokens": 3,
        "cache_creation_ephemeral_1h_input_tokens": 4,
        "usage_service_tier": "standard",
        "usage_inference_geo": "not_available",
    }


def test_extract_anthropic_usage_skips_empty_usage():
    metrics, metadata = extract_anthropic_usage(SimpleNamespace())

    assert metrics == {}
    assert metadata == {}


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
    def test_sync_batches_create_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        result = client.messages.batches.create(requests=_make_batch_requests())

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
    @pytest.mark.asyncio
    async def test_async_batches_create_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client())
        result = await client.messages.batches.create(requests=_make_batch_requests())

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

    def test_sync_batches_results_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        mock_decoder = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "anthropic.resources.messages.batches.Batches.results",
            return_value=mock_decoder,
        ):
            result = client.messages.batches.results("msgbatch_abc123")

        assert result is mock_decoder

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["input"]["message_batch_id"] == "msgbatch_abc123"
        assert span["output"]["type"] == "jsonl_stream"

    @pytest.mark.asyncio
    async def test_async_batches_results_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client())
        mock_decoder = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "anthropic.resources.messages.batches.AsyncBatches.results",
            return_value=mock_decoder,
        ):
            result = await client.messages.batches.results(message_batch_id="msgbatch_abc456")

        assert result is mock_decoder

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["input"]["message_batch_id"] == "msgbatch_abc456"

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


class TestBetaBatchesCreateSpans:
    """Tests verifying that beta.messages.batches.create() produces correct spans."""

    @pytest.mark.vcr
    def test_sync_beta_batches_create_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        result = client.beta.messages.batches.create(requests=_make_batch_requests())

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
    @pytest.mark.asyncio
    async def test_async_beta_batches_create_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client())
        result = await client.beta.messages.batches.create(requests=_make_batch_requests())

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


class TestBetaBatchesResultsSpans:
    """Tests verifying that beta.messages.batches.results() produces correct spans.

    Mocked because the batch results API requires a completed batch, and batches
    can take up to 24 hours to finish processing.
    """

    def test_sync_beta_batches_results_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_client())
        mock_decoder = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "anthropic.resources.beta.messages.batches.Batches.results",
            return_value=mock_decoder,
        ):
            result = client.beta.messages.batches.results("msgbatch_beta123")

        assert result is mock_decoder

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["input"]["message_batch_id"] == "msgbatch_beta123"
        assert span["output"]["type"] == "jsonl_stream"

    @pytest.mark.asyncio
    async def test_async_beta_batches_results_produces_span(self, memory_logger):
        assert not memory_logger.pop()

        client = wrap_anthropic(_get_async_client())
        mock_decoder = unittest.mock.MagicMock()
        with unittest.mock.patch(
            "anthropic.resources.beta.messages.batches.AsyncBatches.results",
            return_value=mock_decoder,
        ):
            result = await client.beta.messages.batches.results(message_batch_id="msgbatch_beta456")

        assert result is mock_decoder

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == "anthropic.messages.batches.results"
        assert span["metadata"]["provider"] == "anthropic"
        assert span["input"]["message_batch_id"] == "msgbatch_beta456"
