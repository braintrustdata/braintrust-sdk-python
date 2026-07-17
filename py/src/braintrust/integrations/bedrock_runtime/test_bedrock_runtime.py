"""Tests for the boto3 Bedrock Runtime integration."""

import base64
import inspect
import json
import os
import time

import pytest
from braintrust import Attachment, logger
from braintrust.integrations.bedrock_runtime import BedrockRuntimeIntegration, setup_bedrock, wrap_bedrock
from braintrust.integrations.bedrock_runtime.patchers import (
    BedrockClientCreatorPatcher,
    BedrockRuntimeClientMethodsPatcher,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


pytest.importorskip("boto3")
pytest.importorskip("botocore")

import boto3  # noqa: E402
import botocore.client  # noqa: E402


PROJECT_NAME = "test-boto3-bedrock"
# Match the Java SDK Bedrock test defaults in Bedrock30TestUtils / BraintrustAWSBedrockTest.
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
CONVERSE_MODEL = os.getenv("BRAINTRUST_BEDROCK_CONVERSE_MODEL", "us.amazon.nova-lite-v1:0")
CONVERSE_STREAM_MODEL = os.getenv("BRAINTRUST_BEDROCK_CONVERSE_STREAM_MODEL", "us.amazon.nova-lite-v1:0")
INVOKE_MODEL = os.getenv("BRAINTRUST_BEDROCK_INVOKE_MODEL", "us.amazon.nova-lite-v1:0")


os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
if not os.environ.get("AWS_PROFILE") and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")


@pytest.fixture(scope="module")
def vcr_config():
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"
    return {
        "record_mode": record_mode,
        "decode_compressed_response": True,
        "filter_headers": [
            "authorization",
            "Authorization",
            "x-amz-security-token",
            "X-Amz-Security-Token",
            "x-amz-date",
            "X-Amz-Date",
            "x-amz-content-sha256",
            "X-Amz-Content-Sha256",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
            "x-amzn-bedrock-api-key",
        ],
    }


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture
def clean_client_creator():
    original = inspect.getattr_static(botocore.client.ClientCreator, "create_client")
    marker = BedrockClientCreatorPatcher.patch_marker_attr()
    try:
        yield
    finally:
        setattr(botocore.client.ClientCreator, "create_client", original)
        for obj in (botocore.client.ClientCreator, original):
            if hasattr(obj, marker):
                try:
                    delattr(obj, marker)
                except AttributeError:
                    pass


def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _converse_kwargs(model_id=CONVERSE_MODEL):
    return {
        "modelId": model_id,
        "system": [{"text": "You answer with concise lowercase text."}],
        "messages": [{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        "inferenceConfig": {"maxTokens": 20, "temperature": 0, "topP": 0.9, "stopSequences": ["STOP"]},
    }


def _assert_converse_span(span, *, name: str, start: float, end: float, model_id=CONVERSE_MODEL):
    assert span["span_attributes"]["name"] == name
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "bedrock"
    assert span["metadata"]["model"] == model_id
    assert span["metadata"]["max_tokens"] == 20
    assert span["metadata"]["temperature"] == 0
    assert span["metadata"]["top_p"] == 0.9
    assert span["metadata"]["stop_sequences"] == ["STOP"]

    assert span["input"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "You answer with concise lowercase text."}],
    }
    assert span["input"][1] == {
        "role": "user",
        "content": [{"type": "text", "text": "Say hello in one word."}],
    }
    assert span["output"][0]["role"] == "assistant"
    assert span["output"][0]["content"]
    assert span["output"][0]["content"][0]["type"] == "text"
    assert "stop_reason" in span["metadata"]
    assert_metrics_are_valid(span["metrics"], start, end)


# ---------------------------------------------------------------------------
# Local integration boilerplate tests
# ---------------------------------------------------------------------------


def test_integration_metadata_and_min_version():
    assert BedrockRuntimeIntegration.name == "bedrock_runtime"
    assert BedrockRuntimeIntegration.min_version == "1.34.116"
    assert BedrockRuntimeIntegration.available_patchers() == ("bedrock_runtime.client_creator",)


def test_wrap_bedrock_returns_non_bedrock_clients_unchanged():
    client = _s3_client()

    assert wrap_bedrock(client) is client
    assert not BedrockRuntimeClientMethodsPatcher.is_patched(None, None, target=client)


def test_wrap_bedrock_is_idempotent():
    client = _bedrock_client()

    assert wrap_bedrock(client) is client
    assert wrap_bedrock(client) is client
    assert BedrockRuntimeClientMethodsPatcher.is_patched(None, None, target=client)


def test_setup_bedrock_wraps_only_new_bedrock_clients(clean_client_creator):
    assert setup_bedrock() is True
    assert setup_bedrock() is True

    bedrock = _bedrock_client()
    s3 = _s3_client()

    assert BedrockRuntimeClientMethodsPatcher.is_patched(None, None, target=bedrock)
    assert not BedrockRuntimeClientMethodsPatcher.is_patched(None, None, target=s3)


# ---------------------------------------------------------------------------
# VCR-backed integration tests
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_wrap_bedrock_converse(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())

    start = time.time()
    response = client.converse(**_converse_kwargs())
    end = time.time()

    assert response["output"]["message"]["role"] == "assistant"
    assert response["usage"]["totalTokens"] > 0

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_converse_span(spans[0], name="bedrock.converse", start=start, end=end)
    assert spans[0]["metadata"]["endpoint"] == "converse"


@pytest.mark.vcr
def test_setup_bedrock_converse_auto_wraps_new_clients(memory_logger, clean_client_creator):
    assert setup_bedrock() is True
    assert not memory_logger.pop()

    client = _bedrock_client()
    start = time.time()
    response = client.converse(**_converse_kwargs())
    end = time.time()

    assert response["output"]["message"]["role"] == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_converse_span(spans[0], name="bedrock.converse", start=start, end=end)


@pytest.mark.vcr
def test_wrap_bedrock_converse_stream(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())

    start = time.time()
    response = client.converse_stream(**_converse_kwargs(CONVERSE_STREAM_MODEL))
    chunks = []
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                chunks.append(delta["text"])
    end = time.time()

    assert "".join(chunks).strip()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "bedrock-runtime-auto"
    _assert_converse_span(span, name="bedrock.converse-stream", start=start, end=end, model_id=CONVERSE_STREAM_MODEL)
    assert span["metadata"]["endpoint"] == "converse-stream"
    assert span["metadata"]["stream"] is True
    assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.vcr
def test_wrap_bedrock_invoke_model_preserves_response_body_and_logs_json(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())
    body = {
        "messages": [{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        "inferenceConfig": {"max_new_tokens": 20, "temperature": 0},
    }

    start = time.time()
    response = client.invoke_model(
        modelId=INVOKE_MODEL,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    end = time.time()

    # The integration reads the body for tracing and must replace it so user code
    # can still consume the provider response normally.
    response_body = json.loads(response["body"].read())
    assert response_body["output"]["message"]["content"]

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "bedrock.invoke_model"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "bedrock"
    assert span["metadata"]["endpoint"] == "invoke_model"
    assert span["metadata"]["model"] == INVOKE_MODEL
    assert span["input"]["modelId"] == INVOKE_MODEL
    assert span["input"]["body"] == body
    assert span["output"] == response_body
    assert_metrics_are_valid(span["metrics"], start, end)


# A 1x1 transparent PNG for image tests.
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# A minimal valid PDF for document tests.
_TINY_PDF_BYTES = (
    b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 8 8]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f\n"
    b"0000000010 00000 n\n0000000053 00000 n\n0000000096 00000 n\n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n144\n%%EOF"
)


_WEATHER_TOOL = {
    "toolSpec": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            }
        },
    }
}


def _converse_tool_kwargs(model_id=CONVERSE_MODEL):
    return {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "What's the weather in Paris? Use the tool."}]}],
        "inferenceConfig": {"maxTokens": 200, "temperature": 0},
        "toolConfig": {
            "tools": [_WEATHER_TOOL],
            "toolChoice": {"tool": {"name": "get_weather"}},
        },
    }


@pytest.mark.vcr
def test_wrap_bedrock_converse_normalizes_tool_config_and_output(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())

    response = client.converse(**_converse_tool_kwargs())

    assert response["output"]["message"]["role"] == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    # Tools are normalized to OpenAI shape and live in metadata.
    assert span["metadata"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }
    ]
    assert span["metadata"]["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    # No raw Bedrock-shaped copy left behind.
    assert "tool_config" not in span["metadata"]

    # Output tool_use block is normalized.
    content = span["output"][0]["content"]
    tool_uses = [block for block in content if block.get("type") == "tool_use"]
    assert tool_uses, f"expected a tool_use block in {content!r}"
    tool_use = tool_uses[0]
    assert tool_use.get("name") == "get_weather"
    assert "id" in tool_use
    assert isinstance(tool_use.get("input"), dict)


@pytest.mark.vcr
def test_wrap_bedrock_converse_with_image_input_materializes_attachment(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())

    kwargs = {
        "modelId": CONVERSE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Describe this image in one word."},
                    {"image": {"format": "png", "source": {"bytes": _TINY_PNG_BYTES}}},
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 20, "temperature": 0},
    }
    response = client.converse(**kwargs)
    assert response["output"]["message"]["role"] == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    parts = span["input"][0]["content"]
    image_part = next(p for p in parts if p.get("type") == "image_url")
    assert image_part["format"] == "png"
    assert isinstance(image_part["image_url"]["url"], Attachment)
    assert image_part["image_url"]["url"].reference["content_type"] == "image/png"
    # No raw bytes hanging around next to the attachment.
    assert "source" not in image_part
    # Original request kwargs are not mutated.
    assert kwargs["messages"][0]["content"][1]["image"]["source"]["bytes"] == _TINY_PNG_BYTES


@pytest.mark.vcr
def test_wrap_bedrock_converse_with_document_input_materializes_attachment(memory_logger):
    assert not memory_logger.pop()
    client = wrap_bedrock(_bedrock_client())
    if "document" not in client.meta.service_model.shape_for("ContentBlock").members:
        pytest.skip("installed botocore does not support Converse document content blocks")

    kwargs = {
        "modelId": CONVERSE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Summarize this document."},
                    {
                        "document": {
                            "format": "pdf",
                            "name": "tiny-pdf",
                            "source": {"bytes": _TINY_PDF_BYTES},
                        }
                    },
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 40, "temperature": 0},
    }
    response = client.converse(**kwargs)
    assert response["output"]["message"]["role"] == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    parts = span["input"][0]["content"]
    file_part = next(p for p in parts if p.get("type") == "file")
    assert file_part["format"] == "pdf"
    assert file_part["name"] == "tiny-pdf"
    assert file_part["file"]["filename"] == "tiny-pdf"
    assert isinstance(file_part["file"]["file_data"], Attachment)
    assert file_part["file"]["file_data"].reference["content_type"] == "application/pdf"
    assert "source" not in file_part


@pytest.mark.vcr
def test_auto_instrument_bedrock_runtime_subprocess():
    verify_autoinstrument_script("test_auto_bedrock_runtime.py", timeout=60)
