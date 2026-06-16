"""Tests for the boto3 Bedrock Runtime integration."""

import inspect
import json
import os
import time

import pytest
from braintrust import logger
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


@pytest.mark.vcr
def test_auto_instrument_bedrock_runtime_subprocess():
    verify_autoinstrument_script("test_auto_bedrock_runtime.py", timeout=60)
