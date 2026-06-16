"""Test auto_instrument for boto3 Bedrock Runtime."""

import os


os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
if not os.environ.get("AWS_PROFILE") and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

import boto3
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


MODEL = os.getenv("BRAINTRUST_BEDROCK_CONVERSE_MODEL", "us.amazon.nova-lite-v1:0")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

results = auto_instrument()
assert results.get("bedrock_runtime") is True

results2 = auto_instrument()
assert results2.get("bedrock_runtime") is True

with autoinstrument_test_context("test_auto_bedrock_runtime", integration="bedrock_runtime") as memory_logger:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    assert response["output"]["message"]["role"] == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "bedrock"
    assert span["metadata"]["model"] == MODEL
    assert span["metadata"]["endpoint"] == "converse"
    assert span["span_attributes"]["name"] == "bedrock.converse"

print("SUCCESS")
