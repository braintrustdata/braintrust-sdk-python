"""Exercise aiobotocore auto-instrumentation before and after provider import."""

import asyncio
import importlib
import os
import sys

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


import_first = os.getenv("BRAINTRUST_TEST_AIOBOTOCORE_IMPORT_FIRST") == "1"
if import_first:
    importlib.import_module("aiobotocore.session")

assert ("aiobotocore.session" in sys.modules) == import_first
assert auto_instrument()["bedrock_runtime"] is True
assert auto_instrument()["bedrock_runtime"] is True

from aiobotocore.session import get_session


async def main():
    with autoinstrument_test_context("test_auto_bedrock_runtime", integration="bedrock_runtime") as memory_logger:
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        model = os.getenv("BRAINTRUST_BEDROCK_CONVERSE_MODEL", "us.amazon.nova-lite-v1:0")
        async with get_session().create_client("bedrock-runtime", region_name=region) as client:
            response = await client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": "Say hello in one word."}]}],
                inferenceConfig={"maxTokens": 20, "temperature": 0},
            )
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == model
        assert span["metadata"]["provider"] == "bedrock"
        assert span["span_attributes"]["name"] == "bedrock.converse"
        assert span["output"][0]["content"] == [
            {"type": "text", "text": block["text"]} for block in response["output"]["message"]["content"]
        ]
        assert span["metrics"]["tokens"] == response["usage"]["totalTokens"]
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "bedrock-runtime-auto"


asyncio.run(main())
print("SUCCESS")
