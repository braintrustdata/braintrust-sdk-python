"""Bedrock tracing with real aiobotocore clients and shared boto3 recordings."""

import asyncio
import gc
import inspect
import json
import os
import time

import pytest
from braintrust import current_span
from braintrust.integrations.bedrock_runtime import setup_bedrock, wrap_bedrock
from braintrust.integrations.bedrock_runtime.patchers import BedrockRuntimeClientMethodsPatcher
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script


pytest.importorskip("aiobotocore")

import aiobotocore.client  # noqa: E402
import botocore.client  # noqa: E402
from aiobotocore.session import get_session  # noqa: E402


REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
MODEL = os.getenv("BRAINTRUST_BEDROCK_CONVERSE_MODEL", "us.amazon.nova-lite-v1:0")


def _converse_kwargs():
    return {
        "modelId": MODEL,
        "system": [{"text": "You answer with concise lowercase text."}],
        "messages": [{"role": "user", "content": [{"text": "Say hello in one word."}]}],
        "inferenceConfig": {"maxTokens": 20, "temperature": 0, "topP": 0.9, "stopSequences": ["STOP"]},
    }


def _invoke_kwargs():
    return {
        "modelId": MODEL,
        "body": json.dumps(
            {
                "messages": [{"role": "user", "content": [{"text": "Say hello in one word."}]}],
                "inferenceConfig": {"max_new_tokens": 20, "temperature": 0},
            }
        ),
        "contentType": "application/json",
        "accept": "application/json",
    }


@pytest.fixture(autouse=True)
def clean_client_creators():
    # AioClientCreator overrides create_client, so restore both factories.
    creators = (botocore.client.ClientCreator, aiobotocore.client.AioClientCreator)
    originals = [(creator, inspect.getattr_static(creator, "create_client")) for creator in creators]
    try:
        yield
    finally:
        for creator, original in originals:
            creator.create_client = original
            for obj in (creator, original):
                for name in list(vars(obj)):
                    if name.startswith("__braintrust_patched_bedrock_runtime_"):
                        delattr(obj, name)


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["wrap", "setup"])
@pytest.mark.parametrize(
    "operation,cassette",
    [
        ("converse", "test_wrap_bedrock_converse.yaml"),
        ("converse_stream", "test_wrap_bedrock_converse_stream.yaml"),
        ("invoke_model", "test_wrap_bedrock_invoke_model_preserves_response_body_and_logs_json.yaml"),
    ],
)
async def test_aiobotocore_tracing(entry, operation, cassette, memory_logger, vcr):
    if entry == "setup":
        assert setup_bedrock()
        assert setup_bedrock()
    async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
        if entry == "wrap":
            assert wrap_bedrock(client) is client
            assert wrap_bedrock(client) is client
        with vcr.use_cassette(cassette):
            parent = current_span()
            start = time.time()
            if operation == "invoke_model":
                body = {
                    "messages": [{"role": "user", "content": [{"text": "Say hello in one word."}]}],
                    "inferenceConfig": {"max_new_tokens": 20, "temperature": 0},
                }
                response = await client.invoke_model(
                    modelId=MODEL, body=json.dumps(body), contentType="application/json", accept="application/json"
                )
                response_body = response["body"]
                assert response_body.tell() == 0
                async with response_body as stream:
                    prefix = await stream.read(4)
                    data = prefix + b"".join([chunk async for chunk in stream.iter_chunks(17)])
                    output = json.loads(data)
                    assert output["output"]["message"]["content"]
                    assert response_body.tell() == len(data)
                    assert await stream.read() == b""
            else:
                response = await getattr(client, operation)(**_converse_kwargs())
                if operation == "converse_stream":
                    # Stream spans remain open until the async stream is consumed.
                    assert all("end" not in record.get().get("metrics", {}) for record in memory_logger.logs)
                    events = [event async for event in response["stream"]]
                    text = "".join(
                        event["contentBlockDelta"]["delta"].get("text", "")
                        for event in events
                        if "contentBlockDelta" in event
                    )
                    assert text
                    response["stream"].close()
                else:
                    assert response["output"]["message"]["role"] == "assistant"
                    assert response["usage"]["totalTokens"] > 0
            end = time.time()
            assert current_span() is parent
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["output"]
        assert span["span_attributes"]["type"] == "llm"
        assert span["metadata"]["provider"] == "bedrock"
        assert span["metadata"]["model"] == MODEL
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "bedrock-runtime-auto"
        assert_metrics_are_valid(span["metrics"], start, end)
        if operation == "invoke_model":
            assert span["output"] == output
            assert span["input"]["body"] == body
        else:
            assert span["output"][0]["role"] == "assistant"
            assert span["input"][0]["role"] == "system"
            if operation == "converse_stream":
                assert span["output"][0]["content"][0]["text"] == text
                assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.asyncio
async def test_aiobotocore_setup_leaves_other_services_unchanged():
    assert setup_bedrock()
    async with get_session().create_client("s3", region_name=REGION) as client:
        assert wrap_bedrock(client) is client
        assert not BedrockRuntimeClientMethodsPatcher.is_patched(None, None, target=client)


@pytest.mark.parametrize("import_first", [True, False])
def test_auto_instrument_aiobotocore_subprocess(monkeypatch, import_first):
    monkeypatch.setenv("BRAINTRUST_TEST_AIOBOTOCORE_IMPORT_FIRST", str(int(import_first)))
    verify_autoinstrument_script("test_auto_aiobotocore.py")


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["wrap", "setup"])
async def test_aiobotocore_invoke_model_stream(entry, memory_logger, vcr):
    if entry == "setup":
        assert setup_bedrock()
    with vcr.use_cassette("test_aiobotocore_invoke_model_stream.yaml"):
        async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
            if entry == "wrap":
                wrap_bedrock(client)
            start = time.time()
            response = await client.invoke_model_with_response_stream(**_invoke_kwargs())
            chunks = [json.loads(event["chunk"]["bytes"]) async for event in response["body"] if "chunk" in event]
            end = time.time()
            response["body"].close()
            spans = memory_logger.pop()
            assert len(spans) == 1
            span = spans[0]
            assert span["span_attributes"]["name"] == "bedrock.invoke_model_stream"
            assert span["output"] == chunks
            assert span["metrics"]["time_to_first_token"] >= 0
            assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["close", "drop", "drop_unread"])
@pytest.mark.parametrize("operation", ["converse_stream", "invoke_model_with_response_stream"])
async def test_aiobotocore_stream_cleanup(cleanup, operation, memory_logger, vcr):
    converse = operation == "converse_stream"
    cassette = "test_wrap_bedrock_converse_stream.yaml" if converse else "test_aiobotocore_invoke_model_stream.yaml"
    with vcr.use_cassette(cassette):
        async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
            wrap_bedrock(client)
            parent = current_span()
            response = await getattr(client, operation)(**(_converse_kwargs() if converse else _invoke_kwargs()))
            stream = response.pop("stream" if converse else "body")
            if cleanup != "drop_unread":
                async for event in stream:
                    if ("contentBlockDelta" if converse else "chunk") in event:
                        break
            if cleanup == "close":
                stream.close()
                stream.close()
            del stream
            gc.collect()
            assert current_span() is parent
            spans = memory_logger.pop()
            assert len(spans) == 1
            span = spans[0]
            assert "end" in span["metrics"]
            if cleanup == "drop_unread":
                assert "output" not in span
            else:
                assert span["output"]
                if converse:
                    assert span["output"][0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_aiobotocore_validation_error_is_logged(memory_logger):
    from botocore.exceptions import ParamValidationError

    async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
        wrap_bedrock(client)
        kwargs = _converse_kwargs()
        kwargs["inferenceConfig"]["maxTokens"] = "invalid"
        with pytest.raises(ParamValidationError):
            await client.converse(**kwargs)
        spans = memory_logger.pop()
        assert len(spans) == 1
        assert "maxTokens" in spans[0]["error"]
        assert "end" in spans[0]["metrics"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_aiobotocore_stream_cancellation_is_logged(monkeypatch, memory_logger, vcr):
    # Recordings cannot reproduce task cancellation. Inject it at the real
    # stream's parser while replaying a normal provider stream.
    def cancel_parse(_event):
        raise asyncio.CancelledError("cancelled while streaming")

    async def cancel_parse_async(event):
        return cancel_parse(event)

    with vcr.use_cassette("test_wrap_bedrock_converse_stream.yaml"):
        async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
            wrap_bedrock(client)
            response = await client.converse_stream(**_converse_kwargs())
            stream = response["stream"]
            parser = stream._stream._parse_event
            monkeypatch.setattr(
                stream._stream,
                "_parse_event",
                cancel_parse_async if inspect.iscoroutinefunction(parser) else cancel_parse,
            )
            with pytest.raises(asyncio.CancelledError):
                await anext(aiter(stream))
            stream.close()
            spans = memory_logger.pop()
            assert len(spans) == 1
            assert "cancelled while streaming" in spans[0]["error"]
            assert "end" in spans[0]["metrics"]


@pytest.mark.asyncio
async def test_aiobotocore_request_cancellation_is_logged(monkeypatch, memory_logger):
    # Cancellation has no HTTP representation; inject it at the real endpoint.
    async def cancel_request(*_args, **_kwargs):
        raise asyncio.CancelledError("cancelled before response")

    async with get_session().create_client("bedrock-runtime", region_name=REGION) as client:
        wrap_bedrock(client)
        monkeypatch.setattr(client._endpoint, "make_request", cancel_request)
        with pytest.raises(asyncio.CancelledError):
            await client.converse(**_converse_kwargs())
        spans = memory_logger.pop()
        assert len(spans) == 1
        assert "cancelled before response" in spans[0]["error"]
        assert "end" in spans[0]["metrics"]
