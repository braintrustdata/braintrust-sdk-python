"""Cassette-backed tests for Cursor SDK instrumentation."""

# pylint: disable=import-error

import base64
import os
from pathlib import Path

import pytest
import requests
from braintrust import Attachment, logger
from braintrust.integrations.anthropic import AnthropicIntegration
from braintrust.integrations.cursor_sdk import CursorSDKIntegration, setup_cursor_sdk
from braintrust.integrations.cursor_sdk._test_vcr import cursor_vcr_config
from braintrust.integrations.openai import OpenAIIntegration
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type, init_test_logger


PROJECT_NAME = "test-cursor-sdk"
TEST_MODEL = "composer-2.5"
_REQUIRES_SYNC_BRIDGE = pytest.mark.skipif(
    os.name == "nt",
    reason="cursor-sdk's sync bridge uses os.get_blocking(), which is unavailable on Windows",
)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as background_logger:
        yield background_logger


def _api_key() -> str:
    return os.environ.get("CURSOR_API_KEY", "crsr_test_key_for_cassette_playback")


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("This is a tiny cassette characterization workspace.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="module")
def vcr_config():
    return cursor_vcr_config()


def _call_custom_tool(client, agent_id: str, tool_name: str, call_id: str, args: dict):
    """Invoke a custom tool the way Cursor's bridge does.

    The bridge reaches the SDK by POSTing a Connect request to the loopback
    callback server it was handed at launch, so this posts the same request to
    the same server rather than standing anything in for it.
    """
    endpoint = client._owned_bridge._tool_callback_server.endpoint
    response = requests.post(
        f"{endpoint.url.rstrip('/')}/sdk.v1.SdkCustomToolCallbackService/CallCustomTool",
        json={"agentId": agent_id, "toolName": tool_name, "toolCallId": call_id, "args": args},
        headers={
            "Authorization": f"Bearer {endpoint.auth_token}",
            "Connect-Protocol-Version": "1",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def test_setup_cursor_sdk_is_idempotent():
    assert setup_cursor_sdk(project=PROJECT_NAME)
    assert setup_cursor_sdk(project=PROJECT_NAME)


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_auto_instrument_cursor_sdk_subprocess():
    verify_autoinstrument_script("test_auto_cursor_sdk.py", timeout=45)


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_characterize_no_downstream_provider_spans(memory_logger, tmp_path):
    """Cursor's bridge owns provider calls, so no Python provider LLM leaf exists."""
    import cursor_sdk

    assert OpenAIIntegration.setup()
    assert AnthropicIntegration.setup()

    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with cursor_sdk.Agent.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
            client=client,
        ) as agent:
            result = agent.send("Reply with exactly: cursor characterization complete").wait()

    assert result.status == "finished"
    assert "cursor characterization complete" in result.result.lower()
    spans = memory_logger.pop()
    origins = {span["context"]["span_origin"]["instrumentation"]["name"] for span in spans}
    assert origins == {"cursor-sdk-auto"}
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_sync_wait_traces_cursor_owned_model_turn(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with cursor_sdk.Agent.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
            client=client,
        ) as agent:
            result = agent.send("Reply with exactly: cursor tracing complete").wait()

    assert result.status == "finished"
    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(task_spans) == 1
    assert len(llm_spans) == 1

    task_span = task_spans[0]
    llm_span = llm_spans[0]
    assert task_span["span_attributes"]["name"] == "Cursor Agent"
    assert task_span["input"] == [{"role": "user", "content": "Reply with exactly: cursor tracing complete"}]
    assert task_span["output"] == result.result
    assert llm_span["span_attributes"]["name"] == "Cursor Model Turn"
    assert llm_span["span_parents"] == [task_span["span_id"]]
    assert llm_span["metadata"] == {"model": TEST_MODEL, "provider": "cursor"}
    assert llm_span["input"] == task_span["input"]
    assert llm_span["output"][-1]["message"]["content"] == result.result
    assert llm_span["output"][-1]["finish_reason"] == "stop"

    for span in spans:
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "cursor-sdk-auto"
    for metric in ("prompt_tokens", "completion_tokens", "tokens", "prompt_cached_tokens"):
        assert llm_span["metrics"][metric] >= 0
        assert task_span["metrics"][metric] == llm_span["metrics"][metric]
    assert llm_span["metrics"]["tokens"] == (
        llm_span["metrics"]["prompt_tokens"] + llm_span["metrics"]["completion_tokens"]
    )
    assert llm_span["metrics"]["time_to_first_token"] >= 0
    assert set(llm_span.get("metadata", {})) == {"model", "provider"}
    allowed_metrics = {
        "start",
        "end",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "time_to_first_token",
        "completion_reasoning_tokens",
        "prompt_cached_tokens",
        "prompt_cache_creation_tokens",
    }
    assert set(llm_span.get("metrics", {})) <= allowed_metrics
    assert set(task_span.get("metrics", {})) <= allowed_metrics


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.parametrize("consumer", ["messages", "stream", "events", "iteration", "iter_text", "text"])
@pytest.mark.vcr
def test_sync_consumption_paths_finalize_once(memory_logger, tmp_path, consumer):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            run = agent.send("Reply with exactly: cursor tracing complete")
            if consumer == "messages":
                consumed = list(run.messages())
            elif consumer == "stream":
                consumed = list(run.stream())
            elif consumer == "events":
                consumed = list(run.events())
            elif consumer == "iteration":
                consumed = list(run)
            elif consumer == "iter_text":
                consumed = list(run.iter_text())
            else:
                consumed = run.text()
            result = run.wait()

    assert consumed
    assert result.status == "finished"
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TASK)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_agent_prompt_one_shot(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        result = cursor_sdk.Agent.prompt(
            "Reply with exactly: cursor tracing complete",
            cursor_sdk.AgentOptions(
                model=TEST_MODEL,
                api_key=_api_key(),
                local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
            ),
            client=client,
        )

    assert result.status == "finished"
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TASK)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_resume_send_observe_finalizes_trace(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        agent = client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        )
        agent_id = agent.agent_id
        first_result = agent.send("Reply with exactly: cursor initial tracing complete").wait()
        assert first_result.status == "finished"
        agent.close()
        with client.agents.resume(
            agent_id,
            cursor_sdk.AgentOptions(
                model=TEST_MODEL,
                api_key=_api_key(),
                local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
            ),
        ) as resumed:
            run = resumed.send("Reply with exactly: cursor resumed tracing complete")
            observed = list(run.observe())

    assert any(event.result_is_full for event in observed)
    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(task_spans) == 2
    # observe() returns only the terminal envelope when attached after send;
    # Cursor does not replay model-turn messages on that stream, so the
    # observed run correctly remains a task rather than a token-less LLM.
    assert len(llm_spans) == 1
    assert {span.get("output") for span in task_spans} == {
        "cursor initial tracing complete",
        "cursor resumed tracing complete",
    }


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_messages_preserve_callbacks_and_trace(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    deltas = []
    steps = []

    async def on_delta(update):
        deltas.append(update)

    async with await cursor_sdk.AsyncClient.launch_bridge(workspace=str(workspace)) as client:
        async with await client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            run = await agent.send(
                "Reply with exactly: cursor async tracing complete",
                cursor_sdk.SendOptions(on_delta=on_delta, on_step=steps.append),
            )
            messages = [message async for message in run.messages()]
            result = await run.wait()

    assert result.status == "finished"
    assert any(message.type == "assistant" for message in messages)
    assert deltas
    assert steps
    spans = memory_logger.pop()
    (task_span,) = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    (llm_span,) = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert task_span["output"] == result.result
    assert llm_span["output"][-1]["message"]["content"] == result.result
    assert llm_span["span_parents"] == [task_span["span_id"]]
    assert all(span["context"]["span_origin"]["instrumentation"]["name"] == "cursor-sdk-auto" for span in spans)


@pytest.mark.parametrize("consumer", ["stream", "events", "iteration", "iter_text", "text"])
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_consumption_paths_finalize_once(memory_logger, tmp_path, consumer):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)

    async def on_delta(_update):
        return None

    async with await cursor_sdk.AsyncClient.launch_bridge(workspace=str(workspace)) as client:
        async with await client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            run = await agent.send(
                "Reply with exactly: cursor async tracing complete",
                cursor_sdk.SendOptions(on_delta=on_delta, on_step=lambda _step: None),
            )
            if consumer == "stream":
                consumed = [item async for item in run.stream()]
            elif consumer == "events":
                consumed = [item async for item in run.events()]
            elif consumer == "iteration":
                consumed = [item async for item in run]
            elif consumer == "iter_text":
                consumed = [item async for item in run.iter_text()]
            else:
                consumed = await run.text()
            result = await run.wait()

    assert consumed
    assert result.status == "finished"
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TASK)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_send_error_propagates_and_logs_parent_error(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            with pytest.raises(cursor_sdk.CursorAgentError) as raised:
                agent.send(
                    "This request must fail before execution.",
                    cursor_sdk.SendOptions(model="braintrust-invalid-cursor-model"),
                )

    assert raised.value is not None
    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert spans[0].get("error")


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_multimodal_input_materializes_attachment(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    message = cursor_sdk.UserMessage(
        text="Briefly describe this image.",
        images=[cursor_sdk.SDKImage.from_data(png, "image/png")],
    )
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            result = agent.send(message).wait()

    assert result.status == "finished"
    spans = memory_logger.pop()
    (task_span,) = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    image_url = task_span["input"][0]["content"][1]["image_url"]["url"]
    assert isinstance(image_url, Attachment)
    assert image_url.reference["type"] == "braintrust_attachment"
    assert image_url.reference["content_type"] == "image/png"
    assert image_url.reference["filename"] == "image.png"


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_custom_tool_dispatch_traces_under_the_agents_run(memory_logger, tmp_path):
    """Cursor calls custom tools by POSTing into the SDK's loopback callback server.

    That request carries only an agent id -- no run, no agent -- so it is the
    one lookup the dispatch hook cannot answer from its arguments. The spans it
    opens are asserted on directly because they are created on the callback
    server's handler thread, and ``memory_logger`` only overrides the
    background logger for the thread that installed it.
    """
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    executed = {}

    def lookup(args, context):
        executed["args"] = dict(args)
        executed["call_id"] = context.tool_call_id
        executed["span"] = logger.current_span()
        return "cursor-custom-tool-complete"

    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            # Register through the SDK's own registrar rather than via
            # `LocalAgentOptions(custom_tools=...)`: declaring tools up front
            # makes the client mint a random agent id, which never replays.
            client._owned_bridge.register_custom_tools(
                agent.agent_id, {"lookup": cursor_sdk.CustomTool(execute=lookup)}
            )
            run = agent.send("Reply with exactly: cursor tracing complete")
            response = _call_custom_tool(client, agent.agent_id, "lookup", "call-1", {"query": "braintrust"})
            result = run.wait()

    assert result.status == "finished"
    assert response["result"]["content"][0]["text"] == "cursor-custom-tool-complete"
    assert executed["args"] == {"query": "braintrust"}
    assert executed["call_id"] == "call-1"

    spans = memory_logger.pop()
    (task_span,) = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    tool_span = executed["span"]
    # A no-op span here means the dispatch never resolved the agent id back to
    # the run's tracker, or the tool span was not made current for the callback.
    assert tool_span is not logger.NOOP_SPAN
    assert tool_span.root_span_id == task_span["root_span_id"]
    # Nested under the run's model turn rather than opened as its own root.
    assert tool_span.span_parents
    assert task_span["span_id"] not in tool_span.span_parents


@_REQUIRES_SYNC_BRIDGE
@pytest.mark.vcr
def test_builtin_tool_events_create_one_tool_span(memory_logger, tmp_path):
    import cursor_sdk

    assert CursorSDKIntegration.setup()
    workspace = _workspace(tmp_path)
    with cursor_sdk.CursorClient.launch_bridge(workspace=str(workspace)) as client:
        with client.agents.create(
            model=TEST_MODEL,
            api_key=_api_key(),
            local=cursor_sdk.LocalAgentOptions(cwd=str(workspace)),
        ) as agent:
            result = agent.send(
                "Use the shell tool to run `printf cursor-tool-complete`, then reply with the output."
            ).wait()

    assert result.status == "finished"
    spans = memory_logger.pop()
    (task_span,) = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert llm_spans
    assert tool_spans
    assert len({span["span_id"] for span in tool_spans}) == len(tool_spans)
    llm_ids = {span["span_id"] for span in llm_spans}
    for tool_span in tool_spans:
        assert tool_span["input"] is not None
        assert tool_span.get("output") is not None
        assert any(parent in llm_ids for parent in tool_span["span_parents"])
        assert tool_span["root_span_id"] == task_span["root_span_id"]
        assert tool_span["context"]["span_origin"]["instrumentation"]["name"] == "cursor-sdk-auto"
