"""Real Antigravity agent coverage with recorded harness transport."""

import asyncio
import gc
import statistics

import pytest
from braintrust import logger
from braintrust.integrations.antigravity import AntigravityIntegration, setup_antigravity, wrap_antigravity_agent
from braintrust.integrations.antigravity._test_transport import cassette_transport
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.logger import Attachment, current_span, start_span
from braintrust.test_helpers import init_test_logger
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.tools.tool_runner import ToolWithSchema


PROMPT = "Reply with exactly: antigravity tracing complete"


@pytest.fixture
def memory_logger():
    init_test_logger("test-antigravity")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def agent_config(tmp_path):
    return LocalAgentConfig(model="gemini-3.8-flash", app_data_dir=str(tmp_path)).lightweight()


def test_setup_and_wrap_are_idempotent():
    assert AntigravityIntegration.setup()
    chat = Agent.__dict__["chat"]
    exit_method = Agent.__dict__["__aexit__"]
    for _ in range(2):
        assert wrap_antigravity_agent(Agent) is Agent
        assert AntigravityIntegration.setup()
        assert Agent.__dict__["chat"] is chat
        assert Agent.__dict__["__aexit__"] is exit_method


@pytest.mark.asyncio
@pytest.mark.parametrize("accessor", ["text", "stream", "chunks", "concurrent", "manual"])
async def test_chat(memory_logger, tmp_path, accessor):
    if accessor == "manual":
        assert wrap_antigravity_agent(Agent) is Agent
    else:
        assert setup_antigravity(project="test-antigravity")
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            with start_span(name="parent") as parent:
                response = await agent.chat(PROMPT)
                assert isinstance(response, types.ChatResponse)
                assert current_span() is parent
                if accessor == "stream":
                    text = "".join([token async for token in response])
                elif accessor == "chunks":
                    await response.resolve()
                    text = await response.text()
                elif accessor == "concurrent":
                    text, again = await asyncio.gather(response.text(), response.text())
                    assert text == again
                else:
                    text = await response.text()
                assert "antigravity tracing complete" in text.lower()
                assert await response.text() == text
                assert current_span() is parent
                assert response.usage_metadata.total_token_count > 0
    spans = memory_logger.pop()
    chats = [s for s in spans if s.get("span_attributes", {}).get("name") == "antigravity.chat"]
    assert len(chats) == 1
    chat = chats[0]
    assert chat["span_attributes"]["type"] == "task"
    assert chat["span_parents"] == [parent.span_id]
    assert chat["input"][0]["content"][0]["text"] == PROMPT
    assert chat["output"][0]["content"] == text
    assert chat["metadata"]["model"] == "gemini-3.8-flash"
    assert chat["metadata"]["tools"]
    assert chat["metrics"]["tokens"] == response.usage_metadata.total_token_count
    assert chat["metrics"]["completion_tokens"] == (response.usage_metadata.candidates_token_count or 0) + (
        response.usage_metadata.thoughts_token_count or 0
    )
    assert chat["metrics"]["time_to_first_token"] >= 0
    assert chat["metrics"]["end"] >= chat["metrics"]["start"]
    children = [s for s in spans if s["span_parents"] == [chat["span_id"]]]
    assert children
    for span in [chat, *children]:
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "antigravity-auto"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", ["drop", "partial", "close", "disconnect"])
async def test_stream_cleanup(memory_logger, tmp_path, exit_path):
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            response = await agent.chat(PROMPT)
            before = current_span()
            if exit_path == "partial":
                async for _ in response:
                    break
            elif exit_path == "close":
                await response._chunk_stream.aclose()
            if exit_path != "disconnect":
                del response
                # The SDK's text cursor and nested chunk cursor are async
                # generators; let Python finish their queued finalizers.
                for _ in range(5):
                    gc.collect()
                    await asyncio.sleep(0)
                spans = memory_logger.pop()
                assert any(s["span_attributes"]["name"] == "antigravity.chat" and "end" in s["metrics"] for s in spans)
            assert current_span() is before
    gc.collect()
    remaining = memory_logger.pop()
    if exit_path != "disconnect":
        assert remaining == []
    else:
        spans = remaining
    chats = [s for s in spans if s.get("span_attributes", {}).get("name") == "antigravity.chat"]
    assert len(chats) == 1
    assert "end" in chats[0]["metrics"]


@pytest.mark.asyncio
async def test_send_receive_is_not_instrumented(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            await agent.conversation.send(PROMPT)
            async for _ in agent.conversation.receive_steps():
                pass
    assert memory_logger.pop() == []


@pytest.mark.parametrize("import_order", ["before", "after"])
def test_auto_instrument(monkeypatch, import_order):
    monkeypatch.setenv("BRAINTRUST_ANTIGRAVITY_IMPORT_ORDER", import_order)
    verify_autoinstrument_script("test_auto_antigravity.py", timeout=120)


@pytest.mark.asyncio
async def test_builtin_tool(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    (tmp_path / "README.md").write_text("The secret word is pineapple.", encoding="utf-8")
    config = agent_config(tmp_path)
    config.workspaces = [str(tmp_path)]
    config.system_instructions = "Always use view_file when asked to read a file."
    with cassette_transport("tool"):
        async with Agent(config) as agent:
            response = await agent.chat("Use view_file to read README.md and tell me the secret word.")
            assert "pineapple" in (await response.text()).lower()
    spans = memory_logger.pop()
    chat = next(s for s in spans if s["span_attributes"]["name"] == "antigravity.chat")
    assert chat["input"][0] == {"role": "system", "content": config.system_instructions}
    tools = [s for s in spans if s["span_attributes"]["type"] == "tool"]
    view_file = next(s for s in tools if s["span_attributes"]["name"] == "view_file")
    assert view_file["span_parents"] == [chat["span_id"]]
    assert view_file["input"]
    assert "end" in view_file["metrics"]


@pytest.mark.asyncio
async def test_cancel_active_turn(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    with cassette_transport("cancel"):
        async with Agent(agent_config(tmp_path)) as agent:
            response = await agent.chat(PROMPT)
            await response.cancel()
            with pytest.raises(types.AntigravityCancelledError) as raised:
                await response.text()
    spans = memory_logger.pop()
    chat = next(s for s in spans if s["span_attributes"]["name"] == "antigravity.chat")
    assert "end" in chat["metrics"]
    assert str(raised.value) in chat["error"]


@pytest.mark.asyncio
async def test_multiple_turns(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    counts = []
    responses = []
    with cassette_transport("multiple_turns"):
        async with Agent(agent_config(tmp_path)) as agent:
            for _ in range(2):
                response = await agent.chat(PROMPT)
                await response.text()
                responses.append(response)
                counts.append(response.usage_metadata.total_token_count)
            assert await responses[0].text() == await responses[-1].text()
            total = agent.conversation.total_usage.total_token_count
    chats = [s for s in memory_logger.pop() if s["span_attributes"]["name"] == "antigravity.chat"]
    assert len(chats) == 2
    assert [s["metrics"]["tokens"] for s in chats] == counts
    assert sum(counts) == total


@pytest.mark.asyncio
async def test_python_tool(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    # A standard-library callable, executed by the real ToolRunner.
    mean = ToolWithSchema(
        statistics.mean,
        {
            "type": "object",
            "properties": {"data": {"type": "array", "items": {"type": "number"}}},
            "required": ["data"],
        },
    )
    config = agent_config(tmp_path)
    config.tools = [mean]
    with cassette_transport("python_tool"):
        async with Agent(config) as agent:
            response = await agent.chat("Use the mean tool to calculate the average of 12, 18, and 24.")
            assert "18" in await response.text()
    spans = memory_logger.pop()
    chat = next(s for s in spans if s["span_attributes"]["name"] == "antigravity.chat")
    tools = [s for s in spans if s["span_attributes"]["name"] == "mean"]
    assert len(tools) == 1
    assert tools[0]["span_attributes"]["type"] == "tool"
    assert tools[0]["span_parents"] == [chat["span_id"]]
    assert tools[0]["input"] == {"data": [12, 18, 24]}
    assert tools[0]["output"] == 18
    definitions = chat["metadata"]["tools"]
    assert any(t.get("function", {}).get("name") == "mean" for t in definitions)
    assert tools[0]["context"]["span_origin"]["instrumentation"]["name"] == "antigravity-auto"


@pytest.mark.asyncio
async def test_provider_error(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            response = await agent.chat(PROMPT)
            # Inject a transport failure into the real SDK queue. Successful
            # recordings cannot provoke an unexpected connection loss.
            error = types.AntigravityConnectionError("Connection lost")
            await agent.conversation.connection._processor.step_queue.put(error)
            for _ in range(2):
                with pytest.raises(types.AntigravityConnectionError) as raised:
                    await response.text()
                assert raised.value is error
    chat = next(s for s in memory_logger.pop() if s["span_attributes"]["name"] == "antigravity.chat")
    assert "Connection lost" in chat["error"]
    assert "end" in chat["metrics"]


@pytest.mark.asyncio
async def test_failed_send_does_not_reuse_previous_usage(memory_logger, tmp_path, monkeypatch):
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            response = await agent.chat(PROMPT)
            await response.text()
            error = types.AntigravityConnectionError("Send failed")

            async def fail_send(*args, **kwargs):
                raise error

            # Failure before Conversation.send establishes a new turn must
            # not attribute the previous successful turn's tokens to this one.
            monkeypatch.setattr(agent.conversation, "send", fail_send)
            with pytest.raises(types.AntigravityConnectionError) as raised:
                await agent.chat(PROMPT)
            assert raised.value is error
    chats = [s for s in memory_logger.pop() if s["span_attributes"]["name"] == "antigravity.chat"]
    assert len(chats) == 2
    failed = next(s for s in chats if "error" in s)
    assert "tokens" not in failed["metrics"]
    assert "prompt_tokens" not in failed["metrics"]
    assert "completion_tokens" not in failed["metrics"]


@pytest.mark.asyncio
async def test_metadata_failure_does_not_leave_pending_span(memory_logger, tmp_path, monkeypatch):
    assert AntigravityIntegration.setup()
    with cassette_transport("chat"):
        async with Agent(agent_config(tmp_path)) as agent:
            runner = agent.conversation.connection._tool_runner

            def fail_tool_names(self):
                raise RuntimeError("Cannot inspect tools")

            # A telemetry-only inspection failure must preserve the real
            # response without leaving a partially initialized chat span.
            monkeypatch.setattr(type(runner), "tool_names", property(fail_tool_names))
            response = await agent.chat(PROMPT)
            assert "antigravity tracing complete" in (await response.text()).lower()
    assert memory_logger.pop() == []


@pytest.mark.asyncio
async def test_document_input(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    document = types.Document(data=b"The secret word is pineapple.", mime_type="text/plain")
    with cassette_transport("document"):
        async with Agent(agent_config(tmp_path)) as agent:
            response = await agent.chat(["What is the secret word in this document?", document])
            assert "pineapple" in (await response.text()).lower()
    chat = next(s for s in memory_logger.pop() if s["span_attributes"]["name"] == "antigravity.chat")
    attachment = chat["input"][0]["content"][1]["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "text/plain"
    assert document.data == b"The secret word is pineapple."


@pytest.mark.asyncio
async def test_structured_output(memory_logger, tmp_path):
    assert AntigravityIntegration.setup()
    config = agent_config(tmp_path)
    config.response_schema = {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }
    with cassette_transport("structured"):
        async with Agent(config) as agent:
            response = await agent.chat("Return the word pineapple using the required output schema.")
            result = await response.structured_output()
    assert result == {"word": "pineapple"}
    chat = next(s for s in memory_logger.pop() if s["span_attributes"]["name"] == "antigravity.chat")
    assert chat["output"] == result
