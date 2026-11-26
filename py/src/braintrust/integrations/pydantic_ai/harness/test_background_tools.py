import asyncio

import pytest
from braintrust import setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from braintrust.test_helpers import with_memory_logger as with_memory_logger
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
async def test_harness_background_tool_execution(with_memory_logger, fails):
    background_tools = pytest.importorskip("pydantic_ai_harness.background_tools", exc_type=ModuleNotFoundError)
    from pydantic_ai.exceptions import ToolFailed

    project = "test-pydantic-ai-harness"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)

    call_id = "background-call"
    receipt_seen = asyncio.Event()
    child = Agent(TestModel(custom_output_text="offline answer"), name="background_child")

    def respond(messages, _info):
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id == call_id:
                    assert "is running in background" in part.content
                    receipt_seen.set()
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("slow", {"query": "lookup"}, tool_call_id=call_id)])

    parent = Agent(FunctionModel(respond), name="background_parent", capabilities=[background_tools.BackgroundTools()])

    @parent.tool_plain(metadata={"background": True})
    async def slow(query: str) -> str:
        await receipt_seen.wait()
        if fails:
            raise ToolFailed(f"failed: {query}")
        return (await child.run(query)).output

    assert (await parent.run("dispatch background tool")).output == "done"
    spans = with_memory_logger.pop()
    parent_span = next(s for s in spans if s["span_attributes"]["name"] == "agent_run [background_parent]")
    tool_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TOOL]
    assert [span["span_attributes"]["name"] for span in tool_spans] == ["slow", "slow"]
    dispatch = next(s for s in tool_spans if s["span_parents"] == [parent_span["span_id"]])
    execution = next(s for s in tool_spans if s["span_parents"] == [dispatch["span_id"]])
    assert "is running in background" in dispatch["output"]
    assert dispatch.get("error") is None
    assert execution["input"] == {"query": "lookup"}
    assert execution["metadata"]["tool_call_id"] == dispatch["metadata"]["tool_call_id"] == call_id
    assert execution["context"]["span_origin"]["instrumentation"]["name"] == "pydantic-ai-auto"
    if fails:
        assert "failed: lookup" in execution["error"]
        assert execution.get("output") is None
    else:
        assert execution["output"] == "offline answer"
        assert execution.get("error") is None
        child_span = next(s for s in spans if s["span_attributes"]["name"] == "agent_run [background_child]")
        assert child_span["span_parents"] == [execution["span_id"]]
