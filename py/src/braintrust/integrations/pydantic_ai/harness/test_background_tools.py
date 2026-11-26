import asyncio
from importlib import import_module

import pytest
from braintrust.integrations.pydantic_ai._test_harness import OFFLINE_ANSWER, make_harness_agent
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
async def test_harness_background_tool_execution(harness_spans, fails):
    from pydantic_ai.exceptions import ToolFailed

    background_tools = import_module("pydantic_ai_harness.background_tools")
    call_id = "background-call"
    receipt_seen = asyncio.Event()
    child = make_harness_agent("background_child")

    def respond(messages, _info):
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id == call_id:
                    assert "is running in background" in part.content
                    receipt_seen.set()
                    return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("slow", {"query": "lookup"}, tool_call_id=call_id)])

    parent = make_harness_agent(
        "background_parent", FunctionModel(respond), capabilities=[background_tools.BackgroundTools()]
    )

    @parent.tool_plain(metadata={"background": True})
    async def slow(query: str) -> str:
        await receipt_seen.wait()
        if fails:
            raise ToolFailed(f"failed: {query}")
        return (await child.run(query)).output

    assert (await parent.run("dispatch background tool")).output == "done"
    spans = harness_spans.pop()
    parent_span = find_span_by_name(spans, "agent_run [background_parent]")
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
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
        assert execution["output"] == OFFLINE_ANSWER
        assert execution.get("error") is None
        child_span = find_span_by_name(spans, "agent_run [background_child]")
        assert child_span["span_parents"] == [execution["span_id"]]
