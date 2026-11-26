import asyncio
from importlib import import_module

import pytest
from braintrust import setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger
from braintrust.test_helpers import with_memory_logger as with_memory_logger
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.usage import UsageLimits


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
async def test_harness_background_tool_execution(with_memory_logger, fails):
    background_tools = import_module("pydantic_ai_harness.background_tools")
    project = "test-pydantic-ai-harness"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)

    child = Agent(OpenAIChatModel("gpt-4.1-mini"), name="background_child")
    parent = Agent(
        OpenAIResponsesModel("gpt-4.1-mini"),
        name="background_parent",
        capabilities=[background_tools.BackgroundTools()],
        model_settings=ModelSettings(parallel_tool_calls=False, max_tokens=100),
    )

    @parent.tool_plain(metadata={"background": True})
    async def slow(query: str) -> str:
        """Perform the requested lookup in the background."""
        await asyncio.sleep(0)
        if fails:
            raise ToolFailed(f"failed: {query}")
        return (await child.run(f"Reply with exactly '{query} complete' and nothing else.")).output

    await parent.run(
        "Call slow exactly once with query 'lookup'. Then answer 'done' without calling any more tools.",
        usage_limits=UsageLimits(request_limit=6),
    )
    spans = with_memory_logger.pop()
    parent_span = find_span_by_name(spans, "agent_run [background_parent]")
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert [span["span_attributes"]["name"] for span in tool_spans] == ["slow", "slow"]
    dispatch = next(s for s in tool_spans if s["span_parents"] == [parent_span["span_id"]])
    execution = next(s for s in tool_spans if s["span_parents"] == [dispatch["span_id"]])
    assert "is running in background" in dispatch["output"]
    assert dispatch.get("error") is None
    assert execution["input"] == {"query": "lookup"}
    assert execution["metadata"]["tool_call_id"] == dispatch["metadata"]["tool_call_id"]
    assert execution["context"]["span_origin"]["instrumentation"]["name"] == "pydantic-ai-auto"
    if fails:
        assert "failed: lookup" in execution["error"]
        assert execution.get("output") is None
    else:
        assert execution["output"] == "lookup complete"
        assert execution.get("error") is None
        child_span = find_span_by_name(spans, "agent_run [background_child]")
        assert child_span["span_parents"] == [execution["span_id"]]
        assert dispatch["metrics"]["end"] <= child_span["metrics"]["start"]
