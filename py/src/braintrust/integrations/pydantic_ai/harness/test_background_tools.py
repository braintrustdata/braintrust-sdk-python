import asyncio
from importlib import import_module

import pydantic_ai
import pytest
from braintrust import setup_pydantic_ai
from braintrust.integrations.pydantic_ai.patchers import BackgroundToolExecutionPatcher
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger
from braintrust.test_helpers import with_memory_logger as with_memory_logger
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.usage import UsageLimits


def test_background_tool_patcher_version_gate():
    assert BackgroundToolExecutionPatcher.applies(pydantic_ai, "2.75.0")
    assert not BackgroundToolExecutionPatcher.applies(pydantic_ai, "1.107.9")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fails", "use_iter"),
    [
        pytest.param(False, False, id="success", marks=pytest.mark.vcr),
        pytest.param(True, False, id="failure", marks=pytest.mark.vcr),
        pytest.param(
            False,
            True,
            id="iter",
            marks=pytest.mark.vcr(cassette_name="test_harness_background_tool_execution[success]"),
        ),
    ],
)
async def test_harness_background_tool_execution(with_memory_logger, fails, use_iter):
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

    prompt = "Call slow exactly once with query 'lookup'. Then answer 'done' without calling any more tools."
    limits = UsageLimits(request_limit=6)
    if use_iter:
        async with parent.iter(prompt, usage_limits=limits) as run:
            async for _ in run:
                pass
    else:
        await parent.run(prompt, usage_limits=limits)
    spans = with_memory_logger.pop()
    parent_span = find_span_by_name(spans, "agent_run [background_parent]")
    assert not parent_span.get("span_parents")
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert [span["span_attributes"]["name"] for span in tool_spans] == ["slow"]
    (execution,) = tool_spans
    assert execution["span_parents"] == [parent_span["span_id"]]
    assert execution["input"] == {"query": "lookup"}
    assert execution["metadata"]["tool_call_id"]
    assert execution["context"]["span_origin"]["instrumentation"]["name"] == "pydantic-ai-auto"
    if fails:
        assert "failed: lookup" in execution["error"]
        assert execution.get("output") is None
    else:
        assert execution["output"] == "lookup complete"
        assert execution.get("error") is None
        child_span = find_span_by_name(spans, "agent_run [background_child]")
        assert child_span["span_parents"] == [execution["span_id"]]
