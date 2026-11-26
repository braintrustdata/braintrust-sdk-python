from importlib import import_module

import pytest
from braintrust.integrations.pydantic_ai._test_harness import OFFLINE_ANSWER, call_tool_once, make_harness_agent
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type


@pytest.mark.asyncio
async def test_harness_run_traces_model_and_tool(harness_spans):
    persistence = import_module("pydantic_ai_harness.step_persistence")
    store = persistence.InMemoryStepStore()
    agent = make_harness_agent(
        "harness_smoke",
        call_tool_once("echo", {"text": "hello"}),
        capabilities=[persistence.StepPersistence(store=store)],
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        return text

    result = await agent.run("Echo hello.", run_id="harness-smoke")

    assert result.output == OFFLINE_ANSWER
    assert await store.list_events(run_id="harness-smoke")

    spans = harness_spans.pop()
    run = find_span_by_name(spans, "agent_run [harness_smoke]")
    assert run["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert not run.get("span_parents")

    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(llm_spans) == 2
    for llm in llm_spans:
        assert llm["span_parents"] == [run["span_id"]]

    tool = find_span_by_name(spans, "echo")
    assert tool["span_attributes"]["type"] == SpanTypeAttribute.TOOL
    assert tool["span_parents"] == [run["span_id"]]
    assert tool["input"] == {"text": "hello"}
    assert tool["output"] == "hello"
