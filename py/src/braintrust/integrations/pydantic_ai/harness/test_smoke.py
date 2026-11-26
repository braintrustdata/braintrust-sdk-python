from importlib import import_module

import pytest
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type
from pydantic_ai import Agent, ModelSettings


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_harness_run_traces_model_and_tool(harness_spans):
    persistence = import_module("pydantic_ai_harness.step_persistence")
    store = persistence.InMemoryStepStore()
    agent = Agent(
        "openai:gpt-4.1-mini",
        name="harness_smoke",
        model_settings=ModelSettings(temperature=0, max_tokens=64),
        capabilities=[persistence.StepPersistence(store=store)],
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        return text

    result = await agent.run(
        "Call the echo tool with text 'hello', then reply with the tool's exact output.",
        run_id="harness-smoke",
    )

    assert result.output == "hello"
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
