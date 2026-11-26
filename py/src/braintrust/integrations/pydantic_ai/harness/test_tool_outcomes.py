from importlib import import_module

import pytest
from braintrust import setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger
from braintrust.test_helpers import with_memory_logger as with_memory_logger
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ModelSettings
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel


MODEL = "gpt-6-luna"
MODEL_SETTINGS = ModelSettings(openai_reasoning_effort="none", max_tokens=200)


@pytest.fixture
def spans(with_memory_logger):
    project = "test-pydantic-ai-harness-tool-outcomes"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)
    return with_memory_logger


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mixed", "approval-only", "stream"])
async def test_declarative_tool_approval(spans, mode):
    lookup_calls = ["lookup"] if mode == "mixed" else []
    agent = Agent(
        f"openai:{MODEL}",
        name="approval",
        output_type=[str, DeferredToolRequests],
        model_settings=MODEL_SETTINGS,
    )
    executed = []

    @agent.tool_plain(requires_approval=True)
    def gated() -> str:
        executed.append("gated")
        return "approved"

    def lookup() -> str:
        executed.append("lookup")
        return "found"

    prompt = "Call the gated tool now. Do not answer with text."
    if lookup_calls:
        agent.tool_plain(lookup)
        prompt = "Call the lookup and gated tools in parallel in your first response. Both calls are required."

    if mode == "stream":
        async with agent.run_stream(prompt) as first:
            pending = await first.get_output()
    else:
        first = await agent.run(prompt)
        pending = first.output
    assert executed == lookup_calls
    assert isinstance(pending, DeferredToolRequests)
    (approval,) = pending.approvals
    assert approval.tool_name == "gated"
    if lookup_calls:
        assert any(
            isinstance(message, ModelResponse)
            and {call.tool_name for call in message.tool_calls} == {"lookup", "gated"}
            for message in first.new_messages()
        )
    tools = find_spans_by_type(spans.pop(), SpanTypeAttribute.TOOL)
    assert len(tools) == 1 + len(lookup_calls)
    gated_span = find_span_by_name(tools, "gated")
    assert gated_span.get("error") is None
    assert gated_span.get("output") is None
    assert gated_span["metrics"]["duration"] >= 0
    assert gated_span["metadata"] == {
        "tool_call_id": approval.tool_call_id,
        "tool_outcome": "deferred",
        "deferred": "unapproved",
    }
    if lookup_calls:
        lookup_span = find_span_by_name(tools, "lookup")
        assert lookup_span["metadata"]["tool_outcome"] == "executed"
        assert lookup_span["output"] == "found"

    await agent.run(
        message_history=first.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={approval.tool_call_id: True}),
        model_settings=ModelSettings(tool_choice="none"),
    )
    assert executed == [*lookup_calls, "gated"]
    (gated_span,) = find_spans_by_type(spans.pop(), SpanTypeAttribute.TOOL)
    assert gated_span["span_attributes"]["name"] == "gated"
    assert gated_span["metadata"]["tool_outcome"] == "executed"
    assert gated_span["metadata"]["tool_call_id"] == approval.tool_call_id
    assert gated_span["output"] == "approved"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_model_router_output_is_not_a_tool_execution(spans):
    router = import_module("pydantic_ai_harness.model_router")
    model = OpenAIResponsesModel(MODEL, settings=MODEL_SETTINGS)
    agent = Agent(
        model,
        name="with_router",
        capabilities=[
            router.ModelRouter(
                choices={
                    "default": router.ModelChoice(model, "For requests in Spanish or French."),
                    "alternate": router.ModelChoice(model, "For requests in English."),
                },
                router_model=model,
                default="default",
            )
        ],
    )

    result = await agent.run("Reply in English with exactly the word ROUTED.")

    assert result.output.strip() == "ROUTED"
    rows = spans.pop()
    router_span = find_span_by_name(rows, "agent_run [model_router]")
    assert router_span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert router_span["output"]["output"]["choice"] == "alternate"
    assert not find_spans_by_type(rows, SpanTypeAttribute.TOOL)
