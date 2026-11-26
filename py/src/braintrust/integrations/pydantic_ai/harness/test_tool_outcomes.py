from importlib import import_module

import pytest
from braintrust.integrations.pydantic_ai._test_harness import make_harness_agent
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel as PydanticTestModel


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["mixed", "approval-only", "stream"])
async def test_declarative_tool_approval(harness_spans, mode):
    lookup_calls = ["lookup"] if mode == "mixed" else []
    agent = make_harness_agent(
        "approval",
        PydanticTestModel(call_tools=["gated", *lookup_calls]),
        output_type=[str, DeferredToolRequests],
    )
    executed = []

    @agent.tool_plain(requires_approval=True)
    def gated() -> str:
        executed.append("gated")
        return "approved"

    @agent.tool_plain
    def lookup() -> str:
        executed.append("lookup")
        return "found"

    if mode == "stream":
        async with agent.run_stream("Request approval") as first:
            pending = await first.get_output()
    else:
        first = await agent.run("Request tool calls")
        pending = first.output
    assert executed == lookup_calls
    assert isinstance(pending, DeferredToolRequests)
    (approval,) = pending.approvals
    assert approval.tool_name == "gated"
    tools = find_spans_by_type(harness_spans.pop(), SpanTypeAttribute.TOOL)
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
    )
    assert executed == [*lookup_calls, "gated"]
    (gated_span,) = find_spans_by_type(harness_spans.pop(), SpanTypeAttribute.TOOL)
    assert gated_span["span_attributes"]["name"] == "gated"
    assert gated_span["metadata"]["tool_outcome"] == "executed"
    assert gated_span["metadata"]["tool_call_id"] == approval.tool_call_id
    assert gated_span["output"] == "approved"


@pytest.mark.asyncio
async def test_model_router_output_is_not_a_tool_execution(harness_spans):
    router = import_module("pydantic_ai_harness.model_router")
    agent = make_harness_agent(
        "with_router",
        capabilities=[
            router.ModelRouter(
                choices={
                    "default": router.ModelChoice(PydanticTestModel(custom_output_text="default response"), "default"),
                    "alternate": router.ModelChoice(
                        PydanticTestModel(custom_output_text="alternate response"), "alternate"
                    ),
                },
                router_model=PydanticTestModel(custom_output_args={"choice": "alternate"}),
                default="default",
            )
        ],
    )

    result = await agent.run("Route to alternate")

    assert result.output == "alternate response"
    spans = harness_spans.pop()
    router_span = find_span_by_name(spans, "agent_run [model_router]")
    assert router_span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert not find_spans_by_type(spans, SpanTypeAttribute.TOOL)
