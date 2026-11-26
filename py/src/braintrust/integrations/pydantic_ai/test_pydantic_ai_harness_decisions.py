import pytest
from braintrust import logger, setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel


pytest.importorskip("pydantic_ai_harness")

PROJECT_NAME = "test-pydantic-ai-harness-decisions"


@pytest.fixture
def memory_logger():
    assert setup_pydantic_ai(project_name=PROJECT_NAME)
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _decision_span(spans, name):
    (agent_span,) = (span for span in spans if span["span_attributes"]["name"] == "agent_run [harness]")
    (decision,) = (span for span in spans if span["span_attributes"]["name"] == name)
    assert decision["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert decision["span_parents"] == [agent_span["span_id"]]
    assert decision["context"]["span_origin"]["instrumentation"]["name"] == "pydantic-ai-auto"
    return decision


@pytest.mark.asyncio
async def test_harness_compaction_decision(memory_logger):
    compaction = pytest.importorskip("pydantic_ai_harness.compaction")
    tool_results = []

    def model(messages, _info):
        results = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not results:
            return ModelResponse(parts=[ToolCallPart("note", {}, tool_call_id="note")])
        tool_results.extend(results)
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model),
        name="harness",
        capabilities=[compaction.ClearToolResults(max_messages=1, keep_pairs=0)],
    )

    @agent.tool_plain
    def note():
        return "old tool result"

    assert (await agent.run("call the tool")).output == "done"
    assert tool_results == ["[tool result cleared]"]

    decision = _decision_span(memory_logger.pop(), "compact_messages")
    assert decision["metadata"]["strategy"] == "ClearToolResults"
    assert decision["input"] == {"messages": 3}
    assert decision["output"] == {"messages": 3}


@pytest.mark.asyncio
async def test_harness_memory_injection_decision(memory_logger):
    memory = pytest.importorskip("pydantic_ai_harness.memory")
    store = memory.InMemoryMemoryStore()
    await store.write("main/MEMORY.md", "offline memory token", expected_version=None)
    agent = Agent(
        TestModel(call_tools=[], custom_output_text="done"),
        name="harness",
        capabilities=[memory.Memory(store=store)],
    )

    assert (await agent.run("remember")).output == "done"

    decision = _decision_span(memory_logger.pop(), "memory.inject")
    assert decision["output"]["outcome"] == "ok"
    assert decision["output"]["appended"] is True
    assert decision["output"]["injected_chars"] >= len("offline memory token")
    assert "offline memory token" not in str(decision)


@pytest.mark.asyncio
async def test_harness_guardrail_block_decision(memory_logger):
    guardrails = pytest.importorskip("pydantic_ai_harness.guardrails")

    async def deny(_ctx, _output):
        return guardrails.GuardrailResult.block("blocked by test policy")

    agent = Agent(
        TestModel(custom_output_text="rejected output"),
        name="harness",
        capabilities=[guardrails.OutputGuardrail(deny)],
    )

    with pytest.raises(guardrails.OutputBlocked, match="blocked by test policy"):
        await agent.run("check output")

    decision = _decision_span(memory_logger.pop(), "guardrail blocked output")
    assert decision["input"] == {"direction": "output"}
    assert decision["output"] == {"action": "block", "reason": "blocked by test policy"}
    assert not decision.get("error")


@pytest.mark.asyncio
@pytest.mark.parametrize("instrument", [False, True])
async def test_harness_model_router_decision(memory_logger, instrument):
    router = pytest.importorskip("pydantic_ai_harness.model_router")
    capabilities = pytest.importorskip("pydantic_ai.capabilities")
    agent = Agent(
        TestModel(),
        name="harness",
        capabilities=[
            *([capabilities.Instrumentation()] if instrument else []),
            router.ModelRouter(
                choices={
                    "default": router.ModelChoice(TestModel(custom_output_text="default"), "default"),
                    "alternate": router.ModelChoice(TestModel(custom_output_text="alternate"), "alternate"),
                },
                router_model=TestModel(custom_output_args={"choice": "alternate"}),
                default="default",
            ),
        ],
    )

    assert (await agent.run("route to alternate")).output == "alternate"

    spans = memory_logger.pop()
    decision = _decision_span(spans, "model_router.select")
    assert decision["input"] == {"choices": ["default", "alternate"], "default": "default"}
    assert decision["metadata"]["mode"] == "once"
    assert decision["output"] == {"choice": "alternate", "fallback_reason": "none"}
    (router_agent,) = (span for span in spans if span["span_attributes"]["name"] == "agent_run [model_router]")
    assert router_agent["span_parents"] == [decision["span_id"]]
