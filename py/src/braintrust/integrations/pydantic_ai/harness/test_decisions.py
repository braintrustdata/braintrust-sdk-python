from importlib import import_module

import pytest
from braintrust import setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, infer_model
from pydantic_ai.usage import RunUsage


MODEL = "openai-chat:gpt-4.1-mini"
DEFAULT_ROUTE_MODEL = "openai-chat:gpt-4.1"
SETTINGS = ModelSettings(temperature=0, max_tokens=32)


@pytest.fixture
def decision_spans(with_memory_logger):
    project = "test-pydantic-ai-harness-decisions"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)
    return with_memory_logger


def _decision_span(spans, name):
    (agent_span,) = (span for span in spans if span["span_attributes"]["name"] == "agent_run [harness]")
    (decision,) = (span for span in spans if span["span_attributes"]["name"] == name)
    assert decision["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert decision["span_parents"] == [agent_span["span_id"]]
    assert decision["context"]["span_origin"]["instrumentation"]["name"] == "pydantic-ai-auto"
    return decision


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_harness_compaction_decision(decision_spans):
    compaction = import_module("pydantic_ai_harness.compaction")

    agent = Agent(
        MODEL,
        name="harness",
        model_settings=SETTINGS,
        capabilities=[compaction.ClearToolResults(max_messages=1, keep_pairs=0)],
    )

    @agent.tool_plain
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    result = await agent.run("Call add with x=1 and y=2 exactly once. After the tool returns, reply only 'done'.")
    assert "done" in result.output.lower()

    spans = decision_spans.pop()
    decision = _decision_span(spans, "harness_message_compaction")
    assert decision["metadata"]["strategy"] == "ClearToolResults"
    assert decision["input"] == {"messages": 3}
    assert decision["output"] == {"messages": 3}
    assert any(
        "[tool result cleared]" in str(span.get("input"))
        for span in spans
        if span["span_attributes"]["type"] == SpanTypeAttribute.LLM
    )


@pytest.mark.asyncio
async def test_harness_background_compaction_retrieval_is_not_traced(decision_spans):
    compaction = import_module("pydantic_ai_harness.compaction")
    model = infer_model(MODEL)
    messages = [ModelRequest(parts=[UserPromptPart(prompt)]) for prompt in ("older context " * 10, "current request")]
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages, conversation_id="background")
    request = ModelRequestContext(
        model=model, messages=messages.copy(), model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    fallback = compaction.FallbackCompaction(
        [compaction.SlidingWindowCompaction(keep_messages=1, preserve_first_user_message=False)],
        max_tokens=100,
        background_start=0.1,
    )

    assert await fallback.before_model_request(ctx, request) is request
    compacted = await fallback.take_background_compaction(messages, model=model, conversation_id="background")

    assert compacted == messages[-1:]
    assert not decision_spans.pop()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_harness_memory_injection_decision(decision_spans):
    memory = import_module("pydantic_ai_harness.memory")
    store = memory.InMemoryMemoryStore()
    stored = "The saved label is orchid."
    await store.write("main/MEMORY.md", stored, expected_version=None)
    agent = Agent(
        MODEL,
        name="harness",
        model_settings=SETTINGS,
        capabilities=[memory.Memory(store=store)],
    )

    result = await agent.run("Reply with only the saved label from your injected memory. Do not call tools.")
    assert "orchid" in result.output.lower()

    decision = _decision_span(decision_spans.pop(), "harness_memory_injection")
    assert decision["output"]["outcome"] == "ok"
    assert decision["output"]["appended"] is True
    assert decision["output"]["injected_chars"] >= len(stored)
    assert stored not in str(decision)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_harness_guardrail_block_decision(decision_spans):
    guardrails = import_module("pydantic_ai_harness.guardrails")

    async def check_output(_ctx, output):
        if "restricted" in output.lower():
            return guardrails.GuardrailResult.block(f"blocked by test policy: {output}")
        return guardrails.GuardrailResult.allow()

    agent = Agent(
        MODEL,
        name="harness",
        model_settings=SETTINGS,
        capabilities=[guardrails.OutputGuardrail(check_output)],
    )

    with pytest.raises(guardrails.OutputBlocked, match="blocked by test policy: restricted"):
        await agent.run("Reply with only the word 'restricted'.")

    decision = _decision_span(decision_spans.pop(), "harness_guardrail_block")
    assert decision["metadata"] == {"direction": "output"}
    assert decision["output"] == {"action": "block"}
    assert "blocked by test policy" not in str(decision)
    assert not decision.get("error")


@pytest.mark.asyncio
@pytest.mark.vcr(cassette_name="test_harness_model_router_decision")
@pytest.mark.parametrize("instrument", [False, True])
async def test_harness_model_router_decision(decision_spans, instrument):
    router = import_module("pydantic_ai_harness.model_router")
    capabilities = import_module("pydantic_ai.capabilities")
    agent = Agent(
        MODEL,
        name="harness",
        model_settings=SETTINGS,
        capabilities=[
            *([capabilities.Instrumentation()] if instrument else []),
            router.ModelRouter(
                choices={
                    "default": router.ModelChoice(DEFAULT_ROUTE_MODEL, "Use for all topics except arithmetic."),
                    "alternate": router.ModelChoice(MODEL, "Use for arithmetic questions."),
                },
                router_model=MODEL,
                default="default",
            ),
        ],
    )

    result = await agent.run("What is 1 + 2? Reply with just the number.")
    assert "3" in result.output

    spans = decision_spans.pop()
    decision = _decision_span(spans, "harness_model_selection")
    assert decision["input"] == {"choices": ["default", "alternate"], "default": "default"}
    assert decision["metadata"]["mode"] == "once"
    assert set(decision["metadata"]) <= {"mode", "run_step", "probability", "error.type"}
    assert decision["output"] == {"choice": "alternate", "fallback_reason": "none"}
    (router_agent,) = (span for span in spans if span["span_attributes"]["name"] == "agent_run [model_router]")
    assert router_agent["span_parents"] == [decision["span_id"]]
