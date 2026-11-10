# pylint: disable=import-error

from pathlib import Path

import pytest
from braintrust import Attachment, logger
from braintrust.integrations.strands import StrandsIntegration
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type, init_test_logger


TEST_MODEL = "gpt-6-luna"
MODEL_OPTIONS = {"reasoning_effort": "none"}
DECISION_MODEL = "gpt-6.1-terra"

PROJECT_NAME = "test-project-strands-py-tracing"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl
    logger._state.reset_parent_state()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_strands_openai_agent_traces_native_otel_lifecycle(memory_logger):
    from strands import Agent
    from strands.models.openai import OpenAIModel

    assert StrandsIntegration.setup()
    # A repeated setup() must not double-patch: the span counts below would duplicate.
    assert StrandsIntegration.setup()

    model = OpenAIModel(
        model_id=TEST_MODEL,
        params={
            "temperature": 0,
            **MODEL_OPTIONS,
            "max_completion_tokens": 16,
        },
    )
    agent = Agent(model=model, name="bt-test-agent", system_prompt="Answer with one short sentence.")

    result = await agent.invoke_async("What is 2 + 2?")

    assert result.message["role"] == "assistant"
    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    names = [span["span_attributes"]["name"] for span in spans]
    agent_spans = [span for span in task_spans if span["span_attributes"]["name"] == "bt-test-agent.invoke"]
    event_loop_spans = [span for span in task_spans if span["span_attributes"]["name"] == "event_loop.cycle"]
    assert len(agent_spans) == 1, names
    assert len(event_loop_spans) == 1, names
    agent_span = agent_spans[0]
    event_loop_span = event_loop_spans[0]
    assert agent_span["output"]["message"]["role"] == "assistant"
    assert "end" in agent_span["metrics"]
    assert event_loop_span["output"]["message"]["role"] == "assistant"
    assert "end" in event_loop_span["metrics"]
    assert len(llm_spans) == 1
    llm_span = llm_spans[0]
    assert llm_span["input"]["messages"][0]["role"] == "user"
    assert llm_span["span_attributes"]["name"] == f"{TEST_MODEL}.chat"
    assert llm_span["metadata"]["model"] == TEST_MODEL
    assert llm_span["metadata"]["stop_reason"] == "end_turn"
    assert llm_span["metadata"]["strands_usage"]["input_tokens"] > 0
    assert llm_span["metadata"]["strands_usage"]["output_tokens"] > 0
    assert "prompt_tokens" not in llm_span.get("metrics", {})
    assert "completion_tokens" not in llm_span.get("metrics", {})
    assert "tokens" not in llm_span.get("metrics", {})
    for span in spans:
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "strands-auto"


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_strands_decision_models_in_tool_trace_decision_spans(memory_logger):
    decisions = pytest.importorskip("strands.decisions", reason="DecisionModel requires strands-agents >= 1.63.0")
    if not hasattr(decisions, "TypeSafeDecisionModel"):
        pytest.skip("TypeSafeDecisionModel requires strands-agents >= 1.64.0")
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel

    assert StrandsIntegration.setup()

    questions = {
        "department": {"question": "Which team should handle this?", "choices": ["billing", "support"]},
        "urgent": {"question": "Is this urgent?", "choices": "boolean"},
    }
    llm_decision_model = decisions.LLMDecisionModel(model=OpenAIModel(model_id=DECISION_MODEL, params=MODEL_OPTIONS))
    typesafe_decision_model = decisions.TypeSafeDecisionModel()

    @tool
    async def route_ticket(ticket: str) -> str:
        """Route a customer support ticket to the right team.

        Args:
            ticket: The customer's message.
        """
        llm_result = await llm_decision_model.ask(ticket, questions)
        typesafe_result = await typesafe_decision_model.ask(ticket, questions)
        return f"llm={llm_result.answers} typesafe={typesafe_result.answers}"

    agent = Agent(
        model=OpenAIModel(model_id=TEST_MODEL, params=MODEL_OPTIONS),
        tools=[route_ticket],
        system_prompt="Always call route_ticket exactly once with the user's message, then report its result.",
    )

    await agent.invoke_async("I was charged twice for my subscription this month, please refund one charge.")

    spans = memory_logger.pop()
    names = [span["span_attributes"]["name"] for span in spans]
    tool_spans = [
        span
        for span in find_spans_by_type(spans, SpanTypeAttribute.TOOL)
        if span["span_attributes"]["name"] == "route_ticket.execute"
    ]
    assert len(tool_spans) == 1, names
    tool_span = tool_spans[0]
    decision_spans = {
        span["span_attributes"]["name"]: span for span in find_spans_by_type(spans, SpanTypeAttribute.DECISION)
    }
    assert set(decision_spans) == {"LLMDecisionModel.ask", "TypeSafeDecisionModel.ask"}, names

    expected_input = {
        "state": "I was charged twice for my subscription this month, please refund one charge.",
        "questions": [
            {
                "id": "department",
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": None, "support": None},
            },
            {"id": "urgent", "type": "boolean", "instructions": "Is this urgent?"},
        ],
    }
    for span in decision_spans.values():
        assert span["span_parents"] == [tool_span["span_id"]]
        assert span["input"] == expected_input
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "strands-auto"
        assert span["metadata"]["latency_ms"] >= 0

    # LLMDecisionModel: answers only; the vended model's provider integration owns token metrics.
    llm_span = decision_spans["LLMDecisionModel.ask"]
    llm_answers = {answer["id"]: answer for answer in llm_span["output"]["answers"]}
    assert llm_answers["department"] == {"id": "department", "type": "choice", "choice": "billing"}
    # A forced categorical boolean is recorded as probability 1.0 / 0.0.
    assert llm_answers["urgent"]["type"] == "boolean"
    assert llm_answers["urgent"]["probability"] in (0.0, 1.0)
    assert llm_span["metadata"]["decision_model"] == "LLMDecisionModel"
    assert llm_span["metadata"]["provider"] == "openai"
    assert llm_span["metadata"]["model"] == DECISION_MODEL
    assert "tokens" not in llm_span.get("metrics", {})

    # TypeSafeDecisionModel: System One probabilities and confidence over raw HTTP.
    typesafe_span = decision_spans["TypeSafeDecisionModel.ask"]
    typesafe_answers = {answer["id"]: answer for answer in typesafe_span["output"]["answers"]}
    assert typesafe_answers["department"]["choice"] == "billing"
    assert set(typesafe_answers["department"]["probabilities"]) == {"billing", "support"}
    assert 0 <= typesafe_answers["department"]["confidence"] <= 1
    assert set(typesafe_answers["urgent"]) == {"id", "type", "probability"}
    assert typesafe_answers["urgent"]["type"] == "boolean"
    assert 0 <= typesafe_answers["urgent"]["probability"] <= 1
    assert typesafe_span["metadata"]["decision_model"] == "TypeSafeDecisionModel"
    assert typesafe_span["metadata"]["provider"] == "typesafe"
    # The provider-reported model replaces the requested ``jev-latest`` alias.
    assert typesafe_span["metadata"]["model"] == "jev-1.14.0"
    assert typesafe_span["metrics"]["tokens"] == (
        typesafe_span["metrics"]["prompt_tokens"] + typesafe_span["metrics"]["completion_tokens"]
    )


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_strands_decision_image_state_is_logged_as_attachment(memory_logger):
    decisions = pytest.importorskip("strands.decisions", reason="DecisionModel requires strands-agents >= 1.63.0")
    from strands.models.openai import OpenAIModel

    assert StrandsIntegration.setup()

    image = (Path(__file__).parent.parent.parent / "fixtures" / "test-image.png").read_bytes()
    state = [{"text": "Describe the attached image."}, {"image": {"format": "png", "source": {"bytes": image}}}]
    decision_model = decisions.LLMDecisionModel(model=OpenAIModel(model_id=DECISION_MODEL, params=MODEL_OPTIONS))

    await decision_model.ask(state, {"has_image": {"question": "Is an image attached?", "choices": "boolean"}})

    (span,) = find_spans_by_type(memory_logger.pop(), SpanTypeAttribute.DECISION)
    logged_state = span["input"]["state"]
    assert logged_state[0] == {"text": "Describe the attached image."}
    attachment = logged_state[1]["image"]["source"]["bytes"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "image/png"
    assert logged_state[1]["image"]["format"] == "png"
    # The model request keeps the original bytes.
    assert state[1]["image"]["source"]["bytes"] == image
