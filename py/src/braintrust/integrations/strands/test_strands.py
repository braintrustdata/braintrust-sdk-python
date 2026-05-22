# pylint: disable=import-error

import pytest
from braintrust import logger
from braintrust.integrations.strands import StrandsIntegration
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type, init_test_logger


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

    model = OpenAIModel(model_id="gpt-4o-mini", params={"temperature": 0, "max_tokens": 16})
    agent = Agent(model=model, name="bt-test-agent", system_prompt="Answer with one short sentence.")

    result = await agent.invoke_async("What is 2 + 2?")

    assert result.message["role"] == "assistant"
    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    names = [span["span_attributes"]["name"] for span in spans]
    assert any(span["span_attributes"]["name"] == "bt-test-agent.invoke" for span in task_spans), names
    assert any(span["span_attributes"]["name"] == "event_loop.cycle" for span in task_spans), names
    assert len(llm_spans) == 1
    llm_span = llm_spans[0]
    assert llm_span["input"]["messages"][0]["role"] == "user"
    assert llm_span["span_attributes"]["name"] == "gpt-4o-mini.chat"
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["metadata"]["stop_reason"] == "end_turn"
    assert llm_span["metadata"]["strands_usage"]["input_tokens"] > 0
    assert llm_span["metadata"]["strands_usage"]["output_tokens"] > 0
    assert "prompt_tokens" not in llm_span.get("metrics", {})
    assert "completion_tokens" not in llm_span.get("metrics", {})
    assert "tokens" not in llm_span.get("metrics", {})


def test_strands_setup_is_idempotent():
    assert StrandsIntegration.setup()
    assert StrandsIntegration.setup()
