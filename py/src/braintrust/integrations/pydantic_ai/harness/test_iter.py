import pytest
from braintrust import logger, setup_pydantic_ai
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_harness_iter_traces_model_and_tool():
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel

    persistence = pytest.importorskip("pydantic_ai_harness.step_persistence", exc_type=ModuleNotFoundError)

    project = "test-pydantic-ai-harness-iter"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)

    store = persistence.InMemoryStepStore()
    agent = Agent(
        OpenAIChatModel("gpt-4.1-mini"), name="harness_iter", capabilities=[persistence.StepPersistence(store=store)]
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        """Return the supplied text."""
        return text

    prompt = 'Call the echo tool exactly once with text "hello", then reply with only the tool result.'
    run_id = "harness-iter"
    with logger._internal_with_memory_background_logger() as memory_logger:
        async with agent.iter(prompt, run_id=run_id) as agent_run:
            async for _ in agent_run:
                pass
        spans = memory_logger.pop()

    assert agent_run.result is not None
    assert agent_run.result.output.strip().lower() == "hello"
    assert await store.list_events(run_id=run_id)

    run = find_span_by_name(spans, "agent_run [harness_iter]")
    assert run["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert [span for span in spans if not span.get("span_parents")] == [run]
    assert run["input"]["user_prompt"] == prompt
    assert run["output"]["output"] == agent_run.result.output

    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert span["span_parents"] == [run["span_id"]]

    tool = find_span_by_name(spans, "echo")
    assert tool["span_attributes"]["type"] == SpanTypeAttribute.TOOL
    assert tool["span_parents"] == [run["span_id"]]
    assert tool["input"] == {"text": "hello"}
    assert tool["output"] == "hello"
