import asyncio

import braintrust
import pytest
from braintrust import logger
from braintrust.integrations.openai_agents import BraintrustTracingProcessor, OpenAIAgentsIntegration
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "test-project-openai-agents-tracing"
TEST_MODEL = "gpt-4o-mini"
TEST_PROMPT = "What is 2+2? Just the number."
TEST_AGENT_INSTRUCTIONS = "You are a helpful assistant. Be very concise."


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(autouse=True)
def isolate_openai_agents_tracing():
    pytest.importorskip("agents", reason="agents package not available")

    import agents

    provider = agents.tracing.get_trace_provider()
    processors = tuple(getattr(getattr(provider, "_multi_processor", None), "_processors", ()))
    manual_disabled = getattr(provider, "_manual_disabled", None)

    yield

    provider.set_processors(list(processors))
    provider._manual_disabled = manual_disabled
    if hasattr(provider, "_refresh_disabled_flag"):
        provider._refresh_disabled_flag()


def test_tracing_processor_trace_spans(memory_logger):
    """Trace metadata flows to the trace span, and on_trace_start sets it current so nested spans work."""
    assert not memory_logger.pop()
    processor = BraintrustTracingProcessor()

    class DummyTrace:
        def __init__(self, trace_id, name, metadata):
            self.trace_id = trace_id
            self.name = name
            self.metadata = metadata

        def export(self):
            return {"group_id": self.trace_id, "metadata": self.metadata}

    # Without a current span the trace becomes a root span.
    root_trace = DummyTrace("test-trace", "Test Trace", {"conversation_id": "test-12345"})
    processor.on_trace_start(root_trace)
    processor.on_trace_end(root_trace)

    # With a current span the trace nests under it and becomes current itself.
    nested_trace = DummyTrace("test-trace-id", "test-trace", {"foo": "bar"})
    with braintrust.start_span(name="parent-span") as parent_span:
        assert braintrust.current_span() == parent_span
        processor.on_trace_start(nested_trace)
        created_span = processor._spans[nested_trace.trace_id]
        assert braintrust.current_span() == created_span

        processor.on_trace_end(nested_trace)
        assert braintrust.current_span() == parent_span

    spans = {span["span_attributes"]["name"]: span for span in memory_logger.pop()}
    root_span = spans[root_trace.name]
    assert not root_span.get("span_parents")
    assert root_span["context"]["span_origin"]["instrumentation"]["name"] == "openai-agents-auto"
    assert root_span["metadata"]["conversation_id"] == "test-12345"
    assert spans[nested_trace.name]["span_parents"] == [spans["parent-span"]["span_id"]]


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_agents_integration_setup_creates_spans(memory_logger):
    import agents
    from agents import Agent
    from agents.run import AgentRunner

    assert not memory_logger.pop()

    assert OpenAIAgentsIntegration.setup() is True
    assert agents.tracing.get_trace_provider()._disabled is False

    agent = Agent(name="test-agent", model=TEST_MODEL, instructions=TEST_AGENT_INSTRUCTIONS)
    result = await AgentRunner().run(agent, TEST_PROMPT)

    assert result is not None
    assert hasattr(result, "final_output") or hasattr(result, "output")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    root_spans = [
        span
        for span in spans
        if span.get("span_attributes", {}).get("name") == "Agent workflow" and not span.get("span_parents")
    ]
    assert len(root_spans) == 1
    assert TEST_PROMPT in str(root_spans[0].get("input"))
    assert root_spans[0].get("output") is not None

    llm_spans = [span for span in spans if span.get("span_attributes", {}).get("type") == "llm"]
    assert llm_spans
    llm_metrics = [span.get("metrics", {}) for span in llm_spans]
    assert any(metrics.get("prompt_tokens") is not None for metrics in llm_metrics)
    assert any(metrics.get("completion_tokens") is not None for metrics in llm_metrics)
    assert any(metrics.get("tokens") is not None for metrics in llm_metrics)
    assert any(metrics.get("prompt_cached_tokens") == 0 for metrics in llm_metrics)
    assert any(metrics.get("completion_reasoning_tokens") == 0 for metrics in llm_metrics)

    # v0.14.0+ produces TaskSpanData and TurnSpanData wrapping agent runs. The
    # tracing processor must give them proper names, metadata, and usage
    # metrics rather than returning "Unknown" / empty dicts.
    if not hasattr(agents.tracing, "TaskSpanData") or not hasattr(agents.tracing, "TurnSpanData"):
        return

    assert len(spans) >= 3  # root + task + turn + agent + response at minimum

    # There should be no spans named "Unknown".
    unknown_spans = [s for s in spans if s.get("span_attributes", {}).get("name") == "Unknown"]
    assert unknown_spans == [], f"Found Unknown spans: {unknown_spans}"

    # Find TaskSpanData-derived span (name should be the workflow name, not "Unknown").
    task_spans = [
        s
        for s in spans
        if s.get("span_attributes", {}).get("name") == "Agent workflow"
        and s.get("span_parents")  # child of the root trace span, not the root itself
    ]
    assert task_spans, "Expected a task span child of the root trace"
    task_span = task_spans[0]
    assert task_span.get("span_attributes", {}).get("type") == "task"

    # Find TurnSpanData-derived span.
    turn_spans = [s for s in spans if "Turn" in (s.get("span_attributes", {}).get("name") or "")]
    assert turn_spans, "Expected at least one turn span"
    turn_span = turn_spans[0]
    assert turn_span.get("span_attributes", {}).get("type") == "task"
    # Turn span should have turn number and agent_name in metadata.
    assert turn_span.get("metadata", {}).get("turn") is not None
    assert turn_span.get("metadata", {}).get("agent_name") is not None


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_braintrust_tracing_processor_current_span_detection(memory_logger):
    import agents
    from agents import Agent
    from agents.run import AgentRunner

    assert not memory_logger.pop()

    @braintrust.traced(name="parent_span_test")
    async def test_function(instructions: str):
        detected_parent = braintrust.current_span()
        assert detected_parent is not None
        assert detected_parent != braintrust.logger.NOOP_SPAN

        processor = BraintrustTracingProcessor()

        agents.set_tracing_disabled(False)
        agents.add_trace_processor(processor)

        try:
            agent = Agent(name="test-agent", model=TEST_MODEL, instructions=TEST_AGENT_INSTRUCTIONS)
            runner = AgentRunner()
            result = await runner.run(agent, instructions)
            assert result is not None
            assert hasattr(result, "final_output") or hasattr(result, "output")
            return result
        finally:
            processor.shutdown()

    result = await test_function(TEST_PROMPT)
    assert result is not None

    spans = memory_logger.pop()
    assert len(spans) >= 2

    parent_span = None
    child_spans = []
    for span in spans:
        if span.get("span_attributes", {}).get("name") == "parent_span_test":
            parent_span = span
        elif span.get("span_attributes", {}).get("name") == "Agent workflow":
            child_spans.append(span)

    assert parent_span is not None
    assert child_spans

    child_span = child_spans[0]
    child_span_parents = child_span.get("span_parents", [])
    parent_span_id = parent_span.get("span_id")

    assert parent_span_id is not None
    assert isinstance(child_span_parents, list) and child_span_parents
    assert parent_span_id in child_span_parents
    assert child_span.get("root_span_id") == parent_span.get("root_span_id")
    assert parent_span.get("input") is not None
    assert parent_span.get("output") is not None

    all_child_spans = [s for s in spans if parent_span_id in (s.get("span_parents") or [])]
    assert all_child_spans
    span_types = [s.get("span_attributes", {}).get("type") for s in all_child_spans]
    assert "llm" in span_types or "task" in span_types


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_braintrust_tracing_processor_concurrency_bug(memory_logger):
    import agents
    from agents import Agent
    from agents.run import AgentRunner

    assert not memory_logger.pop()

    processor = BraintrustTracingProcessor()
    agents.set_tracing_disabled(False)
    agents.add_trace_processor(processor)

    try:
        agent_a = Agent(
            name="agent-a",
            model=TEST_MODEL,
            instructions="You are agent A. Just respond with 'A' and nothing else.",
        )
        agent_b = Agent(
            name="agent-b",
            model=TEST_MODEL,
            instructions="You are agent B. Just respond with 'B' and nothing else.",
        )
        runner = AgentRunner()

        async def run_agent_a():
            result = await runner.run(agent_a, "What's your name?")
            await asyncio.sleep(0.1)
            return result

        async def run_agent_b():
            return await runner.run(agent_b, "Who are you?")

        result_a, result_b = await asyncio.gather(run_agent_a(), run_agent_b())
        assert result_a is not None
        assert result_b is not None
    finally:
        processor.shutdown()

    spans = memory_logger.pop()
    assert len(spans) >= 2

    trace_spans = [
        span
        for span in spans
        if span.get("span_attributes", {}).get("name", "") == "Agent workflow" and not span.get("span_parents")
    ]
    assert len(trace_spans) == 2

    agent_a_trace = None
    agent_b_trace = None
    for trace in trace_spans:
        input_str = str(trace.get("input", ""))
        if "What's your name?" in input_str:
            agent_a_trace = trace
        elif "Who are you?" in input_str:
            agent_b_trace = trace

    assert agent_a_trace is not None
    assert agent_b_trace is not None
    assert agent_a_trace.get("input") is not None
    assert agent_a_trace.get("output") is not None
    assert agent_b_trace.get("input") is not None
    assert agent_b_trace.get("output") is not None
    assert agent_a_trace.get("input") != agent_b_trace.get("input")
    if agent_a_trace.get("output") and agent_b_trace.get("output"):
        assert agent_a_trace.get("output") != agent_b_trace.get("output")


class TestAutoInstrumentOpenAIAgents:
    """Tests for auto_instrument() with the OpenAI Agents SDK."""

    def test_auto_instrument_openai_agents(self):
        verify_autoinstrument_script("test_auto_openai_agents.py")
