# pyright: reportPrivateUsage=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.agno import setup_agno
from braintrust.wrappers.agno.workflow import wrap_workflow

from ._test_agno_helpers import (
    PROJECT_NAME,
    FakeExecutionInput,
    FakeWorkflowRunResponse,
    make_fake_duplicate_content_workflow,
    make_fake_streaming_workflow_with_mutated_run_response,
    make_fake_workflow,
    make_fake_workflow_agent_path,
    make_fake_workflow_with_async_agent,
)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(Path(__file__).parent.parent / "cassettes"),
    }


@pytest.fixture(scope="module", autouse=True)
def setup_wrapper():
    setup_agno(project_name=PROJECT_NAME)
    yield


@pytest.mark.vcr
def test_agno_workflow_with_agent(memory_logger):
    agent_module = pytest.importorskip("agno.agent")
    workflow_module = pytest.importorskip("agno.workflow")
    openai_module = pytest.importorskip("agno.models.openai")
    Agent = agent_module.Agent
    Workflow = workflow_module.Workflow
    OpenAIChat = openai_module.OpenAIChat

    assert not memory_logger.pop()

    author_agent = Agent(
        name="Author Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="You are librarian. Answer the questions by only replying with the author that wrote the book.",
    )

    workflow = Workflow(
        name="Book Lookup Workflow",
        steps=[author_agent],
    )

    response = workflow.run("Charlotte's Web")

    assert response
    assert response.content
    assert len(response.content) > 0

    spans = memory_logger.pop()
    assert len(spans) >= 3, f"Expected at least 3 spans (workflow + agent + llm), got {len(spans)}"

    workflow_span = spans[0]
    assert workflow_span["span_attributes"]["name"] == "Book Lookup Workflow.run"
    assert workflow_span["span_attributes"]["type"].value == "task"
    assert workflow_span["metadata"]["component"] == "workflow"
    assert workflow_span["metadata"]["workflow_name"] == "Book Lookup Workflow"
    assert workflow_span["metadata"]["steps_count"] == 1

    agent_span = None
    for span in spans[1:]:
        if "Agent" in span["span_attributes"]["name"] and ".run" in span["span_attributes"]["name"]:
            agent_span = span
            break

    assert agent_span is not None, "Could not find agent span"
    assert agent_span["span_parents"] == [workflow_span["span_id"]]

    llm_span = None
    for span in spans:
        if span["span_attributes"]["type"].value == "llm":
            llm_span = span
            break

    assert llm_span is not None, "Could not find LLM span"
    assert llm_span["span_parents"] == [agent_span["span_id"]]


@pytest.mark.asyncio
async def test_agno_workflow_async_execution_input_extraction(memory_logger):
    Workflow = wrap_workflow(make_fake_workflow("CompatWorkflowAsyncInput"))
    workflow = Workflow()

    execution_input = FakeExecutionInput({"topic": "async-workflow"})
    run_response = FakeWorkflowRunResponse(input={"stale": True})

    result = await workflow._aexecute("session-1", "user-1", execution_input, run_response)

    assert result.content == "workflow-async"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowAsyncInput.arun"
    assert span["input"]["input"] == {"topic": "async-workflow"}
    assert span["input"]["execution_input"]["input"] == {"topic": "async-workflow"}
    assert span["input"]["execution_input"]["kind"] == "workflow-execution"
    assert span["input"]["run_response"]["input"] == {"stale": True}


@pytest.mark.asyncio
async def test_agno_async_workflow_agent_arun_metadata_includes_workflow_fields(memory_logger):
    Workflow = wrap_workflow(make_fake_workflow_with_async_agent("CompatAsyncWorkflow", "NestedAsyncAgent"))
    workflow = Workflow()

    execution_input = FakeExecutionInput({"topic": "async-workflow"})
    run_response = FakeWorkflowRunResponse(input={"topic": "async-workflow"})

    result = await workflow._aexecute("session-1", "user-1", execution_input, run_response)
    assert result.content == "{'topic': 'async-workflow'}-async"

    spans = memory_logger.pop()
    assert len(spans) == 2

    workflow_span = next(s for s in spans if s["span_attributes"]["name"] == "CompatAsyncWorkflow.arun")
    agent_span = next(s for s in spans if s["span_attributes"]["name"] == "NestedAsyncAgent.arun")

    assert agent_span["span_parents"] == [workflow_span["span_id"]]
    assert workflow_span["metadata"]["workflow_id"] == "workflow-123"
    assert workflow_span["metadata"]["workflow_name"] == "CompatAsyncWorkflow"
    assert agent_span["metadata"]["workflow_id"] == "workflow-123"
    assert agent_span["metadata"]["workflow_name"] == "CompatAsyncWorkflow"


def test_agno_workflow_stream_aggregates_workflow_events(memory_logger):
    Workflow = wrap_workflow(make_fake_workflow("CompatWorkflowStream"))
    workflow = Workflow()

    execution_input = FakeExecutionInput("hello world")
    run_response = FakeWorkflowRunResponse(input="hello world")

    chunks = list(workflow._execute_stream("session-1", execution_input, run_response))
    assert len(chunks) == 4

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowStream.run_stream"
    assert span["output"]["content"] == "hello world"
    assert span["output"]["status"] == "COMPLETED"
    assert span["metrics"]["prompt_tokens"] == 1
    assert span["metrics"]["completion_tokens"] == 2


def test_agno_workflow_stream_prefers_final_workflow_output(memory_logger):
    Workflow = wrap_workflow(make_fake_duplicate_content_workflow("CompatWorkflowDuplicateContent"))
    workflow = Workflow()

    execution_input = FakeExecutionInput("hello")
    run_response = FakeWorkflowRunResponse(input="hello")

    chunks = list(workflow._execute_stream("session-1", execution_input, run_response))
    assert len(chunks) == 2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowDuplicateContent.run_stream"
    assert span["output"]["content"] == "hello"
    assert span["output"]["status"] == "COMPLETED"


def test_agno_workflow_stream_preserves_final_run_response_fields(memory_logger):
    Workflow = wrap_workflow(
        make_fake_streaming_workflow_with_mutated_run_response("CompatWorkflowMutatedRunResponse")
    )
    workflow = Workflow()

    execution_input = FakeExecutionInput("hello world")
    run_response = FakeWorkflowRunResponse(input="hello world")

    chunks = list(workflow._execute_stream("session-1", execution_input, run_response))
    assert len(chunks) == 3

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowMutatedRunResponse.run_stream"
    assert span["output"]["content"] == "hello world"
    assert span["output"]["status"] == "FAILED"
    assert span["metrics"]["prompt_tokens"] == 1
    assert span["metrics"]["completion_tokens"] == 2


def test_agno_workflow_agent_path_sync_run_creates_workflow_span(memory_logger):
    Workflow = wrap_workflow(make_fake_workflow_agent_path("CompatWorkflowAgentPath"))
    workflow = Workflow()

    execution_input = FakeExecutionInput("hello")
    result = workflow._execute_workflow_agent("hello", "session-1", execution_input, "run-context")

    assert result.content == "hello-sync"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowAgentPath.run"
    assert span["input"]["input"] == "hello"
    assert span["metadata"]["workflow_id"] == "workflow-agent-123"


@pytest.mark.asyncio
async def test_agno_workflow_agent_path_async_stream_creates_workflow_span(memory_logger):
    Workflow = wrap_workflow(make_fake_workflow_agent_path("CompatWorkflowAgentPathAsync"))
    workflow = Workflow()

    execution_input = FakeExecutionInput("hello")
    stream = await workflow._aexecute_workflow_agent("hello", "run-context", execution_input, stream=True)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "CompatWorkflowAgentPathAsync.arun_stream"
    assert span["output"]["content"] == "hello-async-stream"
    assert span["metadata"]["workflow_id"] == "workflow-agent-123"
