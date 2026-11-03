"""Tests for BraintrustPlugin combined with Pydantic AI's Temporal durability integration."""

import uuid
import warnings

import pytest


pytest.importorskip("temporalio")
pytest.importorskip("pydantic_ai.durable_exec.temporal")

import braintrust
import pydantic_ai.durable_exec.temporal as pydantic_ai_temporal
import temporalio.workflow
from braintrust.integrations.pydantic_ai import PydanticAIIntegration
from braintrust.integrations.temporal import BraintrustPlugin
from braintrust.integrations.temporal.test_temporal import _spans_named, memory_logger, temporal_env  # noqa: F401
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin
from pydantic_ai.models.test import TestModel
from temporalio.client import Client
from temporalio.worker import Worker


_HAS_DURABILITY_CAPABILITY = hasattr(pydantic_ai_temporal, "TemporalDurability")


def _api_for_current_workflow() -> str:
    # Workflow IDs start with the API under test; see the test below.
    return temporalio.workflow.info().workflow_id.split(":", 1)[0]


async def _delegate(task: str) -> str:
    # Runs in workflow code and starts the sub-agent as a child workflow, like a `delegate_task` tool.
    return await temporalio.workflow.execute_child_workflow(
        SubAgentWorkflow.run,
        task,
        id=f"{temporalio.workflow.info().workflow_id}:sub",
    )


# Deprecated `TemporalAgent` wrapper: the only API on older pydantic-ai releases.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _wrapper_sub = pydantic_ai_temporal.TemporalAgent(
        Agent(TestModel(custom_output_text="sub done"), name="wrapper_sub_agent")
    )
    _wrapper_parent_agent = Agent(TestModel(call_tools=["delegate_task"]), name="wrapper_parent_agent")
    _wrapper_parent_agent.tool_plain(name="delegate_task")(_delegate)
    # delegate_task starts a child workflow, so it has to run in workflow code, not an activity.
    _wrapper_parent = pydantic_ai_temporal.TemporalAgent(
        _wrapper_parent_agent, tool_activity_config={"<agent>": {"delegate_task": False}}
    )

if _HAS_DURABILITY_CAPABILITY:
    _capability_sub = Agent(
        TestModel(custom_output_text="sub done"),
        name="capability_sub_agent",
        capabilities=[pydantic_ai_temporal.TemporalDurability()],
    )
    _capability_parent = Agent(
        TestModel(call_tools=["delegate_task"]),
        name="capability_parent_agent",
        capabilities=[pydantic_ai_temporal.TemporalDurability()],
    )
    # `metadata={"temporal": False}` replaces `tool_activity_config` on the capability path.
    _capability_parent.tool_plain(name="delegate_task", metadata={"temporal": False})(_delegate)

_AGENTS = {
    "wrapper": (_wrapper_parent, _wrapper_sub),
    **({"capability": (_capability_parent, _capability_sub)} if _HAS_DURABILITY_CAPABILITY else {}),
}


@temporalio.workflow.defn
class SubAgentWorkflow:
    @temporalio.workflow.run
    async def run(self, prompt: str) -> str:
        _, sub_agent = _AGENTS[_api_for_current_workflow()]
        return (await sub_agent.run(prompt)).output


@temporalio.workflow.defn
class ParentAgentWorkflow:
    @temporalio.workflow.run
    async def run(self, prompt: str) -> str:
        parent_agent, _ = _AGENTS[_api_for_current_workflow()]
        return (await parent_agent.run(prompt)).output


# With no workflow cache, every workflow task replays the run from the start, as after a worker
# restart or cache eviction. Replays must not log the in-workflow Pydantic AI spans again.
@pytest.mark.parametrize("max_cached_workflows", [1000, 0], ids=["cached", "replay"])
@pytest.mark.parametrize(
    "api",
    [
        "wrapper",
        pytest.param(
            "capability",
            marks=pytest.mark.skipif(
                not _HAS_DURABILITY_CAPABILITY, reason="TemporalDurability needs a newer pydantic-ai"
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_subagent_child_workflow_nests_under_delegate_task_span(
    temporal_env,  # noqa: F811
    memory_logger,  # noqa: F811
    api,
    max_cached_workflows,
):
    assert PydanticAIIntegration.setup()
    parent_agent, sub_agent = _AGENTS[api]
    client = await Client.connect(
        temporal_env.client.service_client.config.target_host,
        namespace=temporal_env.client.namespace,
        plugins=[PydanticAIPlugin()],
    )
    task_queue = f"test-queue-pydantic-ai-{api}-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ParentAgentWorkflow, SubAgentWorkflow],
        max_cached_workflows=max_cached_workflows,
        plugins=[AgentPlugin(parent_agent), AgentPlugin(sub_agent), BraintrustPlugin(logger=memory_logger)],
    ):
        await client.execute_workflow(
            ParentAgentWorkflow.run,
            "delegate something",
            id=f"{api}:{uuid.uuid4()}",
            task_queue=task_queue,
        )

    braintrust.flush()
    spans = memory_logger.pop()
    (parent_workflow_span,) = _spans_named(spans, "temporal.workflow.ParentAgentWorkflow")
    (parent_agent_span,) = _spans_named(spans, f"agent_run [{parent_agent.name}]")
    (tool_span,) = _spans_named(spans, "delegate_task")
    (child_workflow_span,) = _spans_named(spans, "temporal.workflow.SubAgentWorkflow")
    (sub_agent_span,) = _spans_named(spans, f"agent_run [{sub_agent.name}]")

    assert parent_agent_span["span_parents"] == [parent_workflow_span["span_id"]]
    assert tool_span["span_parents"] == [parent_agent_span["span_id"]]
    assert child_workflow_span["span_parents"] == [tool_span["span_id"]]
    assert sub_agent_span["span_parents"] == [child_workflow_span["span_id"]]
    assert {span["root_span_id"] for span in spans} == {parent_workflow_span["root_span_id"]}

    # Each model request is traced once: one `chat` span paired with its model activity span. The
    # wrapper API traces the request in workflow code (activity under `chat`); the capability
    # API hands it to the activity first (`chat` under activity).
    chat_spans = [span for span in spans if span["span_attributes"]["name"].startswith("chat ")]
    model_activity_spans = [span for span in spans if span["span_attributes"]["name"].endswith("__model_request")]
    assert len(chat_spans) == len(model_activity_spans) == 3
    pairs = {
        frozenset((span["span_id"], span["span_parents"][0]))
        for span in chat_spans + model_activity_spans
        if span["span_parents"][0] in {other["span_id"] for other in chat_spans + model_activity_spans}
    }
    assert len(pairs) == 3
