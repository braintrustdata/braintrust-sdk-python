"""Tests for BraintrustPlugin combined with Pydantic AI's Temporal durability integration."""

import asyncio
import uuid
import warnings
from copy import deepcopy

import pytest


pytest.importorskip("temporalio")
pytest.importorskip("pydantic_ai.durable_exec.temporal")

import braintrust
import pydantic_ai.durable_exec.temporal as pydantic_ai_temporal
import temporalio.workflow
from braintrust.integrations.pydantic_ai import PydanticAIIntegration
from braintrust.integrations.temporal import BraintrustPlugin
from braintrust.integrations.temporal.test_temporal import (  # noqa: F401
    _spans_named,
    _wait_for_workflow_state,
    memory_logger,
    temporal_env,
)
from braintrust.merge_row_batch import merge_row_batch
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin
from pydantic_ai.models.test import TestModel
from temporalio.client import Client
from temporalio.worker import Worker


_HAS_DURABILITY_CAPABILITY = hasattr(pydantic_ai_temporal, "TemporalDurability")
_DURABILITY_APIS = [
    "wrapper",
    pytest.param(
        "capability",
        marks=pytest.mark.skipif(
            not _HAS_DURABILITY_CAPABILITY, reason="TemporalDurability needs a newer pydantic-ai"
        ),
    ),
]


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
    _wrapper_chat = pydantic_ai_temporal.TemporalAgent(
        Agent(TestModel(custom_output_text="chat done"), name="wrapper_chat_agent")
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
    _capability_chat = Agent(
        TestModel(custom_output_text="chat done"),
        name="capability_chat_agent",
        capabilities=[pydantic_ai_temporal.TemporalDurability()],
    )

_AGENTS = {
    "wrapper": (_wrapper_parent, _wrapper_sub),
    **({"capability": (_capability_parent, _capability_sub)} if _HAS_DURABILITY_CAPABILITY else {}),
}
_CHAT_AGENTS = {
    "wrapper": _wrapper_chat,
    **({"capability": _capability_chat} if _HAS_DURABILITY_CAPABILITY else {}),
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


@temporalio.workflow.defn
class ChatWorkflow:
    """Keep a chat span open between agent runs while waiting for a later message."""

    def __init__(self) -> None:
        self._message: str | None = None
        self._state = "starting"

    @temporalio.workflow.run
    async def run(self, prompt: str) -> list[str]:
        agent = _CHAT_AGENTS[_api_for_current_workflow()]
        with braintrust.start_span(name="chat_session", type="task") as span:
            first = await agent.run(prompt)
            span.log(input=prompt, metadata={"first_turn_complete": True})
            self._state = "waiting"
            await temporalio.workflow.wait_condition(lambda: self._message is not None)
            second = await agent.run(self._message, message_history=first.all_messages())
            output = [first.output, second.output]
            span.log(output=output)
            return output

    @temporalio.workflow.signal
    def send_message(self, message: str) -> None:
        self._message = message

    @temporalio.workflow.query
    def state(self) -> str:
        return self._state


# With no workflow cache, every workflow task replays the run from the start, as after a worker
# restart or cache eviction. Replays must not log the in-workflow Pydantic AI spans again.
@pytest.mark.parametrize("max_cached_workflows", [1000, 0], ids=["cached", "replay"])
@pytest.mark.parametrize("api", _DURABILITY_APIS)
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


@pytest.mark.parametrize("api", _DURABILITY_APIS)
@pytest.mark.asyncio
async def test_chat_spans_survive_worker_restart(temporal_env, memory_logger, api):  # noqa: F811
    assert PydanticAIIntegration.setup()
    agent = _CHAT_AGENTS[api]
    client = await Client.connect(
        temporal_env.client.service_client.config.target_host,
        namespace=temporal_env.client.namespace,
        plugins=[PydanticAIPlugin()],
    )
    task_queue = f"test-chat-restart-{uuid.uuid4()}"

    def worker():
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[ChatWorkflow],
            max_cached_workflows=0,
            plugins=[AgentPlugin(agent), BraintrustPlugin(logger=memory_logger)],
        )

    async with worker():
        handle = await client.start_workflow(
            ChatWorkflow.run,
            "first message",
            id=f"{api}:{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await _wait_for_workflow_state(handle, "waiting")

    # Shutdown evicts any cached workflow. Capture what was actually exported before replay:
    # a still-running chat must remain open, including after its coroutine has been unwound.
    braintrust.flush()
    before_restart = memory_logger.pop()
    (original_workflow,) = _spans_named(before_restart, "temporal.workflow.ChatWorkflow")
    (original_chat,) = _spans_named(before_restart, "chat_session")
    for span in (original_workflow, original_chat):
        assert "end" not in span["metrics"]
        assert "error" not in span

    # A new worker must rebuild the context from history, then append the next turn to the
    # original trace. No in-memory span or interceptor state is carried across workers.
    async with worker():
        await _wait_for_workflow_state(handle, "waiting")
        await handle.signal(ChatWorkflow.send_message, "second message")
        assert await asyncio.wait_for(handle.result(), timeout=15) == ["chat done", "chat done"]

    braintrust.flush()
    spans = merge_row_batch(deepcopy(before_restart) + memory_logger.pop())
    (workflow_span,) = _spans_named(spans, "temporal.workflow.ChatWorkflow")
    (chat_span,) = _spans_named(spans, "chat_session")
    for original, resumed in ((original_workflow, workflow_span), (original_chat, chat_span)):
        for field in ("id", "span_id", "root_span_id"):
            assert resumed[field] == original[field]
        assert resumed["metrics"]["start"] == original["metrics"]["start"]
        assert resumed["metrics"]["end"] > original["metrics"]["start"]
        assert "error" not in resumed
    assert chat_span["input"] == "first message"
    assert chat_span["metadata"]["first_turn_complete"] is True
    assert chat_span["output"] == ["chat done", "chat done"]
    assert chat_span["span_parents"] == [workflow_span["span_id"]]
    agent_spans = _spans_named(spans, f"agent_run [{agent.name}]")
    assert len(agent_spans) == 2
    assert {span["root_span_id"] for span in spans} == {workflow_span["root_span_id"]}
    assert all(span["span_parents"] == [chat_span["span_id"]] for span in agent_spans)
    assert all(chat_span["metrics"]["end"] >= span["metrics"]["end"] for span in agent_spans)
