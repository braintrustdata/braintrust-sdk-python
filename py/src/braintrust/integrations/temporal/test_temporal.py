"""Unit tests for Braintrust Temporal interceptor."""

import asyncio
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import pytest
import pytest_asyncio
from braintrust.integrations.test_utils import verify_autoinstrument_script


pytest.importorskip("temporalio")

import braintrust
import temporalio.activity
import temporalio.api.common.v1
import temporalio.converter
import temporalio.testing
import temporalio.worker
import temporalio.workflow
from braintrust.integrations.temporal import BraintrustInterceptor, BraintrustPlugin
from braintrust.integrations.temporal.plugin import _workflow_span_context, _workflow_span_ids
from braintrust.span_identifier_v3 import SpanComponentsV3, SpanObjectTypeV3
from braintrust.span_identifier_v4 import SpanComponentsV4
from braintrust.test_helpers import init_test_logger, preserve_env_vars
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker


@dataclass
class WorkflowInfoForTest:
    namespace: str
    workflow_type: str
    workflow_id: str
    run_id: str


class TestHeaderSerialization:
    """Unit tests for header serialization/deserialization."""

    def test_span_context_header_roundtrip(self):
        interceptor = BraintrustInterceptor()

        # An empty span context adds no header, and a missing header reads back as None.
        empty_headers: dict[str, temporalio.api.common.v1.Payload] = {}
        assert interceptor._span_context_to_headers({}, empty_headers) == {}
        assert interceptor._span_context_from_headers(empty_headers) is None

        original_context = {
            "trace_id": "test-trace-id",
            "span_id": "test-span-id",
            "root_span_id": "test-root-span-id",
        }
        existing_payload = interceptor.payload_converter.to_payloads(["existing_value"])[0]

        headers = interceptor._span_context_to_headers(original_context, {"existing_header": existing_payload})

        # Existing headers are preserved alongside the Braintrust span header.
        assert set(headers) == {"existing_header", "_braintrust-span"}
        assert headers["existing_header"] == existing_payload
        assert interceptor._span_context_from_headers(headers) == original_context


class TestWorkflowSpanContext:
    def test_workflow_span_context_preserves_legacy_parent_encoding(self):
        parent_components = SpanComponentsV3(
            object_type=SpanObjectTypeV3.PROJECT_LOGS,
            object_id=str(uuid.uuid4()),
            row_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            root_span_id=str(uuid.uuid4()),
        )
        parent = parent_components.to_str()
        info = WorkflowInfoForTest(
            namespace="default",
            workflow_type="ReplayAfterSignalWorkflow",
            workflow_id="workflow-id",
            run_id="run-id",
        )

        with preserve_env_vars("BRAINTRUST_LEGACY_IDS"):
            os.environ.pop("BRAINTRUST_LEGACY_IDS", None)
            ids = _workflow_span_ids(cast(temporalio.workflow.Info, info), parent)
            context = _workflow_span_context(parent, ids)

        assert SpanComponentsV4.get_version(context) == 3
        parsed = SpanComponentsV3.from_str(context)
        assert parsed.row_id == ids["row_id"]
        assert parsed.span_id == ids["span_id"]
        assert parsed.root_span_id == parent_components.root_span_id

    def test_workflow_span_context_uses_stable_root_for_object_parent(self):
        parent = SpanComponentsV4(
            object_type=SpanObjectTypeV3.PROJECT_LOGS,
            object_id=str(uuid.uuid4()),
        ).to_str()
        info = WorkflowInfoForTest(
            namespace="default",
            workflow_type="ReplayAfterSignalWorkflow",
            workflow_id="workflow-id",
            run_id="run-id",
        )

        ids = _workflow_span_ids(cast(temporalio.workflow.Info, info), parent)
        context = _workflow_span_context(parent, ids)

        assert SpanComponentsV4.get_version(context) == 4
        parsed = SpanComponentsV4.from_str(context)
        assert parsed.row_id == ids["row_id"]
        assert parsed.span_id == ids["span_id"]
        assert parsed.root_span_id == ids["root_span_id"]


# Integration Test Infrastructure


@dataclass
class TaskInput:
    """Input for test activities and workflows."""

    value: int


# Test Workflows and Activities


@temporalio.activity.defn
async def simple_activity(input: TaskInput) -> int:
    """Simple test activity."""
    await asyncio.sleep(0.1)
    return input.value + 10


@temporalio.activity.defn
async def failing_activity(input: TaskInput) -> int:
    """Activity that fails on first attempt."""
    info = temporalio.activity.info()
    attempt = info.attempt

    if attempt == 1:
        raise ValueError("Simulated failure on first attempt")

    return input.value + 20


@temporalio.activity.defn
async def simple_local_activity(input: TaskInput) -> int:
    """Simple local activity."""
    return input.value + 5


@temporalio.workflow.defn
class WorkflowWithRetry:
    """Workflow that executes an activity with retries."""

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        result = await temporalio.workflow.execute_activity(
            failing_activity,
            input,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
            ),
        )

        return result


@temporalio.workflow.defn
class WorkflowWithLocalActivity:
    """Workflow that executes a local activity."""

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        result = await temporalio.workflow.execute_local_activity(
            simple_local_activity,
            input,
            start_to_close_timeout=timedelta(seconds=5),
        )

        return result


@temporalio.workflow.defn
class ChildWorkflow:
    """Child workflow for testing child workflow tracing."""

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        result = await temporalio.workflow.execute_activity(
            simple_activity,
            input,
            start_to_close_timeout=timedelta(seconds=10),
        )

        return result


@temporalio.workflow.defn
class ParentWorkflow:
    """Parent workflow that spawns a child workflow."""

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        # Execute child workflow
        child_result = await temporalio.workflow.execute_child_workflow(
            ChildWorkflow.run,
            input,
            id=f"child-{temporalio.workflow.info().workflow_id}",
        )

        return child_result


@temporalio.workflow.defn
class ReplayAfterSignalWorkflow:
    """Workflow that schedules an activity after a later workflow task."""

    def __init__(self) -> None:
        self._continue = False
        self._state = "starting"

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        first_result = await temporalio.workflow.execute_activity(
            simple_activity,
            input,
            start_to_close_timeout=timedelta(seconds=10),
        )
        self._state = "waiting"
        await temporalio.workflow.wait_condition(lambda: self._continue)
        self._state = "continued"
        return await temporalio.workflow.execute_activity(
            simple_activity,
            TaskInput(value=first_result),
            start_to_close_timeout=timedelta(seconds=10),
        )

    @temporalio.workflow.signal
    def continue_workflow(self) -> None:
        self._continue = True

    @temporalio.workflow.query
    def state(self) -> str:
        return self._state


class TestAutoInstrumentation:
    """Tests for Temporal auto-instrumentation helpers."""

    def test_auto_instrument_temporal_subprocess(self):
        verify_autoinstrument_script("test_auto_temporal.py")

    def test_contrib_temporal_compat_import_deprecated(self):
        with pytest.warns(DeprecationWarning, match="braintrust.contrib.temporal is deprecated"):
            import importlib
            import sys

            sys.modules.pop("braintrust.contrib.temporal", None)
            compat = importlib.import_module("braintrust.contrib.temporal")

        assert compat.BraintrustPlugin is BraintrustPlugin


# Integration Tests


@pytest_asyncio.fixture(scope="function")
async def temporal_env():
    """Create a Temporal test environment.

    If ``BRAINTRUST_TEMPORAL_TEST_SERVER_DIR`` is set, point the SDK's binary
    download cache at that directory and pin a long TTL so existing binaries
    are reused. CI sets this so a cached directory restored from the GitHub
    Actions cache shortcuts the download to temporal.download (which rate
    limits CI runners). When the var is unset (the local default), the SDK
    falls back to its built-in temp-dir download behavior.
    """
    kwargs: dict[str, Any] = {}
    cache_dir = os.environ.get("BRAINTRUST_TEMPORAL_TEST_SERVER_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kwargs["download_dest_dir"] = cache_dir
        kwargs["test_server_download_ttl"] = timedelta(days=365)
    async with await temporalio.testing.WorkflowEnvironment.start_time_skipping(**kwargs) as env:
        yield env


@pytest.fixture
def memory_logger():
    """Set up memory logger to capture spans for testing."""
    init_test_logger("temporal-test")
    with braintrust.logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _spans_named(spans: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [span for span in spans if span.get("span_attributes", {}).get("name") == name]


async def _wait_for_workflow_state(handle: Any, expected: str) -> None:
    for _ in range(50):
        if await handle.query(ReplayAfterSignalWorkflow.state) == expected:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"Workflow did not reach state {expected!r}")


class TestBraintrustPluginIntegration:
    """Integration tests for BraintrustPlugin with real Temporal workflows."""

    @pytest.mark.parametrize("with_client_parent", [False, True])
    @pytest.mark.asyncio
    async def test_plugin_activity_after_replay_stays_under_workflow_span(
        self, temporal_env, memory_logger, with_client_parent
    ):
        task_queue = f"test-queue-replay-{with_client_parent}-{uuid.uuid4()}"
        workflow_id = f"test-workflow-replay-{with_client_parent}-{uuid.uuid4()}"
        workflow_client = temporal_env.client
        if with_client_parent:
            workflow_client = await Client.connect(
                temporal_env.client.service_client.config.target_host,
                namespace=temporal_env.client.namespace,
                plugins=[BraintrustPlugin(logger=memory_logger)],
            )

        async with Worker(
            temporal_env.client,
            task_queue=task_queue,
            workflows=[ReplayAfterSignalWorkflow],
            activities=[simple_activity],
            max_cached_workflows=0,
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            if with_client_parent:
                with braintrust.start_span(name="test.client_replay_operation", type="task"):
                    handle = await workflow_client.start_workflow(
                        ReplayAfterSignalWorkflow.run,
                        TaskInput(value=10),
                        id=workflow_id,
                        task_queue=task_queue,
                    )
            else:
                handle = await workflow_client.start_workflow(
                    ReplayAfterSignalWorkflow.run,
                    TaskInput(value=10),
                    id=workflow_id,
                    task_queue=task_queue,
                )
            await _wait_for_workflow_state(handle, "waiting")
            await handle.signal(ReplayAfterSignalWorkflow.continue_workflow)
            assert await handle.result() == 30

        braintrust.flush()
        spans = memory_logger.pop()
        workflow_spans = _spans_named(spans, "temporal.workflow.ReplayAfterSignalWorkflow")
        activity_spans = _spans_named(spans, "temporal.activity.simple_activity")

        assert len(workflow_spans) == 1
        assert len(activity_spans) == 2
        workflow_span = workflow_spans[0]
        assert workflow_span["context"]["span_origin"]["instrumentation"]["name"] == "temporal-auto"
        assert all(
            span["context"]["span_origin"]["instrumentation"]["name"] == "temporal-auto" for span in activity_spans
        )
        assert all(workflow_span["span_id"] in span.get("span_parents", []) for span in activity_spans)
        assert all(workflow_span["root_span_id"] == span["root_span_id"] for span in activity_spans)
        if with_client_parent:
            client_spans = _spans_named(spans, "test.client_replay_operation")
            assert len(client_spans) == 1
            assert workflow_span["root_span_id"] == client_spans[0]["root_span_id"]

    @pytest.mark.parametrize(
        "workflow,input_value,expected_result,expected_span_counts,expected_activity_errors",
        [
            # failing_activity fails its first attempt, so the retry yields a second activity span.
            pytest.param(
                WorkflowWithRetry,
                30,
                50,
                {"temporal.workflow.WorkflowWithRetry": 1, "temporal.activity.failing_activity": 2},
                [True, False],
                id="activity_retry",
            ),
            pytest.param(
                ParentWorkflow,
                40,
                50,
                {
                    "temporal.workflow.ParentWorkflow": 1,
                    "temporal.workflow.ChildWorkflow": 1,
                    "temporal.activity.simple_activity": 1,
                },
                [False],
                id="child_workflow",
            ),
            # Local activities execute in the worker process and are traced like regular activities.
            pytest.param(
                WorkflowWithLocalActivity,
                100,
                105,
                {"temporal.workflow.WorkflowWithLocalActivity": 1, "temporal.activity.simple_local_activity": 1},
                [False],
                id="local_activity",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_plugin_workflow_tracing(
        self,
        temporal_env,
        memory_logger,
        workflow,
        input_value,
        expected_result,
        expected_span_counts,
        expected_activity_errors,
    ):
        task_queue = f"test-queue-{uuid.uuid4()}"
        async with Worker(
            temporal_env.client,
            task_queue=task_queue,
            workflows=[WorkflowWithRetry, ParentWorkflow, ChildWorkflow, WorkflowWithLocalActivity],
            activities=[failing_activity, simple_activity, simple_local_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            result = await temporal_env.client.execute_workflow(
                workflow.run,
                TaskInput(value=input_value),
                id=f"test-workflow-{uuid.uuid4()}",
                task_queue=task_queue,
            )
        assert result == expected_result

        spans = memory_logger.pop()
        assert Counter(span["span_attributes"]["name"] for span in spans) == expected_span_counts

        activity_spans = sorted(
            (span for span in spans if span["span_attributes"]["name"].startswith("temporal.activity.")),
            key=lambda span: span["metrics"]["start"],
        )
        assert [bool(span.get("error")) for span in activity_spans] == expected_activity_errors
        if any(expected_activity_errors):
            assert "Simulated failure on first attempt" in activity_spans[0]["error"]
