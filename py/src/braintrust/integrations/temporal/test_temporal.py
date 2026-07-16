"""Unit tests for Braintrust Temporal interceptor."""

import asyncio
import os
import uuid
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

    def test_span_context_to_headers_with_valid_context(self):
        interceptor = BraintrustInterceptor()
        span_context = {"trace_id": "test-trace-id", "span_id": "test-span-id"}
        headers: dict[str, temporalio.api.common.v1.Payload] = {}

        result_headers = interceptor._span_context_to_headers(span_context, headers)

        assert "_braintrust-span" in result_headers
        assert len(result_headers) == 1

    def test_span_context_to_headers_with_empty_context(self):
        interceptor = BraintrustInterceptor()
        span_context: dict[str, Any] = {}
        headers: dict[str, temporalio.api.common.v1.Payload] = {}

        result_headers = interceptor._span_context_to_headers(span_context, headers)

        assert "_braintrust-span" not in result_headers
        assert len(result_headers) == 0

    def test_span_context_to_headers_preserves_existing_headers(self):
        interceptor = BraintrustInterceptor()
        span_context = {"trace_id": "test-trace-id"}

        # Create a payload for existing header
        existing_payload = interceptor.payload_converter.to_payloads(["existing_value"])[0]
        headers = {"existing_header": existing_payload}

        result_headers = interceptor._span_context_to_headers(span_context, headers)

        assert "existing_header" in result_headers
        assert "_braintrust-span" in result_headers
        assert len(result_headers) == 2

    def test_span_context_from_headers_with_valid_header(self):
        interceptor = BraintrustInterceptor()
        span_context = {"trace_id": "test-trace-id", "span_id": "test-span-id"}

        # Serialize span context to header
        payloads = interceptor.payload_converter.to_payloads([span_context])
        headers = {"_braintrust-span": payloads[0]}

        result = interceptor._span_context_from_headers(headers)

        assert result is not None
        assert result["trace_id"] == "test-trace-id"
        assert result["span_id"] == "test-span-id"

    def test_span_context_from_headers_with_missing_header(self):
        interceptor = BraintrustInterceptor()
        headers: dict[str, temporalio.api.common.v1.Payload] = {}

        result = interceptor._span_context_from_headers(headers)

        assert result is None

    def test_span_context_roundtrip(self):
        interceptor = BraintrustInterceptor()
        original_context = {
            "trace_id": "test-trace-id",
            "span_id": "test-span-id",
            "root_span_id": "test-root-span-id",
        }

        # Serialize
        headers = interceptor._span_context_to_headers(original_context, {})

        # Deserialize
        result_context = interceptor._span_context_from_headers(headers)

        assert result_context == original_context


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
class TestWorkflow:
    """Simple test workflow."""

    @temporalio.workflow.run
    async def run(self, input: TaskInput) -> int:
        # Execute an activity
        result = await temporalio.workflow.execute_activity(
            simple_activity,
            input,
            start_to_close_timeout=timedelta(seconds=10),
        )

        return result


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

    @pytest.mark.asyncio
    async def test_plugin_basic_workflow_tracing(self, temporal_env, memory_logger):
        """Test basic workflow and activity tracing with BraintrustPlugin.

        Verifies that:
        1. Braintrust can be imported directly in workflows (no unsafe.imports_passed_through)
        2. Spans are created for workflow execution
        3. Spans are created for activity execution
        """
        # Create worker with BraintrustPlugin
        async with Worker(
            temporal_env.client,
            task_queue="test-queue",
            workflows=[TestWorkflow],
            activities=[simple_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            # Execute workflow
            result = await temporal_env.client.execute_workflow(
                TestWorkflow.run,
                TaskInput(value=10),
                id=f"test-workflow-{uuid.uuid4()}",
                task_queue="test-queue",
            )

            # Verify workflow executed correctly
            assert result == 20  # 10 + 10 from activity

            # Flush to ensure all spans are captured
            braintrust.flush()

            # Get captured spans
            spans = memory_logger.pop()

            # Verify spans were created
            assert len(spans) > 0, f"Expected spans to be created, got {len(spans)} spans"

            # Verify workflow span was created
            workflow_spans = [s for s in spans if "temporal.workflow" in s.get("span_attributes", {}).get("name", "")]
            assert len(workflow_spans) > 0, (
                f"Expected workflow span to be created. Span names: {[s.get('span_attributes', {}).get('name', 'unknown') for s in spans]}"
            )

            # Verify activity span was created
            activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]
            assert len(activity_spans) > 0, (
                f"Expected activity span to be created. Span names: {[s.get('span_attributes', {}).get('name', 'unknown') for s in spans]}"
            )

    @pytest.mark.asyncio
    async def test_plugin_context_propagation(self, temporal_env, memory_logger):
        """Test that span context propagates from client to workflow to activity.

        Verifies that parent-child span relationships are maintained across
        the execution chain.
        """
        # Create a parent span at the client level
        with braintrust.start_span(name="test.client_operation", type="task") as parent_span:
            parent_context = parent_span.export()

            # Create worker with BraintrustPlugin
            async with Worker(
                temporal_env.client,
                task_queue="test-queue-2",
                workflows=[TestWorkflow],
                activities=[simple_activity],
                plugins=[BraintrustPlugin(logger=memory_logger)],
            ):
                # Execute workflow (context should propagate via headers)
                result = await temporal_env.client.execute_workflow(
                    TestWorkflow.run,
                    TaskInput(value=15),
                    id=f"test-workflow-ctx-{uuid.uuid4()}",
                    task_queue="test-queue-2",
                )

                assert result == 25  # 15 + 10

        # Get captured spans
        spans = memory_logger.pop()

        # Verify spans were created
        assert len(spans) > 0, "Expected spans to be created"

        # Verify client span exists
        client_spans = [s for s in spans if "test.client_operation" in s.get("span_attributes", {}).get("name", "")]
        assert len(client_spans) > 0, "Expected client span to be created"

        # Verify workflow and activity spans were created
        workflow_spans = [s for s in spans if "temporal.workflow" in s.get("span_attributes", {}).get("name", "")]
        activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]

        assert len(workflow_spans) > 0, "Expected workflow spans"
        assert len(activity_spans) > 0, "Expected activity spans"

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

    @pytest.mark.asyncio
    async def test_plugin_activity_retry_tracing(self, temporal_env, memory_logger):
        """Test that activity retries are properly traced.

        Verifies that each retry attempt creates a span with appropriate
        error information.
        """
        async with Worker(
            temporal_env.client,
            task_queue="test-queue-3",
            workflows=[WorkflowWithRetry],
            activities=[failing_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            # Execute workflow with failing activity
            result = await temporal_env.client.execute_workflow(
                WorkflowWithRetry.run,
                TaskInput(value=30),
                id=f"test-workflow-retry-{uuid.uuid4()}",
                task_queue="test-queue-3",
            )

            # Should eventually succeed on retry
            assert result == 50  # 30 + 20

            # Get captured spans
            spans = memory_logger.pop()

            # Verify spans were created
            assert len(spans) > 0, "Expected spans to be created"

            # Verify activity spans (should have multiple attempts)
            activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]
            assert len(activity_spans) >= 1, "Expected at least one activity span for retries"

    @pytest.mark.asyncio
    async def test_plugin_child_workflow_tracing(self, temporal_env, memory_logger):
        """Test tracing of child workflows.

        Verifies that child workflows are traced and linked to parent workflows.
        """
        async with Worker(
            temporal_env.client,
            task_queue="test-queue-4",
            workflows=[ParentWorkflow, ChildWorkflow],
            activities=[simple_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            # Execute parent workflow which spawns child
            result = await temporal_env.client.execute_workflow(
                ParentWorkflow.run,
                TaskInput(value=40),
                id=f"test-workflow-parent-{uuid.uuid4()}",
                task_queue="test-queue-4",
            )

            # Result should come from child workflow's activity
            assert result == 50  # 40 + 10

            # Get captured spans
            spans = memory_logger.pop()

            # Verify spans were created
            assert len(spans) > 0, "Expected spans to be created"

            # Verify both parent and child workflow spans
            workflow_spans = [s for s in spans if "temporal.workflow" in s.get("span_attributes", {}).get("name", "")]
            assert len(workflow_spans) >= 2, "Expected at least 2 workflow spans (parent and child)"

            # Verify activity span
            activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]
            assert len(activity_spans) > 0, "Expected activity spans"

    @pytest.mark.asyncio
    async def test_plugin_local_activity_tracing(self, temporal_env, memory_logger):
        """Test that local activities are traced correctly.

        Local activities execute in the same worker process and should
        be traced like regular activities.
        """
        async with Worker(
            temporal_env.client,
            task_queue="test-queue-5",
            workflows=[WorkflowWithLocalActivity],
            activities=[simple_local_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            result = await temporal_env.client.execute_workflow(
                WorkflowWithLocalActivity.run,
                TaskInput(value=100),
                id=f"test-workflow-local-{uuid.uuid4()}",
                task_queue="test-queue-5",
            )

            assert result == 105  # 100 + 5

            # Get captured spans
            spans = memory_logger.pop()

            # Verify spans were created
            assert len(spans) > 0, "Expected spans to be created"

            # Verify local activity span was created
            activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]
            assert len(activity_spans) > 0, "Expected local activity span to be created"

    @pytest.mark.asyncio
    async def test_plugin_client_context_propagation(self, temporal_env, memory_logger):
        """Test that BraintrustPlugin works with Client.connect for context propagation.

        Verifies that:
        1. Plugin can be passed to Client.connect (not just Worker)
        2. Client-side spans are linked to workflow/activity spans via headers
        """
        from temporalio.client import Client

        # Create a NEW client with the plugin (simulates user doing Client.connect with plugin)
        plugin = BraintrustPlugin(logger=memory_logger)
        client = await Client.connect(
            temporal_env.client.service_client.config.target_host,
            namespace=temporal_env.client.namespace,
            plugins=[plugin],
        )

        # Create worker (still needs plugin for worker-side tracing)
        async with Worker(
            client,
            task_queue="test-queue-client-plugin",
            workflows=[TestWorkflow],
            activities=[simple_activity],
            plugins=[BraintrustPlugin(logger=memory_logger)],
        ):
            # Create a parent span at the client level
            with braintrust.start_span(name="test.client_with_plugin", type="task") as parent_span:
                parent_context = parent_span.export()

                # Execute workflow - plugin should inject span context via client interceptor
                result = await client.execute_workflow(
                    TestWorkflow.run,
                    TaskInput(value=25),
                    id=f"test-workflow-client-plugin-{uuid.uuid4()}",
                    task_queue="test-queue-client-plugin",
                )

                assert result == 35  # 25 + 10

        # Get captured spans
        spans = memory_logger.pop()

        # Verify spans were created
        assert len(spans) > 0, "Expected spans to be created"

        # Verify client span exists
        client_spans = [s for s in spans if "test.client_with_plugin" in s.get("span_attributes", {}).get("name", "")]
        assert len(client_spans) > 0, "Expected client span to be created"

        # Verify workflow span was created and linked to client span
        workflow_spans = [s for s in spans if "temporal.workflow" in s.get("span_attributes", {}).get("name", "")]
        assert len(workflow_spans) > 0, "Expected workflow span to be created"

        # Verify activity span was created
        activity_spans = [s for s in spans if "temporal.activity" in s.get("span_attributes", {}).get("name", "")]
        assert len(activity_spans) > 0, "Expected activity span to be created"

        # Verify parent-child relationship: workflow should have client span as parent
        workflow_span = workflow_spans[0]
        client_span = client_spans[0]
        assert workflow_span.get("root_span_id") == client_span.get("root_span_id"), (
            "Workflow span should be in same trace as client span"
        )
