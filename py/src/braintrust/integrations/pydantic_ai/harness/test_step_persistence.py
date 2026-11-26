from importlib import import_module

import pytest
from braintrust import logger
from braintrust.integrations.pydantic_ai._test_harness import make_harness_agent
from braintrust.test_helpers import find_span_by_name
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_loader", "native_instrumentation"),
    [("continue_run", False), ("fork_run", False), ("fork_run", True)],
)
async def test_step_persistence_lineage(harness_spans, history_loader, native_instrumentation, monkeypatch):
    persistence = import_module("pydantic_ai_harness.step_persistence")
    capabilities = []
    if native_instrumentation:
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "1")
        monkeypatch.setattr(logger._internal_get_global_state(), "_context_manager", None)
        capabilities.append(Instrumentation(settings=InstrumentationSettings(tracer_provider=TracerProvider())))
    store = persistence.InMemoryStepStore()
    source = make_harness_agent("source", capabilities=[*capabilities, persistence.StepPersistence(store=store)])
    branch = make_harness_agent(
        "branch",
        capabilities=[*capabilities, persistence.StepPersistence(store=store, parent_run_id="source-run")],
    )

    await source.run("start", run_id="source-run")
    history = await getattr(persistence, history_loader)(store, run_id="source-run")
    await branch.run("next", run_id="branch-run", message_history=history)
    spans = harness_spans.pop()

    source_record = await store.get_run(run_id="source-run")
    branch_record = await store.get_run(run_id="branch-run")
    source_span = find_span_by_name(spans, "agent_run [source]")
    branch_span = find_span_by_name(spans, "agent_run [branch]")
    assert source_span["metadata"]["step_persistence"] == {"run_id": source_record.run_id}
    assert branch_span["metadata"]["step_persistence"] == {
        "run_id": branch_record.run_id,
        "parent_run_id": branch_record.parent_run_id,
    }
    assert not source_span.get("span_parents")
    assert not branch_span.get("span_parents")
    assert source_span["root_span_id"] != branch_span["root_span_id"]
