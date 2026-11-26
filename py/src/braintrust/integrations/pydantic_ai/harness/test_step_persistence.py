from importlib import import_module

import pytest
from braintrust import logger, setup_pydantic_ai
from braintrust.test_helpers import find_span_by_name, init_test_logger
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_loader", "native_instrumentation", "entrypoint"),
    [
        pytest.param("continue_run", False, "run", id="continue_run-False"),
        pytest.param("fork_run", False, "run", id="fork_run-False"),
        pytest.param("fork_run", True, "run", id="fork_run-True"),
        pytest.param("fork_run", False, "run_stream", id="fork_run-run_stream"),
    ],
)
async def test_step_persistence_lineage(
    with_memory_logger, history_loader, native_instrumentation, entrypoint, monkeypatch
):
    persistence = import_module("pydantic_ai_harness.step_persistence")
    capabilities = []
    if native_instrumentation:
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "1")
        monkeypatch.setattr(logger._internal_get_global_state(), "_context_manager", None)
        capabilities.append(Instrumentation(settings=InstrumentationSettings(tracer_provider=TracerProvider())))
    project = "test-pydantic-ai-harness-lineage"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)

    store = persistence.InMemoryStepStore()
    model = "openai:gpt-4o-mini"
    settings = ModelSettings(max_tokens=20)
    source = Agent(
        model,
        name="source",
        model_settings=settings,
        capabilities=[*capabilities, persistence.StepPersistence(store=store)],
    )
    branch = Agent(
        model,
        name="branch",
        model_settings=settings,
        capabilities=[*capabilities, persistence.StepPersistence(store=store, parent_run_id="source-run")],
    )

    await source.run("Reply with the word ready.", run_id="source-run")
    history = await getattr(persistence, history_loader)(store, run_id="source-run")
    prompt = "Repeat your previous answer."
    if entrypoint == "run_stream":
        async with branch.run_stream(prompt, run_id="branch-run", message_history=history) as result:
            output = "".join([chunk async for chunk in result.stream_text(delta=True)])
    else:
        output = (await branch.run(prompt, run_id="branch-run", message_history=history)).output
    spans = with_memory_logger.pop()

    source_record = await store.get_run(run_id="source-run")
    branch_record = await store.get_run(run_id="branch-run")
    source_span = find_span_by_name(spans, "agent_run [source]")
    branch_span = find_span_by_name(spans, f"agent_{entrypoint} [branch]")
    assert branch_span["span_attributes"]["type"] == "task"
    assert branch_span["output"]["response"]["parts"][0]["content"] == output
    if entrypoint == "run_stream":
        assert branch_span["metrics"]["time_to_first_token"] >= 0
    assert source_span["metadata"]["step_persistence"] == {"run_id": source_record.run_id}
    assert branch_span["metadata"]["step_persistence"] == {
        "run_id": branch_record.run_id,
        "parent_run_id": branch_record.parent_run_id,
    }
    assert not source_span.get("span_parents")
    assert not branch_span.get("span_parents")
    assert source_span["root_span_id"] != branch_span["root_span_id"]
