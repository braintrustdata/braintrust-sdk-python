"""Tests for the experimental durable eval API."""

import asyncio
from unittest.mock import patch

import pytest

from . import durable_eval as durable_eval_module
from .durable_eval import (
    BatchCompletionPoll,
    BatchCompletionWebhook,
    BatchPollResult,
    BatchScorer,
    BatchScorerResult,
    BatchTask,
    BatchTaskResult,
    DurableEvalCompletedResult,
    DurableEvalFailedResult,
    DurableEvalMemoryStore,
    DurableEvalRedisStore,
    DurableEvalWaitingResult,
    define_durable_eval,
)
from .logger import BraintrustState, Dataset, ObjectMetadata, ProjectDatasetMetadata
from .test_helpers import init_test_exp, with_memory_logger, with_simulate_login  # noqa: F401
from .util import LazyValue


@pytest.mark.asyncio
async def test_memory_store_is_atomic_and_copies_values():
    store = DurableEvalMemoryStore()
    value = bytearray(b"first")

    first = await store.get_or_set("key", value)
    value[:] = b"other"
    second = await store.get_or_set("key", b"second")

    assert first.created is True
    assert first.value == b"first"
    assert second.created is False
    assert second.value == b"first"
    assert await store.read("key") == b"first"


class _SyncRedis:
    def __init__(self):
        self.values = {}
        self.calls = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        self.calls.append((key, kwargs))
        old = self.values.get(key)
        if kwargs.get("nx") and old is not None:
            return old if kwargs.get("get") else None
        self.values[key] = value
        return old if kwargs.get("get") else True


class _AsyncRedis(_SyncRedis):
    async def get(self, key):
        return super().get(key)

    async def set(self, key, value, **kwargs):
        return super().set(key, value, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [_SyncRedis, _AsyncRedis])
async def test_redis_store_supports_sync_and_async_redis_py(client_type):
    client = client_type()
    store = DurableEvalRedisStore(client, key_prefix="test:", ttl_ms=1234)

    await store.write("one", b"value")
    assert await store.read("one") == b"value"
    assert (await store.get_or_set("two", b"first")).created is True
    existing = await store.get_or_set("two", b"second")

    assert existing.created is False
    assert existing.value == b"first"
    assert all(key.startswith("test:") for key, _ in client.calls)
    assert all(options["px"] == 1234 for _, options in client.calls)


@pytest.mark.asyncio
async def test_ordinary_durable_eval_completes_locally_and_is_idempotent():
    task_calls = []
    score_calls = []

    def task(input, hooks):
        task_calls.append((input, hooks.trial_index))
        hooks.metadata["task"] = True
        hooks.tags = ["updated"]
        return input * 2

    def scorer(output, expected, metadata, tags):
        score_calls.append((output, metadata, tags))
        return output == expected

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 2, "expected": 4, "metadata": {"source": "test"}}],
        task=task,
        scores=[scorer],
        trial_count=2,
    )

    result = await durable_eval.start(no_send_logs=True)
    repeated = await durable_eval.status(result.run_id)

    assert isinstance(result, DurableEvalCompletedResult)
    assert isinstance(repeated, DurableEvalCompletedResult)
    assert result.summary.scores["scorer"].score == 1
    assert task_calls == [(2, 0), (2, 1)]
    assert score_calls == [
        (4, {"source": "test", "task": True}, ["updated"]),
        (4, {"source": "test", "task": True}, ["updated"]),
    ]


@pytest.mark.asyncio
async def test_ordinary_only_runs_do_not_require_case_ids_and_get_new_run_ids():
    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"input": 1}],
        task=lambda value: value,
        scores=[lambda: 1],
    )

    first = await durable_eval.start(no_send_logs=True)
    second = await durable_eval.start(no_send_logs=True)

    assert first.status == "completed"
    assert second.status == "completed"
    assert first.run_id != second.run_id


@pytest.mark.asyncio
async def test_poll_advances_task_then_mixed_scorers_without_polling_new_batches():
    submitted = {}
    ready = set()
    calls = []

    async def task_submit(items, context):
        calls.append(("submit-task", context.batch_id))
        submitted[context.batch_id] = items
        return {"provider_id": context.batch_id}

    async def task_collect(_submission, context):
        calls.append(("collect-task", context.batch_id))
        return [BatchTaskResult(id=item.id, output=item.input * 2) for item in submitted[context.batch_id]]

    async def score_submit(items, context):
        calls.append(("submit-score", context.batch_id))
        submitted[context.batch_id] = items
        return {"provider_id": context.batch_id}

    async def score_collect(_submission, context):
        calls.append(("collect-score", context.batch_id))
        return [
            BatchScorerResult(id=item.id, score=item.output == item.expected) for item in submitted[context.batch_id]
        ]

    async def poll(_submission, context):
        calls.append(("poll", context.batch_id))
        return BatchPollResult("complete" if context.batch_id in ready else "pending")

    task = BatchTask(
        submit=task_submit,
        completion=BatchCompletionPoll(poll),
        collect=task_collect,
        batch_size=2,
    )
    batch_score = BatchScorer(
        name="batch",
        submit=score_submit,
        completion=BatchCompletionPoll(poll),
        collect=score_collect,
        batch_size=2,
    )
    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": str(i), "input": i, "expected": i * 2} for i in range(3)],
        task=task,
        scores=[lambda output, expected: output == expected, batch_score],
    )

    started = await durable_eval.start(no_send_logs=True)
    assert isinstance(started, DurableEvalWaitingResult)
    assert started.pending.poll == 2
    task_batch_ids = set(submitted)
    ready.update(task_batch_ids)

    after_tasks = await durable_eval.poll(started.run_id)
    assert isinstance(after_tasks, DurableEvalWaitingResult)
    assert after_tasks.pending.poll == 2
    score_batch_ids = set(submitted) - task_batch_ids
    assert score_batch_ids
    assert not any(call == ("poll", batch_id) for batch_id in score_batch_ids for call in calls)

    ready.update(score_batch_ids)
    completed = await durable_eval.poll(started.run_id)
    assert isinstance(completed, DurableEvalCompletedResult)
    assert completed.summary.scores["scorer_0"].score == 1
    assert completed.summary.scores["batch"].score == 1
    calls_before_status = list(calls)
    assert (await durable_eval.status(started.run_id)).status == "completed"
    assert calls == calls_before_status


@pytest.mark.asyncio
async def test_failed_poll_is_persisted_as_a_terminal_result():
    poll_calls = 0

    async def poll(_submission, _context):
        nonlocal poll_calls
        poll_calls += 1
        return BatchPollResult("failed", error={"code": "provider_failed"})

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 1}],
        task=BatchTask(
            submit=lambda _items, context: {"id": context.batch_id},
            completion=BatchCompletionPoll(poll),
            collect=lambda _submission, _context: pytest.fail("failed batches must not be collected"),
        ),
    )

    started = await durable_eval.start(no_send_logs=True)
    failed = await durable_eval.poll(started.run_id)
    repeated = await durable_eval.status(started.run_id)

    assert isinstance(failed, DurableEvalFailedResult)
    assert repeated == failed
    assert failed.error == {"code": "provider_failed"}
    assert failed.pending.poll == 0
    assert failed.pending.webhook == 0
    assert poll_calls == 1


@pytest.mark.asyncio
async def test_webhook_result_can_be_matched_by_external_id():
    submitted = {}

    async def submit(items, context):
        submitted[context.batch_id] = items
        return {"id": f"external-{context.batch_id}"}

    async def collect(_submission, context):
        return [BatchTaskResult(id=item.id, output=item.input) for item in submitted[context.batch_id]]

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": "ok"}],
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=collect,
        ),
    )
    started = await durable_eval.start(no_send_logs=True)
    batch_id = next(iter(submitted))

    completed = await durable_eval.process_batch_result(started.run_id, external_id=f"external-{batch_id}")

    assert isinstance(completed, DurableEvalCompletedResult)
    assert (await durable_eval.process_batch_result(started.run_id, batch_id=batch_id)).status == "completed"


@pytest.mark.asyncio
async def test_collect_requires_exactly_one_result_per_item():
    batch_ids = []

    async def submit(_items, context):
        batch_ids.append(context.batch_id)
        return {"id": context.batch_id}

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "a", "input": 1}, {"id": "b", "input": 2}],
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=lambda _submission, _context: [BatchTaskResult(id="a:trial:0", output=1)],
        ),
    )
    started = await durable_eval.start(no_send_logs=True)

    with pytest.raises(ValueError, match="missing=.*b:trial:0"):
        await durable_eval.process_batch_result(started.run_id, batch_id=batch_ids[0])


@pytest.mark.asyncio
async def test_case_ids_must_be_stable_and_unique():
    batch_task = BatchTask(
        submit=lambda _items, _context: {"id": "unused"},
        completion=BatchCompletionPoll(lambda _submission, _context: BatchPollResult("pending")),
        collect=lambda _submission, _context: [],
    )
    missing = define_durable_eval("project", store=DurableEvalMemoryStore(), data=[{"input": 1}], task=batch_task)
    with pytest.raises(ValueError, match="non-empty id"):
        await missing.start(no_send_logs=True)

    duplicate = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "same", "input": 1}, {"id": "same", "input": 2}],
        task=batch_task,
    )
    with pytest.raises(ValueError, match="duplicate"):
        await duplicate.start(no_send_logs=True)


@pytest.mark.asyncio
async def test_case_persistence_does_not_deepcopy_inputs():
    class SerializableWithoutDeepcopy:
        def __deepcopy__(self, _memo):
            raise AssertionError("input was deep-copied")

        def model_dump(self, **_kwargs):
            return {"value": "serialized"}

    submitted = []

    async def submit(items, _context):
        submitted.extend(items)
        return {"id": "pending"}

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": SerializableWithoutDeepcopy()}],
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionPoll(lambda _submission, _context: BatchPollResult("pending")),
            collect=lambda _submission, _context: [],
        ),
    )

    result = await durable_eval.start(no_send_logs=True)

    assert result.status == "waiting"
    assert submitted[0].input == {"value": "serialized"}


@pytest.mark.asyncio
async def test_concurrent_webhooks_claim_downstream_submission_once():
    task_items = []
    collects_started = 0
    both_collecting = asyncio.Event()
    score_submissions = 0

    async def submit_task(items, _context):
        task_items.extend(items)
        return {"id": "task-provider"}

    async def collect_task(_submission, _context):
        nonlocal collects_started
        collects_started += 1
        if collects_started == 2:
            both_collecting.set()
        await both_collecting.wait()
        return [BatchTaskResult(id=item.id, output=item.input * 2) for item in task_items]

    async def submit_score(_items, _context):
        nonlocal score_submissions
        score_submissions += 1
        return {"id": "score-provider"}

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 2, "expected": 4}],
        task=BatchTask(
            submit=submit_task,
            completion=BatchCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=collect_task,
        ),
        scores=[
            BatchScorer(
                name="exact",
                submit=submit_score,
                completion=BatchCompletionWebhook(lambda submission, _context: submission["id"]),
                collect=lambda _submission, _context: [],
            )
        ],
    )
    started = await durable_eval.start(no_send_logs=True)

    await asyncio.gather(
        durable_eval.process_batch_result(started.run_id, external_id="task-provider"),
        durable_eval.process_batch_result(started.run_id, external_id="task-provider"),
    )

    assert score_submissions == 1
    current = await durable_eval.status(started.run_id)
    assert current.status == "waiting"
    assert current.pending.webhook == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_ids", [False, True])
async def test_durable_logging_uses_stable_spans_and_resume_metadata(
    monkeypatch, with_memory_logger, with_simulate_login, legacy_ids
):
    if legacy_ids:
        monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "true")
    else:
        monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
    local_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 1, "expected": 2}],
        task=lambda value: value + 1,
        scores=[lambda output, expected: output == expected],
    )
    local_summary = (await local_eval.start(no_send_logs=True)).summary
    experiment = init_test_exp("durable", "project")
    monkeypatch.setattr(durable_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)
    submitted = []
    trace_configurations = []
    classifier_trace_configurations = []

    async def submit(items, _context):
        submitted.extend(items)
        return {"id": "task"}

    def scorer(output, expected, trace):
        trace_configurations.append(trace.get_configuration())
        return output == expected

    def classifier(output, trace):
        classifier_trace_configurations.append(trace.get_configuration())
        return {"id": "positive"}

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 1, "expected": 2}],
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionPoll(lambda _submission, _context: BatchPollResult("complete")),
            collect=lambda _submission, _context: [
                BatchTaskResult(id=item.id, output=item.input + 1) for item in submitted
            ],
        ),
        scores=[scorer],
        classifiers=[classifier],
        experiment_name="durable",
    )

    waiting = await durable_eval.start()
    result = await durable_eval.poll(waiting.run_id)
    logs = with_memory_logger.pop()

    assert result.status == "completed"
    assert len(logs) == 4
    roots = [row for row in logs if not row["span_parents"]]
    assert len(roots) == 1
    assert roots[0]["metadata"]["durable_eval"] == {
        "run_id": result.run_id,
        "case_id": "case",
        "trial_index": 0,
    }
    assert trace_configurations == [
        {
            "object_type": "experiment",
            "object_id": experiment.id,
            "root_span_id": roots[0]["root_span_id"],
        }
    ]
    assert classifier_trace_configurations == trace_configurations
    assert len({row["span_id"] for row in logs}) == 4
    await durable_eval.status(result.run_id)
    assert with_memory_logger.pop() == []


@pytest.mark.asyncio
async def test_durable_logging_flushes_before_persisting_log_markers(
    monkeypatch, with_memory_logger, with_simulate_login
):
    local_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=[{"id": "case", "input": 1}],
        task=lambda value: value,
    )
    local_summary = (await local_eval.start(no_send_logs=True)).summary
    experiment = init_test_exp("durable", "project")
    monkeypatch.setattr(durable_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)

    flush_count = 0
    marker_flush_counts = []
    original_flush = with_memory_logger.flush

    def flush(*args, **kwargs):
        nonlocal flush_count
        flush_count += 1
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(with_memory_logger, "flush", flush)

    class RecordingStore(DurableEvalMemoryStore):
        async def write(self, key, value):
            if "/task-log/" in key or "/score-log-" in key or "/classification-log-" in key:
                marker_flush_counts.append(flush_count)
            await super().write(key, value)

    submitted = []

    async def submit(items, _context):
        submitted.extend(items)
        return {"id": "task"}

    durable_eval = define_durable_eval(
        "project",
        store=RecordingStore(),
        data=[{"id": "case", "input": 1, "expected": 1}],
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionPoll(lambda _submission, _context: BatchPollResult("complete")),
            collect=lambda _submission, _context: [
                BatchTaskResult(id=item.id, output=item.input) for item in submitted
            ],
        ),
        scores=[lambda output, expected: output == expected],
        classifiers=[lambda output: {"id": "positive"}],
        experiment_name="durable",
    )

    waiting = await durable_eval.start()
    completed = await durable_eval.poll(waiting.run_id)

    assert completed.status == "completed"
    assert len(marker_flush_counts) == 3
    assert all(count > 0 for count in marker_flush_counts)


@pytest.mark.asyncio
async def test_durable_dataset_rows_preserve_dataset_origin(monkeypatch, with_memory_logger, with_simulate_login):
    project_metadata = ObjectMetadata(id="test-project", name="test-project", full_info={})
    dataset_metadata = ObjectMetadata(id="active-dataset", name="test-dataset", full_info={})
    dataset = Dataset(
        lazy_metadata=LazyValue(
            lambda: ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata),
            use_mutex=False,
        ),
        state=BraintrustState(),
    )
    row = {
        "id": "dataset-row",
        "_xact_id": "dataset-xact",
        "created": "2026-06-02T00:00:00.000Z",
        "input": 1,
    }
    local_summary = (
        await define_durable_eval(
            "project", store=DurableEvalMemoryStore(), data=[{"input": 1}], task=lambda value: value
        ).start(no_send_logs=True)
    ).summary
    experiment = init_test_exp("durable", "project")
    monkeypatch.setattr(durable_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)

    submitted = []

    async def submit(items, _context):
        submitted.extend(items)
        return {"id": "task"}

    durable_eval = define_durable_eval(
        "project",
        store=DurableEvalMemoryStore(),
        data=dataset,
        task=BatchTask(
            submit=submit,
            completion=BatchCompletionPoll(lambda _submission, _context: BatchPollResult("complete")),
            collect=lambda _submission, _context: [
                BatchTaskResult(id=item.id, output=item.input) for item in submitted
            ],
        ),
        experiment_name="durable",
    )

    with patch.object(dataset, "_refetch", return_value=[row]):
        waiting = await durable_eval.start()
    await durable_eval.poll(waiting.run_id)

    root = next(log for log in with_memory_logger.pop() if not log["span_parents"])
    assert root["origin"] == {
        "object_type": "dataset",
        "object_id": "active-dataset",
        "id": "dataset-row",
        "_xact_id": "dataset-xact",
        "created": "2026-06-02T00:00:00.000Z",
    }
