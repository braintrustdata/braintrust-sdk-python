import inspect
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from braintrust import Attachment, SpanCustomizer, SpanExportData, auto_instrument, logger, set_span_customizers
from braintrust.functions.stream import BraintrustJsonChunk, BraintrustStream
from braintrust.span_customizer import _customize_span_export
from braintrust.test_helpers import init_test_exp, with_memory_logger  # noqa: F401 # type: ignore[reportUnusedImport]
from braintrust.trace import LocalTrace
from braintrust.util import LazyValue


@pytest.fixture(autouse=True)
def reset_customizers():
    set_span_customizers(None)
    try:
        yield
    finally:
        set_span_customizers(None)


@pytest.fixture
def test_logger():
    metadata = logger.OrgProjectMetadata(
        org_id="org", project=logger.ObjectMetadata(id="project", name="project", full_info={})
    )
    return logger.Logger(LazyValue(lambda: metadata, use_mutex=False))


def test_order_replacement_and_registration_snapshot():
    observed = []

    class Replace(SpanCustomizer):
        def on_span_export(self, data: SpanExportData) -> SpanExportData:
            # Reconfiguration during export applies only to subsequent records.
            set_span_customizers([])
            return {"output": "redacted"}

    class Observe(SpanCustomizer):
        def on_span_export(self, data: SpanExportData) -> SpanExportData:
            observed.append(dict(data))
            data["metadata"] = {"safe": True}
            return data

    customizers = [Replace(), Observe()]
    set_span_customizers(customizers)
    customizers.clear()
    result = _customize_span_export({"id": "row", "input": "secret", "output": "secret"})
    assert observed == [{"id": "row", "output": "redacted"}]
    assert result == {"id": "row", "output": "redacted", "metadata": {"safe": True}}
    assert _customize_span_export({"output": "next"}) == {"output": "next"}


class _Valid(SpanCustomizer):
    def on_span_export(self, data):
        return {"output": "valid"}


class _Async(SpanCustomizer):
    async def on_span_export(self, data):  # type: ignore[override]
        return data


@pytest.mark.parametrize(
    "invalid",
    [
        _Valid,  # the class, not an instance
        lambda data: data,
        SimpleNamespace(on_span_export="not callable"),
        object(),
        _Async(),
    ],
    ids=["class", "function", "non-callable-hook", "no-hook", "async-hook"],
)
def test_registration_rejects_misconfigured_customizers(invalid):
    valid = _Valid()
    set_span_customizers([valid])
    with pytest.raises(TypeError):
        set_span_customizers([valid, invalid])
    with pytest.raises(TypeError):
        auto_instrument(span_customizers=[invalid])
    # A rejected registration leaves the previous configuration in place.
    assert _customize_span_export({"id": "row", "output": "secret"}) == {"id": "row", "output": "valid"}


def test_duck_typed_customizers_are_accepted():
    class DuckTyped:
        def on_span_export(self, data):
            return {"output": "duck"}

    set_span_customizers([DuckTyped()])  # type: ignore[list-item]
    assert _customize_span_export({"id": "row", "output": "secret"}) == {"id": "row", "output": "duck"}


def test_explicit_customizers_override_without_changing_global_configuration():
    class Global(SpanCustomizer):
        def on_span_export(self, data):
            return {"output": "global"}

    class Local(SpanCustomizer):
        def on_span_export(self, data):
            return {"output": "local"}

    set_span_customizers([Global()])
    record = {"id": "row", "output": "original"}
    assert _customize_span_export(record, [Local()]) == {"id": "row", "output": "local"}
    assert _customize_span_export(record, []) == record
    assert _customize_span_export(record) == {"id": "row", "output": "global"}


def test_protected_protocol_fields_restored_between_hooks():
    protected = {
        "id": "row",
        "span_id": "span",
        "root_span_id": "root",
        "span_parents": ["parent"],
        "org_id": "org",
        "project_id": "project",
        "log_id": "g",
        "function_data": {"nested": ["routing"]},
        "_is_merge": True,
        "_merge_paths": [["metadata", "nested"]],
        "_parent_id": "parent-row",
        "_object_delete": False,
        "_array_delete": [["tags", "private"]],
        "_xact_id": "transaction",
    }
    original = deepcopy(protected)
    observed = []

    class Corrupt(SpanCustomizer):
        def on_span_export(self, data: SpanExportData) -> SpanExportData:
            data["span_parents"].append("wrong")
            data["_merge_paths"][0].append("wrong")
            data["_array_delete"][0].clear()
            data["function_data"]["nested"].clear()
            for key in protected:
                data[key] = "wrong"
            data["experiment_id"] = "wrong-destination"
            data["dataset_id"] = "wrong-destination"
            data["prompt_session_id"] = "wrong-destination"
            return data

    class Observe(SpanCustomizer):
        def on_span_export(self, data: SpanExportData) -> SpanExportData:
            observed.append(deepcopy(data))
            # A second mutation must not poison the stored snapshot either.
            data["_merge_paths"][0].clear()
            data["span_parents"].clear()
            return {"output": "safe"}

    set_span_customizers([Corrupt(), Observe()])
    result = _customize_span_export(protected)
    assert observed == [original]
    assert result == {**original, "output": "safe"}
    assert protected == original


@pytest.mark.parametrize("failure", ["exception", "none", "list", "coroutine", "awaitable"])
def test_fail_open_keeps_mutations_and_continues(failure):
    observed = []
    coroutines = []

    class AwaitableRecord(dict):
        def __await__(self):
            raise AssertionError("Hooks must not be awaited")
            yield

    async def asynchronous_result():
        raise AssertionError("Async hook must not execute")

    class Broken(SpanCustomizer):
        def on_span_export(self, data):
            data["output"] = "redacted"
            data["id"] = "wrong"
            if failure == "exception":
                raise ValueError("private exception text")
            if failure == "none":
                return None
            if failure == "list":
                return []
            if failure == "awaitable":
                return AwaitableRecord(output="wrong")
            coroutine = asynchronous_result()
            coroutines.append(coroutine)
            return coroutine

    class Next(SpanCustomizer):
        def on_span_export(self, data):
            observed.append(dict(data))
            return data

    set_span_customizers([Broken(), Next()])
    result = _customize_span_export({"id": "row", "output": "secret"})
    assert observed == [{"id": "row", "output": "redacted"}]
    assert result == observed[0]
    assert all(inspect.getcoroutinestate(c) == inspect.CORO_CLOSED for c in coroutines)


def test_native_scope_incremental_redaction_and_provider_isolation(with_memory_logger, test_logger):
    seen = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            seen.append((data["id"], set(data)))
            for key in ("input", "output"):
                if key in data:
                    data[key]["secret"] = "redacted"
            data.pop("error", None)
            return data

    set_span_customizers([Redact()])
    application_input = {"secret": "input"}
    with test_logger.start_span(name="manual", input=application_input) as manual:
        with manual.start_span(
            name="instrumented", input=application_input, internal={"instrumentation": "test-auto"}
        ) as span:
            # Flush before completion to expose the incremental lifecycle.
            first = with_memory_logger.pop()
            assert next(row for row in first if row["id"] == span.id)["input"] == {"secret": "redacted"}
            assert next(row for row in first if row["id"] == manual.id)["input"] == {"secret": "input"}
            stream = BraintrustStream([BraintrustJsonChunk(data='{"secret":"output"}')])
            span.log(output=stream, error="private error")
            span.set_attributes(name="renamed")
            with span.start_span(name="manual child", input=application_input) as manual_child:
                pass
        span.log_feedback(expected="feedback value")

    metadata = logger.ProjectDatasetMetadata(
        project=logger.ObjectMetadata(id="project", name="project", full_info={}),
        dataset=logger.ObjectMetadata(id="dataset", name="dataset", full_info={}),
    )
    dataset = logger.Dataset(LazyValue(lambda: metadata, use_mutex=False), legacy=False)
    dataset_id = dataset.insert(input={"secret": "dataset"})
    rows = with_memory_logger.pop()
    exported = next(row for row in rows if row["id"] == span.id)
    assert exported["output"] == {"secret": "redacted"}
    assert "error" not in exported
    assert exported["expected"] == "feedback value"
    assert exported["span_attributes"]["name"] == "renamed"
    assert next(row for row in rows if row["id"] == manual_child.id)["input"] == application_input
    assert next(row for row in rows if row["id"] == dataset_id)["input"] == {"secret": "dataset"}
    assert application_input == {"secret": "input"}
    assert stream.final_value() == {"secret": "output"}
    assert [row_id for row_id, _ in seen] == [span.id] * 4
    assert ["input" in keys for _, keys in seen] == [True, False, False, False]
    assert ["output" in keys for _, keys in seen] == [False, True, False, False]
    assert not any("expected" in keys for _, keys in seen)


def test_customization_precedes_attachments_masking_and_reuses_records_on_retry(
    monkeypatch, with_memory_logger, test_logger
):
    attachment = Attachment(data=b"private", filename="private.txt", content_type="text/plain")
    seen_attachments = []
    invocations = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            invocations.append(data["id"])
            if "input" in data:
                seen_attachments.append(data["input"])
                data["input"] = "redacted"
            if "output" in data:
                data["output"] = "redacted output"
            return data

    set_span_customizers([Redact()])
    span = test_logger.start_span(input=attachment, internal={"instrumentation": "test-auto"})
    span.log(output="private output")
    span.end()
    pending = list(with_memory_logger.logs)
    with_memory_logger.logs.clear()

    # A later lazy record fails once, after earlier records have been customized.
    attempts = 0

    def resolve_later_record():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry lazy resolution")
        return {"id": "other", "project_id": "project", "log_id": "g", "output": "other"}

    pending.append(LazyValue(resolve_later_record, use_mutex=False))
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    monkeypatch.setattr(logger.time, "sleep", lambda _: None)
    connection = MagicMock()
    payloads = []

    def send(_path, *, data):
        payloads.append(data)
        if len(payloads) == 1:
            raise ConnectionError("retry transport")
        return SimpleNamespace(ok=True)

    connection.post.side_effect = send
    background = logger._HTTPBackgroundLogger(LazyValue(lambda: connection, use_mutex=False))
    background.num_tries = 2
    background.sync_flush = True
    masked = []

    def mask(value):
        masked.append(value)
        return value

    background.set_masking_function(mask)
    background._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
    for item in pending:
        background.queue.put(item)
    background.flush()

    assert attempts == 2
    assert invocations == [span.id] * 3
    assert seen_attachments == [attachment]
    assert seen_attachments[0] is attachment
    assert "redacted" in masked
    assert "redacted output" in masked
    assert payloads[0] == payloads[1]
    exported = next(row for row in json.loads(payloads[1])["rows"] if row["id"] == span.id)
    assert exported["input"] == "redacted"
    assert exported["output"] == "redacted output"


@pytest.mark.parametrize("backend", ["memory", "http"])
def test_masking_remains_logger_local_and_runs_on_merged_manual_records(monkeypatch, backend):
    class InstrumentationOnly(SpanCustomizer):
        def on_span_export(self, data):
            return {"output": "must not run on manual records"}

    set_span_customizers([InstrumentationOnly()])
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    background = (
        logger._MemoryBackgroundLogger()
        if backend == "memory"
        else logger._HTTPBackgroundLogger(LazyValue(lambda: MagicMock(), use_mutex=False))
    )
    other_background = logger._MemoryBackgroundLogger()

    def export(target):
        records = [
            {"id": "row", "project_id": "project", "input": None, "metadata": {"secret": "private"}},
            {"id": "row", "project_id": "project", "_is_merge": True, "metadata": {"redact": True}},
        ]
        pending = [LazyValue(lambda record=record: record, use_mutex=False) for record in records]
        if isinstance(target, logger._MemoryBackgroundLogger):
            target.log(*pending)
            return target.pop()[0]
        rows, _ = target._unwrap_lazy_values(pending)
        return rows[0]

    def mask(value):
        if value is None:
            return "masked null"
        if isinstance(value, dict) and value.get("redact"):
            return {"secret": "redacted"}
        return value

    background.set_masking_function(mask)
    masked = export(background)
    assert masked["metadata"] == {"secret": "redacted"}
    assert masked["input"] == "masked null"
    assert "output" not in masked
    assert export(other_background)["metadata"] == {"secret": "private", "redact": True}

    background.set_masking_function(lambda _: "replacement")
    assert export(background)["input"] == "replacement"
    background.set_masking_function(None)
    unmasked = export(background)
    assert unmasked["metadata"] == {"secret": "private", "redact": True}
    assert unmasked["input"] is None
    assert "output" not in unmasked


def test_auto_instrument_registration_and_disable(monkeypatch, with_memory_logger, test_logger):
    # Registration should work without importing optional provider libraries.
    monkeypatch.setattr("braintrust.auto._instrument_integration", lambda _: False)

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = "redacted"
            return data

    auto_instrument(span_customizers=[Redact()])
    auto_instrument()  # Omitted configuration must not reset customizers.
    with test_logger.start_span(input="private", internal={"instrumentation": "test-auto"}):
        pass
    assert with_memory_logger.pop()[0]["input"] == "redacted"
    auto_instrument(span_customizers=[])
    with test_logger.start_span(input="untouched", internal={"instrumentation": "test-auto"}):
        pass
    assert with_memory_logger.pop()[0]["input"] == "untouched"


def _cached(experiment, span):
    return {cached.span_id: cached for cached in experiment.state.span_cache.get_by_root_span_id(span.root_span_id)}


@pytest.fixture
def span_cache_experiment(with_memory_logger):
    experiment = init_test_exp("span-customizer-cache")
    experiment.state.span_cache.start()
    try:
        yield experiment
    finally:
        experiment.state.span_cache.stop()
        experiment.state.span_cache.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("resolve_initial", [False, True], ids=["pending-span", "pending-output"])
async def test_trace_reads_customized_content_before_export(
    with_memory_logger, span_cache_experiment, resolve_initial
):
    customized_outputs = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            # Nested in-place edits and replacement must both reach the cache.
            if "metadata" in data:
                data["metadata"]["secret"] = "redacted"
            if "output" in data:
                customized_outputs.append(data["output"])
                data["output"] = "redacted"
            return data

    set_span_customizers([Redact()])
    experiment = span_cache_experiment
    with experiment.start_span(name="manual", metadata={"secret": "manual"}) as manual:
        with manual.start_span(
            name="instrumented", metadata={"secret": "private"}, internal={"instrumentation": "test-auto"}
        ) as span:
            if resolve_initial:
                with_memory_logger.pop()
            span.log(output="private output")

    async def unexpected_flush():
        pytest.fail("Local trace reads must not require a backend flush")

    trace = LocalTrace("experiment", experiment.id, manual.root_span_id, unexpected_flush, experiment.state)
    cached = {record.span_id: record for record in await trace.get_spans()}
    assert cached[span.span_id].metadata == {"secret": "redacted"}
    assert cached[span.span_id].output == "redacted"
    assert cached[manual.span_id].metadata == {"secret": "manual"}
    assert customized_outputs == ["private output"]

    # Export must reuse the transformation performed for the local trace read.
    rows = with_memory_logger.pop()
    assert customized_outputs == ["private output"]
    exported = next(row for row in rows if row["id"] == span.id)
    if not resolve_initial:
        assert exported["metadata"] == {"secret": "redacted"}
    assert exported["output"] == "redacted"


def test_trace_cache_waits_for_inflight_customization(with_memory_logger, span_cache_experiment):
    entered = Event()
    release = Event()
    customized_outputs = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "output" in data:
                customized_outputs.append(data["output"])
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Customization was not released")
                data["output"] = "redacted"
            return data

    set_span_customizers([Redact()])
    experiment = span_cache_experiment
    with experiment.start_span(name="manual") as manual:
        with manual.start_span(name="instrumented", internal={"instrumentation": "test-auto"}) as span:
            span.log(output="private output")

    with ThreadPoolExecutor(max_workers=2) as executor:
        export = executor.submit(with_memory_logger.pop)
        try:
            assert entered.wait(timeout=5)
            read = executor.submit(_cached, experiment, span)
            # An incomplete cache must not escape while the publisher is in a hook.
            with pytest.raises(TimeoutError):
                read.result(timeout=0.1)
        finally:
            release.set()
        cached = read.result(timeout=5)
        rows = export.result(timeout=5)

    assert cached[span.span_id].output == "redacted"
    assert next(row for row in rows if row["id"] == span.id)["output"] == "redacted"
    assert customized_outputs == ["private output"]


def test_span_cache_written_eagerly_without_customizers(with_memory_logger, span_cache_experiment):
    experiment = span_cache_experiment
    with experiment.start_span(name="instrumented", internal={"instrumentation": "test-auto"}) as span:
        span.log(output="output")
    assert _cached(experiment, span)[span.span_id].output == "output"
