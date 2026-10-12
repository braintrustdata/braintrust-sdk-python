import asyncio
import inspect
import json
import logging
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import exceptiongroup
import pytest
from braintrust import Attachment, SpanCustomizer, SpanExportData, auto_instrument, logger, set_span_customizers
from braintrust.functions.stream import BraintrustJsonChunk, BraintrustStream
from braintrust.integrations.utils import _resolved_attachment_from_bytes
from braintrust.span_customizer import _customize_span_export, _StrippedSpanExportData
from braintrust.test_helpers import init_test_exp, with_memory_logger  # noqa: F401 # type: ignore[reportUnusedImport]
from braintrust.trace import LocalTrace
from braintrust.util import LazyValue


@pytest.fixture(autouse=True)
def reset_customizers(monkeypatch):
    monkeypatch.setattr("braintrust.span_customizer._failure_logged", False)
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


def test_registration_rejected_in_otel_compat_mode(monkeypatch, caplog):
    # Keep auto-instrumentation independent of optional provider libraries.
    monkeypatch.setattr("braintrust.auto._instrument_integration", lambda _module, _class: False)
    valid = _Valid()
    set_span_customizers([valid])
    monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        # A different hook makes replacement of the previous registration observable.
        set_span_customizers([SpanCustomizer()])
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert _customize_span_export({"output": "secret"}) == {"output": "valid"}

        auto_instrument(span_customizers=[SpanCustomizer()])
        assert len(caplog.records) == 2
        assert caplog.records[1].levelno == logging.ERROR
        # Record processing must neither replace the previous hooks nor repeat the diagnostic.
        for output in ("first", "second"):
            assert _customize_span_export({"output": output}) == {"output": "valid"}
        assert len(caplog.records) == 2

        # Both clearing forms remain silent, including through auto_instrument.
        auto_instrument(span_customizers=[])
        assert _customize_span_export({"output": "secret"}) == {"output": "secret"}
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "false")
        set_span_customizers([valid])
        assert _customize_span_export({"output": "secret"}) == {"output": "valid"}
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
        set_span_customizers(None)
        assert _customize_span_export({"output": "secret"}) == {"output": "secret"}
        assert len(caplog.records) == 2


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
def test_fail_closed_strips_record_stops_hooks_and_logs_safely(failure, caplog):
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
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        result = _customize_span_export({"id": "private row", "output": "private payload"})
    assert observed == []
    assert result == {"id": "private row", "context": {"customizer_error": True}}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert caplog.records[0].exc_info is None
    assert "private" not in caplog.text
    assert all(inspect.getcoroutinestate(c) == inspect.CORO_CLOSED for c in coroutines)


@pytest.mark.parametrize("timing", [{}, {"start": 12}, {"end": 15}, {"start": 12, "end": 15}])
def test_failures_preserve_only_original_identity_protocol_and_timing(timing):
    protected = {
        "id": "row",
        "span_id": "span",
        "root_span_id": "root",
        "span_parents": ["parent"],
        "org_id": "org",
        "experiment_id": "experiment",
        "_is_merge": True,
        "_merge_paths": [["metadata"]],
        "_parent_id": "parent-row",
        "_xact_id": "transaction",
    }
    record = {
        **deepcopy(protected),
        "created": "2026-01-01T00:00:00Z",
        "metrics": {**timing, "tokens": 100, "time_to_first_token": 1},
        **{
            field: {"private": "payload"}
            for field in (
                "name",
                "input",
                "output",
                "expected",
                "metadata",
                "context",
                "attributes",
                "events",
                "links",
                "error",
                "tags",
                "attachments",
                "scores",
            )
        },
        "span_attributes": {"type": "llm", "name": "private", "private": "payload"},
    }

    class Mutate(SpanCustomizer):
        def on_span_export(self, data):
            data["span_parents"].clear()
            data["span_attributes"]["type"] = "tool"
            data["metrics"].update(start=999, end=1000)
            data["context"] = {"private": "earlier hook"}
            data["new_field"] = "private"
            return data

    class Fail(SpanCustomizer):
        def on_span_export(self, data):
            data["id"] = "wrong"
            data["metrics"].clear()
            raise RuntimeError("private")

    result = _customize_span_export(record, [Mutate(), Fail()])
    expected = {
        **protected,
        "created": "2026-01-01T00:00:00Z",
        "span_attributes": {"type": "llm"},
        "context": {"customizer_error": True},
    }
    if timing:
        expected["metrics"] = timing
    assert result == expected


def test_hook_lookup_failure_preserves_original_timing():
    class Broken:
        @property
        def on_span_export(self):
            raise GeneratorExit("private")

    assert _customize_span_export({"id": "row", "metrics": {"start": 12}, "input": "private"}, [Broken()]) == {
        "id": "row",
        "metrics": {"start": 12},
        "context": {"customizer_error": True},
    }


class _RaisingMetrics(dict):
    def __contains__(self, key):
        raise RuntimeError("private")


def test_uncopyable_protocol_fields_strip_instead_of_raising(caplog):
    parents = ["parent"]
    parents.append(parents)
    record = {"id": "row", "span_parents": parents, "metrics": _RaisingMetrics(start=1), "input": "private"}

    with caplog.at_level(logging.ERROR, logger="braintrust"):
        result = _customize_span_export(record, [_Valid()])

    # Self-referential containers can't be copied for hooks, so the record is
    # stripped, keeping identity as-is and omitting unreadable timing.
    assert type(result) is _StrippedSpanExportData
    assert result["id"] == "row"
    assert result["span_parents"] is parents
    assert "metrics" not in result
    assert "input" not in result
    assert len(caplog.records) == 1
    assert "private" not in caplog.text


@pytest.mark.parametrize("operation", ["lookup", "copy"])
@pytest.mark.parametrize(
    "error", [RuntimeError, BaseException, GeneratorExit, asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_snapshot_failures_preserve_exception_boundaries(operation, error):
    class UnreadableMetrics(dict):
        def __contains__(self, key):
            raise error("private")

    class UncopyableParents(list):
        def __iter__(self):
            raise error("private")

    record = {"id": "row", "input": "private"}
    expected = {"id": "row", "context": {"customizer_error": True}}
    if operation == "lookup":
        record["metrics"] = UnreadableMetrics(start=1)
    else:
        parents = UncopyableParents(["parent"])
        record["span_parents"] = parents
        expected["span_parents"] = parents

    customizers = [SimpleNamespace(on_span_export=lambda _: None)]
    if issubclass(error, (KeyboardInterrupt, SystemExit)):
        with pytest.raises(error):
            _customize_span_export(record, customizers)
    else:
        result = _customize_span_export(record, customizers)
        assert type(result) is _StrippedSpanExportData
        assert result == expected


@pytest.mark.parametrize("error", [GeneratorExit, asyncio.CancelledError])
def test_masking_base_exceptions_fail_closed_per_field(error):
    def mask(value):
        if value == "secret":
            raise error("private")
        return value

    result = _customize_span_export(
        {"id": "row", "input": "secret", "output": "safe", "scores": {"a": "secret"}},
        [logger._MaskingCustomizer(mask)],
    )
    assert type(result) is dict
    assert result["input"] == f"ERROR: Failed to mask field 'input' - {error.__name__}"
    assert result["output"] == "safe"
    assert result["scores"] == {"a": "secret"}


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_masking_interpreter_exits_propagate(error):
    def mask(_value):
        raise error()

    with pytest.raises(error):
        _customize_span_export({"id": "row", "input": "secret"}, [logger._MaskingCustomizer(mask)])


def test_failure_logging_is_process_wide_thread_safe_and_private(caplog):
    class Broken(SpanCustomizer):
        def on_span_export(self, data):
            raise ValueError("private exception")

    class Invalid(SpanCustomizer):
        def on_span_export(self, data):
            return None

    def export(index):
        return _customize_span_export({"id": str(index), "input": "private"}, [Broken()])

    with caplog.at_level(logging.ERROR, logger="braintrust"):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(export, range(32)))
        for customizer in (Invalid(), Broken()):
            set_span_customizers([customizer])
            assert _customize_span_export({"id": "next"}) == {
                "id": "next",
                "context": {"customizer_error": True},
            }
            set_span_customizers(None)
    assert all(result["context"] == {"customizer_error": True} for result in results)
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is None
    assert "private" not in caplog.text


def test_stripped_records_keep_marker_with_error_logging_disabled(caplog):
    with caplog.at_level(logging.CRITICAL, logger="braintrust"):
        result = _customize_span_export({"output": "private"}, [SimpleNamespace(on_span_export=lambda _: None)])
    assert result == {"context": {"customizer_error": True}}
    assert caplog.records == []


@pytest.mark.parametrize("factory", [OrderedDict, lambda: defaultdict(list)], ids=["ordered-dict", "defaultdict"])
def test_dict_subclass_results_are_accepted_as_plain_dicts(factory):
    class Replace(SpanCustomizer):
        def on_span_export(self, data):
            result = factory()
            result.update(data, output="redacted")
            return result

    set_span_customizers([Replace(), Replace()])
    result = _customize_span_export({"id": "row", "output": "secret"})
    assert type(result) is dict
    assert result == {"id": "row", "output": "redacted"}


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interpreter_exits_propagate_from_hooks(error, caplog):
    class Interrupt(SpanCustomizer):
        def on_span_export(self, data):
            raise error()

    set_span_customizers([Interrupt()])
    with caplog.at_level(logging.ERROR, logger="braintrust"), pytest.raises(error):
        _customize_span_export({"id": "row", "output": "secret"})
    assert caplog.records == []


@pytest.mark.parametrize("backend", ["memory", "http"])
@pytest.mark.parametrize("include_healthy", [False, True], ids=["all-stripped", "mixed-batch"])
def test_stripped_records_skip_attachments_and_continue_export(
    monkeypatch, with_memory_logger, test_logger, caplog, backend, include_healthy
):
    attachment = Attachment(data=b"private", filename="private.txt", content_type="text/plain")
    upload = MagicMock()
    monkeypatch.setattr(attachment, "upload", upload)
    invocations = []
    later_hooks = []

    class Reject(SpanCustomizer):
        def on_span_export(self, data):
            invocations.append(data["id"])
            if data["id"] == failed.id:
                raise ValueError("private exception")
            return data

    class Next(SpanCustomizer):
        def on_span_export(self, data):
            later_hooks.append(data["id"])
            return data

    set_span_customizers([Reject(), Next()])
    failed = test_logger.start_span(input=attachment)
    healthy = test_logger.start_span(input="safe") if include_healthy else None
    pending = list(with_memory_logger.logs)
    with_memory_logger.logs.clear()
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    connection = MagicMock()
    connection.post.return_value = SimpleNamespace(ok=True)
    background = (
        logger._MemoryBackgroundLogger()
        if backend == "memory"
        else logger._HTTPBackgroundLogger(LazyValue(lambda: connection, use_mutex=False))
    )
    masked = []

    def mask(value):
        masked.append(value)
        return value

    background.set_masking_function(mask)
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        if isinstance(background, logger._MemoryBackgroundLogger):
            background.log(*pending)
            background.flush()
            background.flush()
            rows = background.pop()
            assert background.upload_attempts == []
        else:
            background.sync_flush = True
            background._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
            for item in pending:
                background.queue.put(item)
            background.flush()
            # Reusing a stripped LazyValue must neither invoke nor log its failure again.
            waves, attachments = background._unwrap_lazy_values(pending)
            rows = [row for wave in waves for row in wave]
            assert attachments == []
            sent = [row for call in connection.post.call_args_list for row in json.loads(call.kwargs["data"])["rows"]]
            assert [row["id"] for row in sent] == [failed.id, *([healthy.id] if healthy else [])]

    healthy_ids = [healthy.id] if healthy else []
    assert [row["id"] for row in rows] == [failed.id, *healthy_ids]
    assert rows[0]["context"] == {"customizer_error": True}
    assert "input" not in rows[0]
    assert set(rows[0]["metrics"]) == {"start"}
    assert later_hooks == healthy_ids
    assert invocations == [failed.id, *healthy_ids]
    upload.assert_not_called()
    assert attachment not in masked
    assert bool(masked) == include_healthy
    assert len(caplog.records) == 1
    assert "private" not in caplog.text


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
            assert next(row for row in first if row["id"] == manual.id)["input"] == {"secret": "redacted"}
            manual.log(output={"secret": "manual output"}, error="private manual error")
            stream = BraintrustStream([BraintrustJsonChunk(data='{"secret":"output"}')])
            span.log(output=stream, error="private error")
            span.set_attributes(name="renamed")
            with span.start_span(name="manual child", input=application_input) as manual_child:
                manual_child.log(output={"secret": "child output"}, error="private child error")
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
    for manual_span in (manual, manual_child):
        manual_export = next(row for row in rows if row["id"] == manual_span.id)
        assert manual_export["output"] == {"secret": "redacted"}
        assert "error" not in manual_export
    assert next(row for row in rows if row["id"] == manual_child.id)["input"] == {"secret": "redacted"}
    assert next(row for row in rows if row["id"] == dataset_id)["input"] == {"secret": "dataset"}
    assert application_input == {"secret": "input"}
    assert stream.final_value() == {"secret": "output"}
    assert {row_id for row_id, _ in seen} == {manual.id, span.id, manual_child.id}
    for current_span, input_fields, output_fields in (
        (manual, [True, False, False], [False, True, False]),
        (span, [True, False, False, False], [False, True, False, False]),
        (manual_child, [True, False, False], [False, True, False]),
    ):
        keys = [keys for row_id, keys in seen if row_id == current_span.id]
        assert ["input" in fields for fields in keys] == input_fields
        assert ["output" in fields for fields in keys] == output_fields
    assert not any("expected" in keys for _, keys in seen)


@pytest.mark.parametrize("api", ["logger", "experiment", "exported"])
def test_update_span_customizes_each_record_before_attachments(with_memory_logger, test_logger, api, caplog):
    parent = init_test_exp("span-customizer-updates") if api == "experiment" else test_logger
    with parent.start_span(input="original") as span:
        exported = span.export()
    with_memory_logger.pop()

    def update(**event):
        if api == "exported":
            logger.update_span(exported, **event)
        else:
            parent.update_span(span.id, **event)

    calls = []
    accepted = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            calls.append(data.get("output"))
            if data.get("output") == "reject":
                raise ValueError("private exception")
            data["input"] = "redacted"
            return data

    class Observe(SpanCustomizer):
        def on_span_export(self, data):
            accepted.append(data["output"])
            return data

    attachment = Attachment(data=b"private", filename="private.txt", content_type="text/plain")
    set_span_customizers([Redact(), Observe()])
    update(input=attachment, output="reject")
    update(input=attachment, output="accepted")
    # Registration changes must not change hooks for already queued updates.
    set_span_customizers(None)
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        rows = with_memory_logger.pop()

    assert calls == ["reject", "accepted"]
    assert accepted == ["accepted"]
    assert len(rows) == 2
    assert rows[0]["id"] == span.id
    assert rows[0]["context"] == {"customizer_error": True}
    assert "metrics" not in rows[0]
    assert "input" not in rows[0]
    assert rows[1]["input"] == "redacted"
    assert rows[1]["output"] == "accepted"
    assert all(row["_is_merge"] is True for row in rows)
    assert with_memory_logger.upload_attempts == []
    assert "private" not in caplog.text


def test_customization_precedes_attachments_masking_and_reuses_records_on_retry(
    monkeypatch, with_memory_logger, test_logger, caplog
):
    attachment = Attachment(data=b"private", filename="private.txt", content_type="text/plain")
    rejected_attachment = Attachment(data=b"secret", filename="secret.txt", content_type="text/plain")
    upload = MagicMock()
    monkeypatch.setattr(rejected_attachment, "upload", upload)
    seen_attachments = []
    invocations = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            invocations.append(data["id"])
            if data.get("input") is rejected_attachment:
                raise ValueError("private exception")
            if "input" in data:
                seen_attachments.append(data["input"])
                data["input"] = "redacted"
            if "output" in data:
                data["output"] = "redacted output"
            return data

    set_span_customizers([Redact()])
    span = test_logger.start_span(input=attachment)
    span.log(output="private output")
    span.end()
    rejected = test_logger.start_span(input=rejected_attachment)
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
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        background.flush()

    assert attempts == 2
    assert invocations == [span.id] * 3 + [rejected.id]
    assert seen_attachments == [attachment]
    assert seen_attachments[0] is attachment
    assert "redacted" in masked
    assert "redacted output" in masked
    assert payloads[0] == payloads[1]
    exported = next(row for row in json.loads(payloads[1])["rows"] if row["id"] == span.id)
    assert exported["input"] == "redacted"
    assert exported["output"] == "redacted output"
    stripped = next(row for payload in payloads[1:] for row in json.loads(payload)["rows"] if row["id"] == rejected.id)
    assert stripped["context"] == {"customizer_error": True}
    assert "input" not in stripped
    upload.assert_not_called()
    assert len(caplog.records) == 1
    assert "private" not in caplog.text


def _retrying_http_logger(monkeypatch, pending):
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
    background._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
    for item in pending:
        background.queue.put(item)
    return background, payloads


@pytest.mark.parametrize("error", [BaseException, GeneratorExit, asyncio.CancelledError])
def test_base_exception_hook_failures_strip_only_that_record(
    monkeypatch, with_memory_logger, test_logger, caplog, error
):
    invocations = []

    class Reject(SpanCustomizer):
        def on_span_export(self, data):
            invocations.append(data["id"])
            if data["id"] == failed.id:
                raise error("private exception text")
            return data

    set_span_customizers([Reject()])
    failed = test_logger.start_span(input="private payload")
    healthy = test_logger.start_span(input="safe")
    pending = list(with_memory_logger.logs)
    with_memory_logger.logs.clear()
    background, payloads = _retrying_http_logger(monkeypatch, pending)
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        background.flush()

    # The transport retry reuses the memoized stripped record instead of rerunning the hook.
    # Different row IDs need no barrier, so both records share one request.
    assert invocations == [failed.id, healthy.id]
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    rows = json.loads(payloads[1])["rows"]
    assert [row["id"] for row in rows] == [failed.id, healthy.id]
    assert rows[0]["context"] == {"customizer_error": True}
    assert "input" not in rows[0]
    assert rows[1]["input"] == "safe"
    assert len(caplog.records) == 1
    assert "private" not in caplog.text


def test_redacting_integration_attachment_prevents_upload(monkeypatch, with_memory_logger, test_logger):
    # Integrations convert inline media into Attachment objects at capture time.
    attachment = _resolved_attachment_from_bytes(b"private image", "image/png", prefix="input").attachment
    upload = MagicMock()
    monkeypatch.setattr(attachment, "upload", upload)
    seen = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            for message in data.get("input", []):
                for part in message["content"]:
                    if isinstance(part.get("image_url", {}).get("url"), Attachment):
                        seen.append(part["image_url"]["url"])
                        part["image_url"]["url"] = "[redacted image]"
            return data

    set_span_customizers([Redact()])
    span = test_logger.start_span(
        input=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": attachment}}]}]
    )
    pending = list(with_memory_logger.logs)
    with_memory_logger.logs.clear()
    background, payloads = _retrying_http_logger(monkeypatch, pending)
    background.flush()

    assert seen == [attachment]
    upload.assert_not_called()
    exported = next(row for row in json.loads(payloads[-1])["rows"] if row["id"] == span.id)
    assert exported["input"][0]["content"][0]["image_url"]["url"] == "[redacted image]"


@pytest.mark.parametrize("backend", ["memory", "http"])
def test_masking_remains_logger_local_and_runs_on_merged_manual_records(
    monkeypatch, with_memory_logger, test_logger, backend
):
    class Customize(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = None
            if "metadata" in data and "secret" in data["metadata"]:
                data["metadata"]["customized"] = True
            if "output" in data:
                data["output"] = "customized output"
            return data

    set_span_customizers([Customize()])
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    background = (
        logger._MemoryBackgroundLogger()
        if backend == "memory"
        else logger._HTTPBackgroundLogger(LazyValue(lambda: MagicMock(), use_mutex=False))
    )
    other_background = logger._MemoryBackgroundLogger()

    def export(target):
        with test_logger.start_span(input="private input", metadata={"secret": "private"}) as span:
            span.log(metadata={"redact": True}, output="private output")
        pending = list(with_memory_logger.logs)
        with_memory_logger.logs.clear()
        if isinstance(target, logger._MemoryBackgroundLogger):
            target.log(*pending)
            return target.pop()[0]
        waves, _ = target._unwrap_lazy_values(pending)
        return waves[0][0]

    def mask(value):
        if value is None:
            return "masked null"
        if value == "customized output":
            return "masked customized output"
        if isinstance(value, dict) and value.get("redact") and value.get("customized"):
            return {"secret": "redacted"}
        return value

    background.set_masking_function(mask)
    masked = export(background)
    assert masked["metadata"] == {"secret": "redacted"}
    assert masked["input"] == "masked null"
    assert masked["output"] == "masked customized output"
    other = export(other_background)
    assert other["metadata"] == {"secret": "private", "redact": True, "customized": True}
    assert other["input"] is None
    assert other["output"] == "customized output"

    background.set_masking_function(lambda _: "replacement")
    assert export(background)["input"] == "replacement"
    background.set_masking_function(None)
    unmasked = export(background)
    assert unmasked["metadata"] == {"secret": "private", "redact": True, "customized": True}
    assert unmasked["input"] is None
    assert unmasked["output"] == "customized output"


def test_auto_instrument_registration_and_disable(monkeypatch, with_memory_logger, test_logger):
    # Registration should work without importing optional provider libraries.
    monkeypatch.setattr("braintrust.auto._instrument_integration", lambda _module, _class: False)

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = "redacted"
            return data

    auto_instrument(span_customizers=[Redact()])
    auto_instrument()  # Omitted configuration must not reset customizers.
    with test_logger.start_span(input="private"):
        pass
    assert with_memory_logger.pop()[0]["input"] == "redacted"
    auto_instrument(span_customizers=[])
    with test_logger.start_span(input="untouched"):
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


def test_stripped_cache_record_is_released_and_later_records_are_independent(
    with_memory_logger, span_cache_experiment, caplog
):
    calls = []

    class RejectPrivate(SpanCustomizer):
        def on_span_export(self, data):
            calls.append(data["id"])
            if "input" in data:
                raise ValueError("private exception")
            return data

    set_span_customizers([RejectPrivate()])
    experiment = span_cache_experiment
    span = experiment.start_span(input="private payload")
    cache = experiment.state.span_cache
    with caplog.at_level(logging.ERROR, logger="braintrust"):
        for _ in range(2):
            cached = _cached(experiment, span)[span.span_id]
            assert cached.input is None
            assert cached.output is None
            # Only the SDK span type survives a failure, not its name.
            assert cached.span_attributes == {"type": "eval"}
        rows = with_memory_logger.pop()
        assert rows[0]["context"] == {"customizer_error": True}
        assert "input" not in rows[0]
    assert calls == [span.id]
    assert len(caplog.records) == 1
    assert cache._pending_records == {}
    assert span.root_span_id in cache._root_span_index
    assert "private" not in caplog.text

    span.log(output="safe")
    cached = _cached(experiment, span)[span.span_id]
    assert cached.input is None
    assert cached.output == "safe"
    rows = with_memory_logger.pop()
    assert rows[0]["output"] == "safe"
    assert "input" not in rows[0]
    assert calls == [span.id, span.id]


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
        manual.log(output="manual output")

    async def unexpected_flush():
        pytest.fail("Local trace reads must not require a backend flush")

    trace = LocalTrace("experiment", experiment.id, manual.root_span_id, unexpected_flush, experiment.state)
    cached = {record.span_id: record for record in await trace.get_spans()}
    assert cached[span.span_id].metadata == {"secret": "redacted"}
    assert cached[span.span_id].output == "redacted"
    assert cached[manual.span_id].metadata == {"secret": "redacted"}
    assert cached[manual.span_id].output == "redacted"
    assert customized_outputs == ["private output", "manual output"]

    # Export must reuse the transformation performed for the local trace read.
    rows = with_memory_logger.pop()
    assert customized_outputs == ["private output", "manual output"]
    exported = next(row for row in rows if row["id"] == span.id)
    if not resolve_initial:
        assert exported["metadata"] == {"secret": "redacted"}
    assert exported["output"] == "redacted"
    exported_manual = next(row for row in rows if row["id"] == manual.id)
    assert exported_manual["output"] == "redacted"
    if not resolve_initial:
        assert exported_manual["metadata"] == {"secret": "redacted"}


@pytest.mark.parametrize("fail", [False, True], ids=["customized", "stripped"])
def test_trace_cache_waits_for_inflight_customization(with_memory_logger, span_cache_experiment, fail):
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
                if fail:
                    raise ValueError("private exception")
                data["output"] = "redacted"
            return data

    set_span_customizers([Redact()])
    experiment = span_cache_experiment
    with experiment.start_span(name="manual") as manual:
        with manual.start_span(name="manual child") as span:
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

    exported = next(row for row in rows if row["id"] == span.id)
    if fail:
        assert cached[span.span_id].output is None
        assert "output" not in exported
    else:
        assert cached[span.span_id].output == "redacted"
        assert exported["output"] == "redacted"
    assert customized_outputs == ["private output"]


def test_span_cache_written_eagerly_without_customizers(with_memory_logger, span_cache_experiment):
    experiment = span_cache_experiment
    with experiment.start_span(name="instrumented", internal={"instrumentation": "test-auto"}) as span:
        span.log(output="output")
    assert _cached(experiment, span)[span.span_id].output == "output"


def test_http_export_smoke_preserves_failure_barriers_and_attachments(
    monkeypatch, with_memory_logger, test_logger, caplog
):
    requests = []
    invocations = []
    later_hooks = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, payload))
            # Retry the failed record's first request using the real HTTP client.
            self.send_response(503 if len(requests) == 2 else 200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args):
            pass

    class Customize(SpanCustomizer):
        def on_span_export(self, data):
            invocations.append(data["id"])
            if data.get("output") == "private":
                data["metrics"]["start"] = 999
                data["metrics"]["end"] = 1000
                data["span_parents"].append("wrong")
                raise ValueError("private exception")
            return data

    class Observe(SpanCustomizer):
        def on_span_export(self, data):
            later_hooks.append(data["id"])
            return data

    failed_attachment = Attachment(data=b"private", filename="private.txt", content_type="text/plain")
    healthy_attachment = Attachment(data=b"safe", filename="safe.txt", content_type="text/plain")
    failed_upload = MagicMock()
    healthy_upload = MagicMock(return_value={"upload_status": "done"})
    monkeypatch.setattr(failed_attachment, "upload", failed_upload)
    monkeypatch.setattr(healthy_attachment, "upload", healthy_upload)
    set_span_customizers([Customize(), Observe()])
    span = test_logger.start_span(input="before")
    span.log(output="private", input=failed_attachment, metrics={"start": 12, "end": 15, "tokens": 42})
    span.log(output="after")
    healthy = test_logger.start_span(input=healthy_attachment)
    pending = list(with_memory_logger.logs)
    with_memory_logger.logs.clear()
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    monkeypatch.setattr(logger.time, "sleep", lambda _: None)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = logger.HTTPConnection(f"http://127.0.0.1:{server.server_port}")
        background = logger._HTTPBackgroundLogger(LazyValue(lambda: connection, use_mutex=False))
        background.sync_flush = True
        background.num_tries = 2
        background._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
        # Arbitrary masking must not change the SDK's diagnostic or timing.
        background.set_masking_function(lambda value: "masked" if not isinstance(value, Attachment) else value)
        try:
            for item in pending:
                background.queue.put(item)
            with caplog.at_level(logging.ERROR, logger="braintrust"):
                background.flush()
        finally:
            connection.session.close()
            server.shutdown()
            thread.join(timeout=5)

    assert len(requests) == 4
    assert all(path == "/logs3" for path, _payload in requests)
    assert requests[1] == requests[2]
    before, failed, after = [json.loads(requests[index][1])["rows"] for index in (0, 2, 3)]
    # The barrier only orders updates to the failed row; other rows go first.
    assert [row["id"] for row in before] == [span.id, healthy.id]
    assert before[0]["input"] == "masked"
    assert failed == [
        {
            "id": span.id,
            "span_id": span.span_id,
            "root_span_id": span.root_span_id,
            "span_parents": span.span_parents,
            "project_id": "project",
            "log_id": "g",
            "_is_merge": True,
            "metrics": {"start": 12, "end": 15},
            "context": {"customizer_error": True},
        }
    ]
    assert after == [{**after[0], "output": "masked"}]
    assert after[0]["id"] == span.id
    # Merging/masking adjacent successes must not poison the memoized failure.
    assert pending[1].get() == failed[0]
    assert invocations == [span.id, span.id, span.id, healthy.id]
    assert later_hooks == [span.id, span.id, healthy.id]
    failed_upload.assert_not_called()
    healthy_upload.assert_called_once()
    assert len(caplog.records) == 1
    assert "private" not in caplog.text


def _barrier_pending():
    class Fail(SpanCustomizer):
        def on_span_export(self, data):
            raise ValueError("private")

    row = {"project_id": "project", "log_id": "g"}
    records = [
        {"id": "a", **row, "input": "start"},
        {"id": "b", **row, "input": "other"},
        _customize_span_export({"id": "a", **row, "_is_merge": True, "output": "private"}, [Fail()]),
        {"id": "a", **row, "_is_merge": True, "output": "end"},
    ]
    return [LazyValue(lambda record=record: record, use_mutex=False) for record in records]


def _recording_http_logger(monkeypatch, pending, fails=lambda _data: False):
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    monkeypatch.setattr(logger.time, "sleep", lambda _: None)
    attempts = []

    def send(_path, *, data):
        attempts.append([(row["id"], row.get("input") or row.get("output")) for row in json.loads(data)["rows"]])
        if fails(data):
            raise ConnectionError("permanent failure")
        return SimpleNamespace(ok=True)

    connection = MagicMock()
    connection.post.side_effect = send
    background = logger._HTTPBackgroundLogger(LazyValue(lambda: connection, use_mutex=False))
    background.num_tries = 2
    background.sync_flush = True
    background._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
    for item in pending:
        background.queue.put(item)
    return background, attempts


class _ShutDownPool:
    def submit(self, *_args, **_kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")


_BARRIER_WAVES = [[("a", "start"), ("b", "other")], [("a", None)], [("a", "end")]]


@pytest.mark.parametrize("pool", ["running", "shut-down"])
def test_flush_sends_waves_in_order_once_each(monkeypatch, pool):
    if pool == "shut-down":
        monkeypatch.setattr(logger, "HTTP_REQUEST_THREAD_POOL", _ShutDownPool())
    background, attempts = _recording_http_logger(monkeypatch, _barrier_pending())
    background.flush()
    assert attempts == _BARRIER_WAVES


@pytest.mark.parametrize("pool", ["running", "shut-down"])
def test_flush_runtime_error_in_request_does_not_resend_batches(monkeypatch, pool):
    if pool == "shut-down":
        monkeypatch.setattr(logger, "HTTP_REQUEST_THREAD_POOL", _ShutDownPool())
    background, attempts = _recording_http_logger(monkeypatch, _barrier_pending())
    submit = background._submit_logs_request
    calls = 0

    def submit_or_fail(batch, max_request_size_result):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("request failed outside its retry loop")
        return submit(batch, max_request_size_result)

    monkeypatch.setattr(background, "_submit_logs_request", submit_or_fail)
    with pytest.raises(exceptiongroup.BaseExceptionGroup) as raised:
        background.flush()

    assert [type(e) for e in raised.value.exceptions] == [RuntimeError]
    assert calls == 3
    assert attempts == [_BARRIER_WAVES[0], _BARRIER_WAVES[2]]


def test_flush_failed_batch_does_not_skip_later_batches_or_waves(monkeypatch):
    background, attempts = _recording_http_logger(
        monkeypatch, _barrier_pending(), fails=lambda data: b'"input": "start"' in data
    )
    with pytest.raises(exceptiongroup.BaseExceptionGroup) as raised:
        background.flush(batch_size=1)

    assert len(raised.value.exceptions) == 1
    # Batches within a wave run concurrently; waves run in order.
    assert sorted(attempts[:3]) == [[("a", "start")], [("a", "start")], [("b", "other")]]
    assert attempts[3:] == [[("a", None)], [("a", "end")]]
