import concurrent.futures
import logging
import logging.handlers
import os
import pickle
import queue
import select
import signal
import sys
import threading
import warnings
from unittest.mock import MagicMock, patch

import braintrust.logs as braintrust_logs
import pytest
from braintrust.api._transport import HTTPConnection
from braintrust.logs import BraintrustLogHandler
from braintrust.test_helpers import init_test_logger, with_memory_logger  # noqa: F401


def test_handler_forwards_log_record(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord(
        name="payments.checkout",
        level=logging.WARNING,
        pathname="/app/checkout.py",
        lineno=42,
        msg="Payment %s failed",
        args=("pay_123",),
        exc_info=None,
        func="charge",
    )
    record.created = 1234.5
    record.customer_id = "cus_123"

    handler.handle(record)

    assert len(with_memory_logger.logs) == 1
    [row] = with_memory_logger.pop()
    assert row["output"] == "Payment pay_123 failed"
    assert row["created"] == "1970-01-01T00:20:34.500000+00:00"
    assert row["metrics"] == {"start": 1234.5, "end": 1234.5}
    assert "otel" not in row.get("context", {})
    assert row["metadata"] == {
        "braintrust.template": "Payment %s failed",
        "braintrust.template.parameter.0": "pay_123",
        "code.file.path": "/app/checkout.py",
        "code.function.name": "charge",
        "code.line.number": 42,
        "customer_id": "cus_123",
        "logger.name": "payments.checkout",
    }
    assert row["span_attributes"]["log_level"] == "warn"


@pytest.mark.parametrize(
    ("template", "expected_output", "expected_parameters"),
    [
        ("payload=%s", "payload={'id': 1}", {"braintrust.template.parameter.0": {"id": 1}}),
        ("payload=%(id)s", "payload=1", {"braintrust.template.parameter.id": 1}),
        (
            "%s %(id)s",
            "{'id': 1} 1",
            {
                "braintrust.template.parameter.0": {"id": 1},
                "braintrust.template.parameter.id": 1,
            },
        ),
        (
            "literal=%%(id)s payload=%s",
            "literal=%(id)s payload={'id': 1}",
            {"braintrust.template.parameter.0": {"id": 1}},
        ),
    ],
)
def test_handler_distinguishes_positional_and_named_mapping_arguments(
    with_memory_logger, template, expected_output, expected_parameters
):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord("app", logging.INFO, __file__, 1, template, ({"id": 1},), None)

    handler.handle(record)

    [row] = with_memory_logger.pop()
    assert row["output"] == expected_output
    assert row["metadata"] == {
        "braintrust.template": template,
        **expected_parameters,
        "code.file.path": __file__,
        "code.function.name": None,
        "code.line.number": 1,
        "logger.name": "app",
    }


def test_handler_preserves_unix_epoch_timestamp(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord("app", logging.INFO, __file__, 1, "message", (), None)
    record.created = 0

    handler.handle(record)

    [row] = with_memory_logger.pop()
    assert row["created"] == "1970-01-01T00:00:00+00:00"
    assert row["metrics"] == {"start": 0, "end": 0}


@pytest.mark.parametrize(
    ("python_level", "braintrust_level"),
    [
        (1, "trace"),
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warn"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "fatal"),
    ],
)
def test_handler_maps_python_log_levels(with_memory_logger, python_level, braintrust_level):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord("app", python_level, __file__, 1, "message", (), None)

    handler.handle(record)

    [row] = with_memory_logger.pop()
    assert row["span_attributes"]["log_level"] == braintrust_level
    assert "braintrust.log_level" not in row.get("metadata", {})


def test_handler_forwards_exception_info(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))

    try:
        raise ValueError("invalid payment")
    except ValueError:
        record = logging.LogRecord("payments", logging.ERROR, __file__, 1, "Charge failed", (), None)
        record.exc_info = sys.exc_info()

    handler.handle(record)

    [row] = with_memory_logger.pop()
    assert row["output"].startswith("Charge failed\nTraceback (most recent call last):")
    assert row["output"].endswith("ValueError: invalid payment")
    assert "error" not in row


def test_handler_ignores_braintrust_loggers(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord("braintrust.logger", logging.ERROR, __file__, 1, "internal", (), None)

    handler.handle(record)

    assert with_memory_logger.pop() == []


def test_handler_forwards_application_urllib3_logs(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    source_logger = logging.getLogger("urllib3.connectionpool")
    original_disabled = source_logger.disabled
    original_level = source_logger.level
    original_propagate = source_logger.propagate

    source_logger.disabled = False
    source_logger.setLevel(logging.DEBUG)
    source_logger.propagate = False
    source_logger.addHandler(handler)
    try:
        source_logger.debug("request")
    finally:
        source_logger.removeHandler(handler)
        source_logger.disabled = original_disabled
        source_logger.setLevel(original_level)
        source_logger.propagate = original_propagate

    [row] = with_memory_logger.pop()
    assert row["output"] == "request"
    assert row["metadata"]["logger.name"] == "urllib3.connectionpool"


def test_handler_ignores_internal_logs_before_acquiring_lock():
    handler = BraintrustLogHandler(init_test_logger(__name__))
    source_logger = logging.getLogger("urllib3.connectionpool")
    original_disabled = source_logger.disabled
    original_level = source_logger.level
    original_propagate = source_logger.propagate
    connection = HTTPConnection("")
    connection.session.get = MagicMock(side_effect=lambda *_args, **_kwargs: source_logger.debug("internal"))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    source_logger.disabled = False
    source_logger.setLevel(logging.DEBUG)
    source_logger.propagate = False
    source_logger.addHandler(handler)
    handler.acquire()
    future = executor.submit(connection.get, "https://api.braintrust.dev")
    try:
        assert future.result(timeout=1) is None
    finally:
        handler.release()
        executor.shutdown(wait=True)
        source_logger.removeHandler(handler)
        source_logger.disabled = original_disabled
        source_logger.setLevel(original_level)
        source_logger.propagate = original_propagate


def test_handler_ignores_queued_internal_transport_logs(with_memory_logger):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    log_queue = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(log_queue, handler)
    source_logger = logging.getLogger("urllib3.connectionpool")
    original_disabled = source_logger.disabled
    original_level = source_logger.level
    original_propagate = source_logger.propagate
    connection = HTTPConnection("")

    def emit_internal_log(*_args, **_kwargs):
        source_logger.debug("internal")

    connection.session.get = MagicMock(side_effect=emit_internal_log)

    source_logger.disabled = False
    source_logger.setLevel(logging.DEBUG)
    source_logger.propagate = False
    source_logger.addHandler(queue_handler)
    try:
        connection.get("https://api.braintrust.dev")
        assert log_queue.qsize() == 1
        queued_record = log_queue.get_nowait()
        assert getattr(queued_record, "_braintrust_internal_http_transport", False) is True
        log_queue.put(queued_record)
        listener.start()
    finally:
        listener.stop()
        source_logger.removeHandler(queue_handler)
        source_logger.disabled = original_disabled
        source_logger.setLevel(original_level)
        source_logger.propagate = original_propagate

    assert with_memory_logger.pop() == []


def test_handler_forwards_queued_application_logs_during_transport():
    test_logger = init_test_logger(__name__)
    handler = BraintrustLogHandler(test_logger)
    log_queue = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(log_queue, handler)
    source_logger = logging.getLogger(f"application.http.{__name__}")
    original_disabled = source_logger.disabled
    original_level = source_logger.level
    original_propagate = source_logger.propagate
    connection = HTTPConnection("")

    connection.session.get = MagicMock(side_effect=lambda *_args, **_kwargs: source_logger.info("application"))
    source_logger.disabled = False
    source_logger.setLevel(logging.INFO)
    source_logger.propagate = False
    source_logger.addHandler(queue_handler)
    listener_started = False
    try:
        with patch.object(test_logger, "_emit_log_record") as emit_log_record:
            connection.get("https://api.braintrust.dev")
            listener.start()
            listener_started = True
            listener.stop()
            listener_started = False

            emit_log_record.assert_called_once()
            assert emit_log_record.call_args.kwargs["body"] == "application"
    finally:
        if listener_started:
            listener.stop()
        source_logger.removeHandler(queue_handler)
        source_logger.disabled = original_disabled
        source_logger.setLevel(original_level)
        source_logger.propagate = original_propagate


def test_handler_preserves_queued_template_arguments_and_span_context(with_memory_logger):
    test_logger = init_test_logger(__name__)
    handler = BraintrustLogHandler(test_logger)
    log_queue = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(log_queue, handler)
    source_logger = logging.getLogger(f"payments.queue.{__name__}")
    original_disabled = source_logger.disabled
    original_level = source_logger.level
    original_propagate = source_logger.propagate

    source_logger.disabled = False
    source_logger.setLevel(logging.INFO)
    source_logger.propagate = False
    source_logger.addHandler(queue_handler)
    listener_started = False
    try:
        with patch.object(test_logger, "_emit_log_record") as emit_log_record:
            with test_logger.start_span(name="checkout") as owner:
                source_logger.info("Payment %s failed", "pay_123")
                owner_span_id = owner.span_id
                owner_root_span_id = owner.root_span_id

            # QueueHandler prepares the record on the producer thread before the
            # listener sees it, including rendering msg and clearing args.
            assert log_queue.queue[0].args is None
            listener.start()
            listener_started = True
            listener.stop()
            listener_started = False

            emit_log_record.assert_called_once()
            call_kwargs = emit_log_record.call_args.kwargs
    finally:
        if listener_started:
            listener.stop()
        source_logger.removeHandler(queue_handler)
        source_logger.disabled = original_disabled
        source_logger.setLevel(original_level)
        source_logger.propagate = original_propagate

    assert call_kwargs["body"] == "Payment pay_123 failed"
    assert call_kwargs["span_id"] == owner_span_id
    assert call_kwargs["root_span_id"] == owner_root_span_id
    assert call_kwargs["lookup_current_span"] is False
    assert call_kwargs["metadata"]["braintrust.template"] == "Payment %s failed"
    assert call_kwargs["metadata"]["braintrust.template.parameter.0"] == "pay_123"


def test_handler_keeps_queued_records_with_non_pickleable_arguments_pickleable():
    test_logger = init_test_logger(__name__)
    handler = BraintrustLogHandler(test_logger)
    queue_handler = logging.handlers.QueueHandler(queue.Queue())
    callback = lambda: None  # noqa: E731
    callback_repr = repr(callback)
    record = logging.getLogger("app").makeRecord("app", logging.INFO, __file__, 1, "callback=%s", (callback,), None)

    prepared_record = queue_handler.prepare(record)
    with patch.object(test_logger, "_emit_log_record") as emit_log_record:
        handler.handle(prepared_record)
        assert emit_log_record.call_args.kwargs["metadata"]["braintrust.template.parameter.0"] == callback_repr

        restored_record = pickle.loads(pickle.dumps(prepared_record))
        handler.handle(restored_record)
        assert emit_log_record.call_args.kwargs["metadata"]["braintrust.template.parameter.0"] == callback_repr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_record_factory_resets_lock_after_fork():
    BraintrustLogHandler(init_test_logger(__name__))
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_logging_hooks_lock():
        with braintrust_logs._LOGGING_HOOKS_LOCK:
            lock_acquired.set()
            release_lock.wait()

    lock_holder = threading.Thread(target=hold_logging_hooks_lock)
    lock_holder.start()
    assert lock_acquired.wait(timeout=1)

    read_fd, write_fd = os.pipe()
    child_pid = None
    child_reaped = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                logging.getLogRecordFactory()("app", logging.INFO, __file__, 1, "message", (), None)
                os.write(write_fd, b"ok")
            except BaseException:
                os._exit(1)
            else:
                os._exit(0)

        os.close(write_fd)
        readable, _, _ = select.select([read_fd], [], [], 2)
        assert readable, "child hung on the inherited logging-hooks lock"
        assert os.read(read_fd, 2) == b"ok"
        _, status = os.waitpid(child_pid, 0)
        child_reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        if child_pid not in (None, 0) and not child_reaped:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
        os.close(read_fd)
        release_lock.set()
        lock_holder.join(timeout=1)
