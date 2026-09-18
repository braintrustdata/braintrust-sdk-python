import logging
import sys

import pytest
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


@pytest.mark.parametrize("logger_name", ["braintrust.logger", "urllib3.connectionpool"])
def test_handler_ignores_internal_transport_loggers(with_memory_logger, logger_name):
    handler = BraintrustLogHandler(init_test_logger(__name__))
    record = logging.LogRecord(logger_name, logging.ERROR, __file__, 1, "internal", (), None)

    handler.handle(record)

    assert with_memory_logger.pop() == []
