# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
"""
Tests that public tracing APIs never raise exceptions into customer code.

Even when the SDK's internals fail (HTTP errors, validation errors, bad
serialization, buggy integrations), methods like Span.log, Span.end,
Logger.log, start_span, and @traced-wrapped functions must degrade
gracefully rather than crashing the customer's application.

Failures should be swallowed and surfaced via the `braintrust` Python
logger so customers can diagnose them without catching exceptions.
"""

import asyncio
import logging

import pytest
from braintrust import logger
from braintrust.logger import SpanImpl, start_span
from braintrust.test_helpers import (
    init_test_logger,
    with_memory_logger,  # noqa: F401
    with_simulate_login,  # noqa: F401
)


BOOM = "boom"


class ExplodingRepr:
    """Object whose repr/str raise, simulating user-supplied objects
    with broken dunder methods that flow into serialization."""

    def __repr__(self):
        raise RuntimeError(BOOM)

    def __str__(self):
        raise RuntimeError(BOOM)


@pytest.fixture
def sabotage_log_internal(monkeypatch):
    """Force SpanImpl.log_internal to raise on every call."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(SpanImpl, "log_internal", _boom)


def _assert_warning_emitted(caplog, needle: str = BOOM) -> None:
    """Confirm the SDK logged a warning instead of raising."""
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("braintrust")]
    assert any(needle in m for m in messages), f"Expected {needle!r} in braintrust warnings, got: {messages}"


# ---------------------------------------------------------------------------
# Span.log / Span.end / Span.start_span / Span.log_feedback
# ---------------------------------------------------------------------------


def test_span_log_swallows_internal_exception(with_memory_logger, sabotage_log_internal, caplog):
    """Span.log() must not propagate SDK-internal exceptions."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    with start_span("outer", set_current=False) as span:
        span.log(output="hello")  # must not raise

    _assert_warning_emitted(caplog)


def test_span_end_swallows_internal_exception(with_memory_logger, sabotage_log_internal, caplog):
    """Calling span.end() explicitly should never raise."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    span = start_span("outer", set_current=False)
    span.end()  # must not raise


def test_span_log_feedback_swallows_internal_exception(with_memory_logger, monkeypatch, caplog):
    """Span.log_feedback() must not propagate errors."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "_log_feedback_impl", _boom)

    with start_span("outer", set_current=False) as span:
        span.log_feedback(scores={"quality": 1.0})  # must not raise


def test_span_start_span_returns_noop_on_failure(with_memory_logger, monkeypatch, caplog):
    """If creating a child span fails, return NOOP_SPAN instead of raising."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    with start_span("outer", set_current=False) as span:
        original_init = SpanImpl.__init__

        def _boom_init(self, *args, **kwargs):
            raise RuntimeError(BOOM)

        monkeypatch.setattr(SpanImpl, "__init__", _boom_init)

        child = span.start_span("child")  # must not raise
        assert child is not None
        child.log(output="ignored")  # noop must also not raise
        child.end()

        monkeypatch.setattr(SpanImpl, "__init__", original_init)


def test_span_exit_swallows_error_log_failure(with_memory_logger, monkeypatch, caplog):
    """Span.__exit__ must not raise when an exception is active even if
    logging the error itself fails."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    call_count = {"n": 0}
    original = SpanImpl.log_internal

    def _maybe_boom(self, *args, **kwargs):
        call_count["n"] += 1
        # First call (during __init__) succeeds. Subsequent calls (log_internal
        # inside __exit__ and end()) blow up.
        if call_count["n"] > 1:
            raise RuntimeError(BOOM)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SpanImpl, "log_internal", _maybe_boom)

    # Customer's exception should propagate; SDK's error-logging failure must not.
    with pytest.raises(ValueError, match="customer error"):
        with start_span("outer", set_current=False):
            raise ValueError("customer error")


# ---------------------------------------------------------------------------
# Logger.log / Logger.start_span
# ---------------------------------------------------------------------------


def test_logger_log_swallows_internal_exception(with_memory_logger, sabotage_log_internal, caplog):
    """Logger.log() must not propagate errors from internal span creation."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    l = logger.current_logger()
    assert l is not None
    result = l.log(input="hello", output="world")  # must not raise
    # The method should still return a value (span id or empty string).
    assert result is None or isinstance(result, str)


def test_logger_start_span_returns_noop_on_failure(with_memory_logger, monkeypatch, caplog):
    """Logger.start_span() must return a usable span even if creation fails."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(self, *args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger.Logger, "_start_span_impl", _boom)

    l = logger.current_logger()
    assert l is not None
    span = l.start_span("s")  # must not raise
    assert span is not None
    with span:
        span.log(output="ignored")  # must not raise


# ---------------------------------------------------------------------------
# Top-level start_span
# ---------------------------------------------------------------------------


def test_top_level_start_span_returns_noop_on_failure(with_memory_logger, monkeypatch, caplog):
    """Top-level braintrust.start_span must never raise."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "get_span_parent_object", _boom)

    span = start_span("s")  # must not raise
    with span:
        span.log(output="ignored")  # must not raise


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def test_traced_runs_user_function_when_span_creation_fails(with_memory_logger, monkeypatch, caplog):
    """If span creation fails inside @traced, the user function must still run."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "start_span", _boom)

    @logger.traced
    def add(a, b):
        return a + b

    assert add(2, 3) == 5  # must return result even though tracing blew up


def test_traced_async_runs_user_function_when_span_creation_fails(with_memory_logger, monkeypatch, caplog):
    """Async @traced must behave identically to sync when tracing fails."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "start_span", _boom)

    @logger.traced
    async def mul(a, b):
        return a * b

    assert asyncio.run(mul(4, 5)) == 20


def test_traced_swallows_log_input_output_errors(with_memory_logger, monkeypatch, caplog):
    """Errors logging input/output must not crash the wrapped function."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    # Break both helpers; the wrapped function should still return.
    monkeypatch.setattr(logger, "_try_log_input", _boom)
    monkeypatch.setattr(logger, "_try_log_output", _boom)

    @logger.traced
    def greet(name):
        return f"hi {name}"

    assert greet("matt") == "hi matt"


def test_traced_sync_generator_completes_when_tracing_fails(with_memory_logger, monkeypatch, caplog):
    """Sync generator @traced must yield all values even when tracing fails."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "start_span", _boom)

    @logger.traced
    def counts():
        yield 1
        yield 2
        yield 3

    assert list(counts()) == [1, 2, 3]


def test_traced_async_generator_completes_when_tracing_fails(with_memory_logger, monkeypatch, caplog):
    """Async generator @traced must yield all values even when tracing fails."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    def _boom(*args, **kwargs):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(logger, "start_span", _boom)

    @logger.traced
    async def counts():
        yield 1
        yield 2
        yield 3

    async def consume():
        out = []
        async for v in counts():
            out.append(v)
        return out

    assert asyncio.run(consume()) == [1, 2, 3]


def test_traced_preserves_user_exceptions(with_memory_logger, caplog):
    """When the user's function raises, @traced must re-raise their original
    exception — it must not swallow customer errors, only SDK errors."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    class MyError(Exception):
        pass

    @logger.traced
    def will_fail():
        raise MyError("from user code")

    with pytest.raises(MyError, match="from user code"):
        will_fail()


# ---------------------------------------------------------------------------
# Exploding __repr__ / bt_safe_deep_copy resilience
# ---------------------------------------------------------------------------


def test_span_log_with_exploding_repr_does_not_raise(with_memory_logger, caplog):
    """User objects whose __repr__ / __str__ raise must not crash span.log."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    with start_span("outer", set_current=False) as span:
        span.log(input=ExplodingRepr(), output=ExplodingRepr())  # must not raise


def test_traced_with_exploding_repr_does_not_raise(with_memory_logger, caplog):
    """A traced function returning an exploding object must still return the
    value to the caller; only the logging step should be affected."""
    init_test_logger(__name__)
    caplog.set_level(logging.WARNING, logger="braintrust")

    @logger.traced
    def build():
        return ExplodingRepr()

    result = build()  # must not raise
    assert isinstance(result, ExplodingRepr)
