"""Tests for SpanCache (disk-based cache)."""

import io
import json
import sys
from pathlib import Path

import pytest
from braintrust.span_cache import CachedSpan, SpanCache


def test_span_cache_write_and_read():
    """Test storing and retrieving spans by rootSpanId."""
    cache = SpanCache()
    cache.start()  # Start for testing (cache is disabled by default)

    root_span_id = "root-123"
    span1 = CachedSpan(
        span_id="span-1",
        input={"text": "hello"},
        output={"response": "world"},
        error={"message": "retryable"},
        metrics={"start": 1, "end": 3},
        tags=["production"],
    )
    span2 = CachedSpan(
        span_id="span-2",
        input={"text": "foo"},
        output={"response": "bar"},
    )

    cache.queue_write(root_span_id, span1.span_id, span1)
    cache.queue_write(root_span_id, span2.span_id, span2)

    spans = cache.get_by_root_span_id(root_span_id)
    assert spans is not None
    assert len(spans) == 2

    span_ids = {s.span_id for s in spans}
    assert "span-1" in span_ids
    assert "span-2" in span_ids
    stored_span1 = next(span for span in spans if span.span_id == "span-1")
    assert stored_span1.error == {"message": "retryable"}
    assert stored_span1.metrics == {"start": 1, "end": 3}
    assert stored_span1.tags == ["production"]

    cache.stop()
    cache.dispose()


def test_span_cache_return_none_for_unknown():
    """Test that unknown rootSpanId returns None."""
    cache = SpanCache()
    cache.start()

    spans = cache.get_by_root_span_id("nonexistent")
    assert spans is None

    cache.stop()
    cache.dispose()


def test_span_cache_merge_on_duplicate_writes():
    """Test that subsequent writes to same spanId merge data."""
    cache = SpanCache()
    cache.start()

    root_span_id = "root-123"
    span_id = "span-1"

    cache.queue_write(
        root_span_id,
        span_id,
        CachedSpan(span_id=span_id, input={"text": "hello"}),
    )

    cache.queue_write(
        root_span_id,
        span_id,
        CachedSpan(span_id=span_id, output={"response": "world"}),
    )

    spans = cache.get_by_root_span_id(root_span_id)
    assert spans is not None
    assert len(spans) == 1
    assert spans[0].span_id == span_id
    assert spans[0].input == {"text": "hello"}
    assert spans[0].output == {"response": "world"}

    cache.stop()
    cache.dispose()


def test_span_cache_merge_metadata():
    """Test that metadata objects are merged."""
    cache = SpanCache()
    cache.start()

    root_span_id = "root-123"
    span_id = "span-1"

    cache.queue_write(
        root_span_id,
        span_id,
        CachedSpan(span_id=span_id, metadata={"key1": "value1"}),
    )

    cache.queue_write(
        root_span_id,
        span_id,
        CachedSpan(span_id=span_id, metadata={"key2": "value2"}),
    )

    spans = cache.get_by_root_span_id(root_span_id)
    assert spans is not None
    assert spans[0].metadata == {"key1": "value1", "key2": "value2"}

    cache.stop()
    cache.dispose()


def test_span_cache_has():
    """Test the has() method."""
    cache = SpanCache()
    cache.start()

    cache.queue_write("root-123", "span-1", CachedSpan(span_id="span-1"))
    assert cache.has("root-123") is True
    assert cache.has("nonexistent") is False

    cache.stop()
    cache.dispose()


def test_span_cache_clear():
    """Test clearing spans for a specific rootSpanId."""
    cache = SpanCache()
    cache.start()

    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))

    cache.clear("root-1")

    assert cache.has("root-1") is False
    assert cache.has("root-2") is True

    cache.stop()
    cache.dispose()


def test_span_cache_clear_all():
    """Test clearing all cached spans."""
    cache = SpanCache()
    cache.start()

    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))

    cache.clear_all()

    assert cache.size == 0

    cache.stop()
    cache.dispose()


def test_span_cache_size():
    """Test the size property."""
    cache = SpanCache()
    cache.start()

    assert cache.size == 0

    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 1

    cache.queue_write("root-1", "span-2", CachedSpan(span_id="span-2"))  # Same root
    assert cache.size == 1

    cache.queue_write("root-2", "span-3", CachedSpan(span_id="span-3"))  # Different root
    assert cache.size == 2

    cache.stop()
    cache.dispose()


def test_span_cache_dispose():
    """Test that dispose cleans up and allows reuse."""
    cache = SpanCache()
    cache.start()

    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 1

    # Stop first to decrement refcount, then dispose
    cache.stop()
    cache.dispose()

    assert cache.size == 0
    assert cache.has("root-1") is False

    # Should be able to write again after dispose (if we start again)
    cache.start()
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))
    assert cache.size == 1

    cache.stop()
    cache.dispose()


def test_span_cache_disable():
    """Test that disable() prevents writes."""
    cache = SpanCache()
    cache.start()

    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 1

    cache.disable()

    # Writes after disable should be no-ops
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))
    assert cache.size == 1  # Still 1, not 2

    cache.stop()
    cache.dispose()


def test_span_cache_disabled_getter():
    """Test the disabled property."""
    # Cache is disabled by default until start() is called
    cache = SpanCache()
    assert cache.disabled is True

    cache.start()
    assert cache.disabled is False

    cache.disable()
    assert cache.disabled is True

    cache.dispose()


def test_span_cache_disabled_from_constructor():
    """Test that cache can be disabled via constructor."""
    cache = SpanCache(disabled=True)
    assert cache.disabled is True

    # Writes should be no-ops
    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 0
    assert cache.get_by_root_span_id("root-1") is None

    cache.dispose()


def test_span_cache_start_stop_lifecycle():
    """Test that stop() allows start() to work again."""
    cache = SpanCache()

    # Initially disabled by default
    assert cache.disabled is True

    # Start for first "eval"
    cache.start()
    assert cache.disabled is False
    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 1

    # Stop after first "eval"
    cache.stop()
    cache.dispose()
    assert cache.disabled is True

    # Start for second "eval" - should work!
    cache.start()
    assert cache.disabled is False
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))
    assert cache.size == 1

    cache.stop()
    cache.dispose()


def test_span_cache_disable_prevents_start():
    """Test that disable() prevents start() from working."""
    cache = SpanCache()

    # Simulate disable being called
    cache.disable()
    assert cache.disabled is True

    # start() should be a no-op after disable()
    cache.start()
    assert cache.disabled is True

    # Writes should still be no-ops
    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 0

    cache.dispose()


def test_span_cache_parallel_eval_refcount():
    """Test reference counting for parallel evals."""
    cache = SpanCache()

    # Simulate two evals starting
    cache.start()  # Eval 1
    assert cache.disabled is False

    cache.start()  # Eval 2
    assert cache.disabled is False

    # Write data from both evals
    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    cache.queue_write("root-2", "span-2", CachedSpan(span_id="span-2"))
    assert cache.size == 2

    # Eval 1 finishes first
    cache.dispose()  # Should NOT dispose (refcount = 2)
    cache.stop()  # Decrements to 1

    # Cache should still be enabled and data intact
    assert cache.disabled is False
    assert cache.size == 2
    assert cache.get_by_root_span_id("root-1") is not None
    assert cache.get_by_root_span_id("root-2") is not None

    # Eval 2 finishes
    cache.dispose()  # Should NOT dispose yet (refcount = 1)
    cache.stop()  # Decrements to 0, disables cache

    # Now cache should be disabled
    assert cache.disabled is True

    # Final dispose should now work
    cache.dispose()  # NOW it disposes (refcount = 0)
    assert cache.size == 0


def test_span_cache_refcount_underflow():
    """Test that refcount handles underflow gracefully."""
    cache = SpanCache()

    # Call stop without start
    cache.stop()

    # Should work normally after
    cache.start()
    cache.queue_write("root-1", "span-1", CachedSpan(span_id="span-1"))
    assert cache.size == 1

    cache.stop()
    cache.dispose()


def test_span_cache_serialization_retries_and_retains_unflushed_data():
    import json
    from pathlib import Path

    attempts = []

    class RetryDict(dict):
        def items(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("first serialization fails")
            return super().items()

    cache = SpanCache()
    cache.start()
    try:
        cache.queue_write("root", "first", CachedSpan(span_id="first", metadata={"nested": RetryDict(value=1)}))
        assert cache.get_by_root_span_id("root")[0].metadata["nested"] == {"value": 1}
        assert cache._write_buffer
        cache.queue_write("root", "second", CachedSpan(span_id="second"))
        rows = [json.loads(line) for line in Path(cache._cache_file_path).read_text().splitlines()]
        assert [row["spanId"] for row in rows] == ["first", "second"]
        assert len(attempts) == 2
        assert not cache._write_buffer
    finally:
        cache.stop()
        cache.dispose()


def test_span_cache_write_error_retains_data_for_read_and_retry():
    import json
    from pathlib import Path

    cache = SpanCache()
    cache.start()
    try:
        cache.queue_write("root", "initial", CachedSpan(span_id="initial"))
        path = Path(cache._cache_file_path)
        path.unlink()
        path.mkdir()
        cache.queue_write("root", "retry", CachedSpan(span_id="retry", output="retained"))
        assert cache.get_by_root_span_id("root")[0].output == "retained"
        assert cache._write_buffer
        path.rmdir()
        path.touch()
        cache.queue_write("root", "last", CachedSpan(span_id="last"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["spanId"] for row in rows] == ["retry", "last"]
        assert not cache._write_buffer
    finally:
        cache.stop()
        cache.dispose()


def test_span_cache_serialization_can_disable_cache():
    cache = SpanCache()

    class DisableDict(dict):
        def items(self):
            cache.disable()
            return super().items()

    cache.start()
    try:
        cache.queue_write("root", "span", CachedSpan(span_id="span", metadata={"nested": DisableDict(value=1)}))
        assert cache.disabled
    finally:
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("remove_file", [False, True])
def test_span_cache_serialization_cannot_write_into_next_evaluation(remove_file):
    import threading
    from pathlib import Path

    cache = SpanCache()
    entered = threading.Event()
    resume = threading.Event()
    timed_out = []
    errors = []

    class BlockingDict(dict):
        def items(self):
            if not entered.is_set():
                entered.set()
                if not resume.wait(5):
                    timed_out.append(True)
            return super().items()

    def write_old():
        try:
            cache.queue_write("old", "old", CachedSpan(span_id="old", metadata={"nested": BlockingDict(value=1)}))
        except BaseException as exc:
            errors.append(exc)

    cache.start()
    writer = threading.Thread(target=write_old, daemon=True)
    writer.start()
    try:
        assert entered.wait(5)
        old_path = Path(cache._cache_file_path)
        if remove_file:
            old_path.unlink()
        cache.stop()
        cache.dispose()
        cache.start()
        resume.set()
        writer.join(5)
        assert not writer.is_alive()
        assert not timed_out
        assert not errors
        assert not old_path.exists()
        cache.queue_write("new", "new", CachedSpan(span_id="new", output="new-eval"))
        new_path = Path(cache._cache_file_path)
        assert new_path != old_path
        assert new_path.is_file()
        assert not cache.has("old")
        assert cache.get_by_root_span_id("old") is None
        assert [span.output for span in cache.get_by_root_span_id("new")] == ["new-eval"]
        assert '"rootSpanId": "old"' not in new_path.read_text()
        assert not cache._write_buffer
    finally:
        resume.set()
        writer.join(5)
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("mode", ["w", "a"])
@pytest.mark.parametrize("action", ["write", "restart", "disable"])
def test_span_cache_io_callback_preserves_generation(mode, action):
    import json
    import sys
    from pathlib import Path

    cache = SpanCache()
    calls = []
    active = [True]
    old_paths = []

    def audit(event, args):
        if not active[0] or event != "open" or calls:
            return
        path, open_mode, *_ = args
        if not isinstance(path, str) or "braintrust-span-cache-" not in path or open_mode != mode:
            return
        calls.append("entered")
        old_paths.append(Path(path))
        if action == "restart":
            cache.stop()
            cache.dispose()
            cache.start()
        elif action == "disable":
            cache.disable()
        cache.queue_write("callback", "callback", CachedSpan(span_id="callback", output=action))
        calls.append("returned")

    sys.addaudithook(audit)
    cache.start()
    try:
        cache.queue_write("outer", "outer", CachedSpan(span_id="outer"))
        assert calls == ["entered", "returned"]
        if action == "disable":
            assert cache.disabled
        else:
            assert [span.output for span in cache.get_by_root_span_id("callback")] == [action]
            assert cache._flush_in_progress is None
            cache._flush_write_buffer()
            path = Path(cache._cache_file_path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [row["spanId"] for row in rows] == (["callback"] if action == "restart" else ["outer", "callback"])
            assert not cache._write_buffer
            assert cache._flush_in_progress is None
            if action == "restart":
                assert path != old_paths[0]
                assert not old_paths[0].exists()
                assert cache.get_by_root_span_id("outer") is None
    finally:
        active[0] = False
        cache.stop()
        cache.dispose()
    assert not old_paths[0].exists()


def test_span_cache_unlink_callback_can_start_next_generation():
    import sys
    from pathlib import Path

    cache = SpanCache()
    cache.start()
    cache.queue_write("old", "old", CachedSpan(span_id="old"))
    old_path = Path(cache._cache_file_path)
    active = [True]
    calls = []

    def audit(event, args):
        if active[0] and event == "os.remove" and args[0] == str(old_path) and not calls:
            calls.append("entered")
            cache.start()
            cache.queue_write("new", "new", CachedSpan(span_id="new", output="survived"))
            calls.append("returned")

    sys.addaudithook(audit)
    try:
        cache.stop()
        cache.dispose()
        assert calls == ["entered", "returned"]
        new_path = Path(cache._cache_file_path)
        assert new_path != old_path and new_path.is_file()
        assert not old_path.exists()
        assert cache._active_eval_count == 1
        assert [span.output for span in cache.get_by_root_span_id("new")] == ["survived"]
        assert cache.get_by_root_span_id("old") is None
    finally:
        active[0] = False
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("mode", ["w", "a"])
def test_span_cache_concurrent_initialization_disposal(mode):
    import sys
    import threading
    from pathlib import Path

    cache = SpanCache()
    entered = threading.Event()
    resume = threading.Event()
    errors = []
    old_paths = []
    active = [True]

    def audit(event, args):
        if not active[0] or event != "open" or entered.is_set():
            return
        path, open_mode, *_ = args
        if isinstance(path, str) and "braintrust-span-cache-" in path and open_mode == mode:
            old_paths.append(Path(path))
            entered.set()
            if not resume.wait(5):
                errors.append("I/O retained lifecycle ownership")

    def write_old():
        try:
            cache.queue_write("old", "old", CachedSpan(span_id="old"))
        except BaseException as exc:
            errors.append(exc)

    sys.addaudithook(audit)
    cache.start()
    writer = threading.Thread(target=write_old, daemon=True)
    writer.start()
    try:
        assert entered.wait(5)
        cache.stop()
        cache.dispose()
        cache.start()
        cache.queue_write("new", "new", CachedSpan(span_id="new"))
        resume.set()
        writer.join(5)
        assert not writer.is_alive() and not errors
        assert [span.span_id for span in cache.get_by_root_span_id("new")] == ["new"]
        assert cache._flush_in_progress is None
        cache._flush_write_buffer()
        new_path = Path(cache._cache_file_path)
        assert new_path.is_file() and new_path != old_paths[0]
        assert not old_paths[0].exists()
        assert [span.span_id for span in cache.get_by_root_span_id("new")] == ["new"]
        assert cache.get_by_root_span_id("old") is None
        assert not cache._write_buffer
    finally:
        active[0] = False
        resume.set()
        writer.join(5)
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("action", ["restart", "error"])
def test_span_cache_io_callback_during_physical_write(action):
    import json
    import sys
    from pathlib import Path

    cache = SpanCache()
    active = [True]
    calls = []
    old_paths = []
    previous_profile = sys.getprofile()

    def profile(frame, event, function):
        if active[0] and event == "c_call" and getattr(function, "__name__", None) == "writelines" and not calls:
            calls.append("entered")
            old_paths.append(Path(cache._cache_file_path))
            if action == "error":
                raise OSError("instrumented write failed")
            cache.stop()
            cache.dispose()
            cache.start()
            cache.queue_write("new", "new", CachedSpan(span_id="new"))
            calls.append("returned")

    cache.start()
    sys.setprofile(profile)
    try:
        cache.queue_write("old", "old", CachedSpan(span_id="old", output="retained"))
        sys.setprofile(previous_profile)
        if action == "error":
            assert calls == ["entered"]
            assert cache._write_buffer
            assert [span.output for span in cache.get_by_root_span_id("old")] == ["retained"]
            cache.queue_write("new", "new", CachedSpan(span_id="new"))
        else:
            assert calls == ["entered", "returned"]
            assert not old_paths[0].exists()
            assert cache.get_by_root_span_id("old") is None
            assert [span.span_id for span in cache.get_by_root_span_id("new")] == ["new"]
            assert cache._flush_in_progress is None
            cache._flush_write_buffer()
            assert Path(cache._cache_file_path) != old_paths[0]
        rows = [json.loads(line) for line in Path(cache._cache_file_path).read_text().splitlines()]
        assert [row["spanId"] for row in rows] == (["old", "new"] if action == "error" else ["new"])
        assert not cache._write_buffer and cache._flush_in_progress is None
    finally:
        active[0] = False
        sys.setprofile(previous_profile)
        cache.stop()
        cache.dispose()


import io
import json
import sys
from pathlib import Path

from braintrust import EvalAsync
from braintrust.test_helpers import init_test_exp, with_memory_logger  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["open", "writelines", "__exit__"])
async def test_every_native_io_callback_retains_real_spans(operation, with_memory_logger):
    exp = init_test_exp("bounded-" + operation)
    cache = exp.state.span_cache
    cache.start()  # Keep the cache alive to inspect records after EvalAsync releases it.
    state = {"active": True, "inside": False}
    emitted = []
    old_profile = sys.getprofile()

    def emit():
        if not state["active"] or state["inside"]:
            return
        state["inside"] = True
        try:
            with exp.start_span(name="every-native-" + operation) as span:
                span.log(output="instrumented-" + operation)
                emitted.append((span.root_span_id, span.span_id))
            if len(emitted) == 1 or len(emitted) % 1000 == 0:
                print("NATIVE_CALLBACK", operation, len(emitted), flush=True)
        finally:
            state["inside"] = False

    def audit(event, args):
        if operation == "open" and event == "open":
            path, mode, *_ = args
            if isinstance(path, str) and "braintrust-span-cache-" in path and mode == "a":
                emit()

    def profile(frame, event, function):
        if event != "c_call":
            return
        owner = getattr(function, "__self__", None)
        if event == "c_call" and getattr(function, "__name__", None) == operation:
            if (
                isinstance(owner, io.TextIOWrapper)
                and owner.mode == "a"
                and "braintrust-span-cache-" in str(owner.name)
            ):
                emit()

    sys.addaudithook(audit)
    if operation != "open":
        sys.setprofile(profile)
    try:

        async def score(trace):
            spans = await trace.get_spans()
            assert spans
            return 1

        result = await EvalAsync(
            "bounded",
            data=[{"input": 7}],
            task=lambda input: "root-output",
            scores=[score],
            state=exp.state,
            no_send_logs=True,
        )
        state["active"] = False
        sys.setprofile(old_profile)
        assert result.results[0].output == "root-output" and result.results[0].error is None
        assert len(emitted) > 1
        with_memory_logger.pop()
        for root, identity in emitted:
            spans = cache.get_by_root_span_id(root)
            assert spans and any(s.span_id == identity and s.output == "instrumented-" + operation for s in spans)
        assert cache._flush_in_progress is None
        cache._flush_write_buffer()
        assert not cache._write_buffer
        rows = [json.loads(line) for line in Path(cache._cache_file_path).read_text().splitlines()]
        persisted = {row["spanId"] for row in rows}
        assert {identity for _, identity in emitted} <= persisted
        assert any(row["data"].get("output") == "root-output" for row in rows)
        print(
            "NATIVE_CALLBACK_COMPLETE",
            operation,
            len(emitted),
            "all records readable and persisted on retry",
            flush=True,
        )
    finally:
        state["active"] = False
        sys.setprofile(old_profile)
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("operation", ["w", "a", "writelines", "__exit__"])
def test_every_restart_callback_retains_replacement_for_next_flush(operation):
    cache = SpanCache()
    cache.start()
    state = {"active": True, "inside": False}
    old_paths = []
    old_profile = sys.getprofile()

    def restart(path):
        if not state["active"] or state["inside"]:
            return
        state["inside"] = True
        try:
            old_paths.append(Path(path))
            cache.stop()
            cache.dispose()
            cache.start()
            cache.queue_write("replacement", "replacement", CachedSpan("replacement", output="retained"))
        finally:
            state["inside"] = False

    def audit(event, args):
        if operation in ("w", "a") and event == "open":
            path, mode, *_ = args
            if isinstance(path, str) and "braintrust-span-cache-" in path and mode == operation:
                restart(path)

    def profile(frame, event, function):
        if event != "c_call":
            return
        owner = getattr(function, "__self__", None)
        if event == "c_call" and getattr(function, "__name__", None) == operation:
            if (
                isinstance(owner, io.TextIOWrapper)
                and owner.mode == "a"
                and "braintrust-span-cache-" in str(owner.name)
            ):
                restart(owner.name)

    sys.addaudithook(audit)
    if operation not in ("w", "a"):
        sys.setprofile(profile)
    try:
        cache.queue_write("retired", "retired", CachedSpan("retired"))
        state["active"] = False
        sys.setprofile(old_profile)
        assert len(old_paths) == 1
        assert cache._flush_in_progress is None
        assert not old_paths[0].exists()
        assert cache.get_by_root_span_id("retired") is None
        assert [s.output for s in cache.get_by_root_span_id("replacement")] == ["retained"]
        assert len(cache._write_buffer) == 1
        cache._flush_write_buffer()
        assert not cache._write_buffer and cache._flush_in_progress is None
        rows = [json.loads(line) for line in Path(cache._cache_file_path).read_text().splitlines()]
        assert [row["spanId"] for row in rows] == ["replacement"]
        assert Path(cache._cache_file_path) != old_paths[0]
    finally:
        state["active"] = False
        sys.setprofile(old_profile)
        cache.stop()
        cache.dispose()


@pytest.mark.parametrize("mode", ["w", "a"])
@pytest.mark.parametrize("inject_close_error", [False, True])
def test_retirement_followed_by_close_error_removes_old_path(mode, inject_close_error):
    cache = SpanCache()
    cache.start()
    state = {"active": True, "inside": False}
    retired = []
    close_errors = []
    previous = sys.getprofile()

    def audit(event, args):
        if not state["active"] or state["inside"] or event != "open":
            return
        path, opened, *_ = args
        if opened != mode or not isinstance(path, str) or "braintrust-span-cache-" not in path:
            return
        state["inside"] = True
        try:
            retired.append(Path(path))
            cache.stop()
            cache.dispose()
            cache.start()
            cache.queue_write("replacement", "replacement", CachedSpan("replacement", output="survives"))
        finally:
            state["inside"] = False

    def profile(frame, event, function):
        if event != "c_call" or getattr(function, "__name__", None) != "__exit__":
            return
        owner = getattr(function, "__self__", None)
        if (
            inject_close_error
            and state["active"]
            and isinstance(owner, io.TextIOWrapper)
            and retired
            and str(owner.name) == str(retired[-1])
        ):
            close_errors.append(owner.name)
            raise OSError("close error after generation retirement")

    sys.addaudithook(audit)
    sys.setprofile(profile)
    try:
        cache.queue_write("old", "old", CachedSpan("old"))
        state["active"] = False
        sys.setprofile(previous)
        assert retired and bool(close_errors) == inject_close_error
        assert cache._flush_in_progress is None and not cache.disabled
        assert [s.output for s in cache.get_by_root_span_id("replacement")] == ["survives"]
        cache._flush_write_buffer()
        assert not cache._write_buffer
        assert [json.loads(line)["spanId"] for line in Path(cache._cache_file_path).read_text().splitlines()] == [
            "replacement"
        ]
        print("RETIRED_CLOSE_ERROR", mode, "retired=", str(retired[0]), "exists=", retired[0].exists(), flush=True)
        assert not retired[0].exists(), "Retired path recreated by open survives close exception"
    finally:
        state["active"] = False
        sys.setprofile(previous)
        cache.stop()
        cache.dispose()
