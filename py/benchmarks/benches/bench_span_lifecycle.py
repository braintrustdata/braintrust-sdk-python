"""Benchmarks for the span creation / log / end lifecycle.

These measure pure SDK overhead per span — no network, no I/O.  A discard
logger replaces the HTTP background logger so log() calls enqueue a
LazyValue that is immediately thrown away, isolating the in-process cost.

Scenarios
---------
* noop_span          — NoopSpan.log() + end(): absolute zero-overhead floor.
* root_create_end    — SpanImpl init + end, root span, no payload.
* root_log_small     — SpanImpl.log() with a small flat payload dict.
* root_log_medium    — SpanImpl.log() with a medium nested payload.
* child_create_end   — start_span() + child.end() from an existing root span.
* export             — span.export() (SpanComponentsV3 serialisation).
"""

import pathlib
import sys
import uuid

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Import internals used by the test suite for mocking.
from braintrust.logger import (
    NOOP_SPAN,
    BraintrustState,
    SpanImpl,
    _MemoryBackgroundLogger,
)
from braintrust.span_identifier_v3 import SpanObjectTypeV3
from braintrust.util import LazyValue

from benchmarks._utils import disable_pyperf_psutil
from benchmarks.fixtures import make_medium_payload, make_small_payload


# ---------------------------------------------------------------------------
# One-time setup: a BraintrustState wired to a discard logger so span logs
# never touch the network or accumulate unboundedly in memory.
# ---------------------------------------------------------------------------


class _DiscardLogger(_MemoryBackgroundLogger):
    """Drop all log items immediately; never accumulate memory."""

    def log(self, *args: LazyValue) -> None:  # type: ignore[override]
        pass  # intentionally discard — we measure enqueue cost, not storage


_state = BraintrustState()
_state._override_bg_logger.logger = _DiscardLogger()

# Pre-resolved LazyValue for parent_object_id — avoids lazy resolution cost
# inside the hot loop (same UUID every iteration; fine for benchmarking).
_OBJECT_ID = str(uuid.uuid4())
_PARENT_OBJECT_ID: LazyValue[str] = LazyValue(lambda: _OBJECT_ID, use_mutex=False)
_PARENT_OBJECT_ID.get()  # resolve once so subsequent .get() calls are O(1)

_PARENT_OBJECT_TYPE = SpanObjectTypeV3.EXPERIMENT

# Payloads reused across log() benchmarks.
_SMALL_PAYLOAD = make_small_payload()
_MEDIUM_PAYLOAD = make_medium_payload()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _make_root_span() -> SpanImpl:
    return SpanImpl(
        parent_object_type=_PARENT_OBJECT_TYPE,
        parent_object_id=_PARENT_OBJECT_ID,
        parent_compute_object_metadata_args=None,
        parent_span_ids=None,
        name="bench-root",
        state=_state,
    )


def _make_child_span(root: SpanImpl) -> SpanImpl:
    return root.start_span(name="bench-child")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _bench_noop_span() -> None:
    NOOP_SPAN.log(input="x", output="y")
    NOOP_SPAN.end()


def _bench_root_create_end() -> None:
    span = _make_root_span()
    span.end()


def _bench_root_log_small() -> None:
    span = _make_root_span()
    span.log(**_SMALL_PAYLOAD)
    span.end()


def _bench_root_log_medium() -> None:
    span = _make_root_span()
    span.log(**_MEDIUM_PAYLOAD)
    span.end()


def _bench_child_create_end() -> None:
    root = _make_root_span()
    child = _make_child_span(root)
    child.end()
    root.end()


def _bench_export() -> None:
    span = _make_root_span()
    span.export()
    span.end()


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("span_lifecycle.noop[log+end]", _bench_noop_span)
    runner.bench_func("span_lifecycle.root[create+end]", _bench_root_create_end)
    runner.bench_func("span_lifecycle.root[log-small+end]", _bench_root_log_small)
    runner.bench_func("span_lifecycle.root[log-medium+end]", _bench_root_log_medium)
    runner.bench_func("span_lifecycle.child[create+end]", _bench_child_create_end)
    runner.bench_func("span_lifecycle.root[export]", _bench_export)


if __name__ == "__main__":
    main()
