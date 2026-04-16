"""Overhead of the _safe decorator.

The _safe decorator wraps public tracing methods so exceptions never reach
customer code. This benchmark isolates its happy-path cost by comparing a
raw function vs. the same function wrapped with _safe, across call shapes
representative of tracing hot paths (no args, positional args, kwargs, and
a cheap "log-like" call).

The absolute numbers matter less than the _safe/raw delta: that delta is
pure decorator overhead. The delta should stay well under a microsecond
per call.
"""

import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.logger import _safe

from benchmarks._utils import disable_pyperf_psutil


def _noop() -> None:
    return None


def _add(a: int, b: int) -> int:
    return a + b


def _log_like(**event: object) -> None:
    # Cheap stand-in for Span.log: dict access patterns without serialization.
    _ = event.get("input"), event.get("output"), event.get("metadata")
    return None


_noop_safe = _safe(_noop)
_add_safe = _safe(_add)
_log_like_safe = _safe(_log_like)


def _bench_raw_noop() -> None:
    _noop()


def _bench_safe_noop() -> None:
    _noop_safe()


def _bench_raw_add() -> None:
    _add(1, 2)


def _bench_safe_add() -> None:
    _add_safe(1, 2)


def _bench_raw_log_like() -> None:
    _log_like(input={"x": 1}, output={"y": 2}, metadata={"z": 3})


def _bench_safe_log_like() -> None:
    _log_like_safe(input={"x": 1}, output={"y": 2}, metadata={"z": 3})


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("safe_decorator.noop[raw]", _bench_raw_noop)
    runner.bench_func("safe_decorator.noop[safe]", _bench_safe_noop)
    runner.bench_func("safe_decorator.add[raw]", _bench_raw_add)
    runner.bench_func("safe_decorator.add[safe]", _bench_safe_add)
    runner.bench_func("safe_decorator.log_like[raw]", _bench_raw_log_like)
    runner.bench_func("safe_decorator.log_like[safe]", _bench_safe_log_like)


if __name__ == "__main__":
    main()
