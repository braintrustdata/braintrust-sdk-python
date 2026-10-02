"""Preparation and producer cost, independent of HTTP server latency.

Use benchmarks.log_ingestion for end-to-end healthy/throttled measurements.
"""

import contextlib
import os
import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.logger import _HTTPBackgroundLogger, construct_logs3_data, stringify_with_overflow_meta
from braintrust.util import LazyValue

from benchmarks._utils import disable_pyperf_psutil
from benchmarks.fixtures import ingestion_rows


_ROWS = ingestion_rows()
_RECORDS = [LazyValue(lambda row=row: row, use_mutex=False) for row in _ROWS]


def _prepare() -> None:
    construct_logs3_data([stringify_with_overflow_meta(row) for row in _ROWS]).encode("utf-8")


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()
    # Pause the real writer's publisher to isolate producer work; no network calls occur.
    os.environ["BRAINTRUST_DISABLE_ATEXIT_FLUSH"] = "1"
    writer = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(None))
    writer.sync_flush = True
    writer._start()

    def enqueue() -> None:
        writer.log(*_RECORDS)
        writer.queue.drain_all()

    runner.bench_func("logs3[prepare-100-rows]", _prepare)
    runner.bench_func("logs3[enqueue-100-rows]", enqueue)


if __name__ == "__main__":
    main()
