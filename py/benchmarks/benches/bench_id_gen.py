"""Benchmarks for ID generation.

get_span_id and get_trace_id are called on every span creation, so their
cost accumulates in high-throughput tracing workloads.  This module
compares the two generators: UUIDGenerator (default) and OTELIDGenerator
(enabled via BRAINTRUST_OTEL_COMPAT=true).
"""

import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.id_gen import OTELIDGenerator, UUIDGenerator

from benchmarks._utils import disable_pyperf_psutil


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    uuid_gen = UUIDGenerator()
    otel_gen = OTELIDGenerator()

    runner.bench_func("id_gen.uuid.span_id", uuid_gen.get_span_id)
    runner.bench_func("id_gen.uuid.trace_id", uuid_gen.get_trace_id)
    runner.bench_func("id_gen.otel.span_id", otel_gen.get_span_id)
    runner.bench_func("id_gen.otel.trace_id", otel_gen.get_trace_id)


if __name__ == "__main__":
    main()
