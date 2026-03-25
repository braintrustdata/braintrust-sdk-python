"""Benchmarks for SpanComponentsV3 and SpanComponentsV4 encode/decode.

These are on the hot path: every span serializes/deserializes parent context.
"""

import pathlib
import secrets
import sys
import uuid

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.span_identifier_v3 import SpanComponentsV3, SpanObjectTypeV3
from braintrust.span_identifier_v4 import SpanComponentsV4

from benchmarks._utils import disable_pyperf_psutil


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    # V3 — UUID-based IDs
    v3_obj_only = SpanComponentsV3(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id=str(uuid.uuid4()),
    )
    v3_full = SpanComponentsV3(
        object_type=SpanObjectTypeV3.EXPERIMENT,
        object_id=str(uuid.uuid4()),
        row_id=str(uuid.uuid4()),
        span_id=str(uuid.uuid4()),
        root_span_id=str(uuid.uuid4()),
    )
    v3_obj_only_str = v3_obj_only.to_str()
    v3_full_str = v3_full.to_str()

    runner.bench_func("span_components.v3.to_str[object-only]", v3_obj_only.to_str)
    runner.bench_func("span_components.v3.to_str[full-uuid]", v3_full.to_str)
    runner.bench_func("span_components.v3.from_str[object-only]", SpanComponentsV3.from_str, v3_obj_only_str)
    runner.bench_func("span_components.v3.from_str[full-uuid]", SpanComponentsV3.from_str, v3_full_str)

    # V4 — OTEL hex IDs for span_id (8-byte) and root_span_id (16-byte)
    v4_obj_only = SpanComponentsV4(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id=str(uuid.uuid4()),
    )
    v4_full_otel = SpanComponentsV4(
        object_type=SpanObjectTypeV3.EXPERIMENT,
        object_id=str(uuid.uuid4()),
        row_id=str(uuid.uuid4()),
        span_id=secrets.token_hex(8),
        root_span_id=secrets.token_hex(16),
    )
    v4_obj_only_str = v4_obj_only.to_str()
    v4_full_otel_str = v4_full_otel.to_str()

    runner.bench_func("span_components.v4.to_str[object-only]", v4_obj_only.to_str)
    runner.bench_func("span_components.v4.to_str[full-otel]", v4_full_otel.to_str)
    runner.bench_func("span_components.v4.from_str[object-only]", SpanComponentsV4.from_str, v4_obj_only_str)
    runner.bench_func("span_components.v4.from_str[full-otel]", SpanComponentsV4.from_str, v4_full_otel_str)

    # Cross-version: V4 decoder reading a V3-encoded string (backwards-compat path)
    runner.bench_func("span_components.v4.from_str[v3-encoded]", SpanComponentsV4.from_str, v3_full_str)


if __name__ == "__main__":
    main()
