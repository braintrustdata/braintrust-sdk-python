"""End-to-end CPU time benchmark + cProfile analysis for tracing.

Usage:
  python bench_e2e.py           # benchmark only
  python bench_e2e.py --profile # benchmark + cProfile breakdown
"""

import cProfile
import os
import pstats
import sys
import time

os.environ["BRAINTRUST_DISABLE_ATEXIT_FLUSH"] = "true"
sys.path.insert(0, "src")

from braintrust.logger import (
    BraintrustState,
    SpanImpl,
    _MemoryBackgroundLogger,
    SpanObjectTypeV3,
    stringify_with_overflow_meta,
)
from braintrust.merge_row_batch import merge_row_batch
from braintrust.util import LazyValue


def make_state():
    state = BraintrustState()
    ml = _MemoryBackgroundLogger()
    state._override_bg_logger.logger = ml
    pid = LazyValue(lambda: "proj-abc123", use_mutex=False)
    pid.get()
    return state, ml, pid


def run_workload(state, ml, pid, num_requests):
    """Simulate num_requests LLM calls with root + child spans."""
    t_start = time.perf_counter()
    for i in range(num_requests):
        root = SpanImpl(
            parent_object_type=SpanObjectTypeV3.PROJECT_LOGS,
            parent_object_id=pid,
            parent_compute_object_metadata_args=None,
            parent_span_ids=None,
            name="handle_request",
            state=state,
            event={
                "input": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Question {i}: What is {i} + {i}?"},
                    ]
                },
                "metadata": {"user_id": f"user_{i % 100}", "session_id": "sess_abc"},
            },
            lookup_span_parent=False,
        )
        child = root.start_span(
            name="llm_call",
            input={"model": "gpt-4", "temperature": 0.7, "max_tokens": 500},
        )
        child.log(
            output={
                "choices": [
                    {"message": {"role": "assistant", "content": f"The answer is {i * 2}."}}
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
            metrics={"latency": 0.234, "tokens_per_second": 85.5},
        )
        child.end()
        root.log(
            output=f"The answer is {i * 2}.",
            scores={"accuracy": 0.95, "relevance": 0.88},
        )
        root.end()
    t_user = time.perf_counter() - t_start
    return t_user


def run_flush(ml):
    """Simulate the flush path (unwrap lazy values, merge, stringify)."""
    items = ml.logs[:]
    t0 = time.perf_counter()
    unwrapped = [it.get() for it in items]
    merged = merge_row_batch(unwrapped)
    _ = [stringify_with_overflow_meta(m) for m in merged]
    t_flush = time.perf_counter() - t0
    return t_flush, len(items), len(merged)


def benchmark():
    # Warmup
    s, ml, pid = make_state()
    run_workload(s, ml, pid, 10)

    print("End-to-end benchmark")
    print("=" * 70)
    for n in [100, 1000, 5000]:
        s, ml, pid = make_state()
        t_user = run_workload(s, ml, pid, n)
        t_flush, num_items, num_merged = run_flush(ml)
        t_total = t_user + t_flush
        print(
            f"  {n:5d} reqs: "
            f"user={t_user * 1000:7.1f}ms ({t_user / n * 1e6:5.0f} us/req)  "
            f"flush={t_flush * 1000:7.1f}ms ({t_flush / num_merged * 1e6:5.0f} us/item)  "
            f"total={t_total * 1000:7.1f}ms"
        )


def profile():
    N = 3000

    # Profile user thread
    s, ml, pid = make_state()
    pr = cProfile.Profile()
    pr.enable()
    run_workload(s, ml, pid, N)
    pr.disable()
    print(f"\n=== User thread profile ({N} requests) ===")
    pstats.Stats(pr).sort_stats("tottime").print_stats(30)

    # Profile flush
    pr2 = cProfile.Profile()
    pr2.enable()
    run_flush(ml)
    pr2.disable()
    print(f"\n=== Flush profile ({N} requests) ===")
    pstats.Stats(pr2).sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    benchmark()
    if "--profile" in sys.argv:
        profile()
