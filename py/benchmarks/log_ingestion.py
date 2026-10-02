"""End-to-end HTTP ingestion measurements (run from py/).

python -m benchmarks.log_ingestion --output /tmp/ingestion.json
Works on the legacy writer as well, for a same-harness baseline comparison.
"""

import argparse
import contextlib
import json
import os
import statistics
import threading
import time
import tracemalloc

from braintrust.api._test_server import scripted_server
from braintrust.api._transport import HTTPConnection
from braintrust.logger import _HTTPBackgroundLogger
from braintrust.util import LazyValue

from benchmarks.fixtures import ingestion_rows


def measure(throttled, rows=2000, *, continuous=False):
    rejected = 0
    accepted = 0
    arrivals = []
    connections = set()
    lock = threading.Lock()
    started = time.monotonic()

    def respond(method, path, body, headers):
        nonlocal rejected, accepted
        if path == "/version":
            return 200, {}, b"{}"
        now = time.monotonic()
        with lock:
            arrivals.append(now - started)
            if throttled and now - started < 1:
                rejected += 1
                return 429, {"Retry-After": "1"}, b"limited"
            accepted += len(json.loads(body)["rows"])
        return "sleep", 0.005, 200, {}, b"ok"

    os.environ["BRAINTRUST_DISABLE_ATEXIT_FLUSH"] = "1"
    os.environ["BRAINTRUST_NUM_RETRIES"] = "2"
    with scripted_server(respond, persistent=True) as (url, handler):
        # Select the branch's ingestion service when available.
        try:
            from braintrust.api._ingestion import LogIngestionAPI
            from braintrust.api._routing import EndpointRouter

            connection = LogIngestionAPI(EndpointRouter(app_url=url, api_url=url), "benchmark", concurrency=4)
            source = lambda: contextlib.nullcontext(connection)
        except ImportError:
            connection = HTTPConnection(url)
            source = LazyValue(lambda: connection, use_mutex=False)
        writer = _HTTPBackgroundLogger(source)
        writer.sync_flush = not continuous
        writer._max_request_size_result = {"max_request_size": 6_000_000, "can_use_overflow": False}
        latencies = []
        tracemalloc.start()
        started = time.monotonic()
        peak_outstanding = 0
        for index, row in enumerate(ingestion_rows(rows)):
            tick = time.perf_counter()
            writer.log(LazyValue(lambda row=row: row, use_mutex=False))
            latencies.append(time.perf_counter() - tick)
            with lock:
                peak_outstanding = max(peak_outstanding, index + 1 - accepted)
            if continuous:
                time.sleep(0.0005)
        queued = writer.queue.size()
        tick = time.monotonic()
        writer.flush()
        flush_seconds = time.monotonic() - tick
        elapsed = time.monotonic() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if accepted != rows:
            raise RuntimeError(f"Benchmark delivered {accepted} of {rows} rows")
        writer.sync_flush = False
        connection.close()
        connections.update(getattr(handler, "connections", []))
        return {
            "rows": rows,
            "delivered_rows": accepted,
            "rejected_requests": rejected,
            "requests": len(arrivals),
            "queued_rows_at_flush": queued,
            "outstanding_rows_peak": peak_outstanding,
            "pending_rows_after_flush": getattr(writer, "pending_count", 0),
            "rows_per_second": rows / elapsed,
            "flush_seconds": flush_seconds,
            "producer_p50_us": statistics.median(latencies) * 1e6,
            "producer_p99_us": sorted(latencies)[int(len(latencies) * 0.99)] * 1e6,
            "peak_allocated_bytes": peak,
            "recovery_seconds": max(arrivals) if arrivals else 0,
            "connections": len(connections),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    results = {
        name: [measure(throttled) for _ in range(args.runs)]
        for name, throttled in [("healthy", False), ("throttled", True)]
    }
    results["continuous_throttled"] = [measure(True, continuous=True) for _ in range(args.runs)]
    with open(args.output, "w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
