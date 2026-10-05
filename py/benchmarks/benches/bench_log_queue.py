import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.logger import _enrich_attachments, _extract_attachments
from braintrust.queue import LogQueue

from benchmarks._utils import disable_pyperf_psutil
from benchmarks.fixtures import make_large_payload, make_medium_payload, make_small_payload


_queue = LogQueue(maxsize=1000)


def _bench_queue() -> None:
    _queue.drain_all()
    for i in range(1000):
        _queue.put(i)


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("log_queue.put[1000]", _bench_queue)
    for name, payload in (
        ("small", make_small_payload()),
        ("medium", make_medium_payload()),
        ("large", make_large_payload()),
    ):
        runner.bench_func(f"logger.extract_attachments[{name}]", _extract_attachments, payload, [])
        runner.bench_func(f"logger.enrich_attachments[{name}]", _enrich_attachments, payload)


if __name__ == "__main__":
    main()
