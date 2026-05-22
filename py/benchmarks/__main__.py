"""Run every ``bench_*.py`` module inside ``benchmarks.benches``.

Usage::

    cd py
    python -m benchmarks               # run all benchmarks
    python -m benchmarks --fast         # pyperf flags are forwarded
    python -m benchmarks -o /tmp/b.json # save results
"""

import importlib
import pathlib
import pkgutil
import sys


# Ensure ``py/`` is on sys.path so pyperf worker subprocesses can resolve
# the ``benchmarks`` package regardless of their working directory.
_PY_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

import pyperf

import benchmarks.benches as _benches_pkg
from benchmarks._utils import disable_pyperf_psutil


def _discover_bench_modules():
    """Yield imported bench modules that expose a ``main()`` callable."""
    for info in pkgutil.iter_modules(_benches_pkg.__path__):
        if not info.name.startswith("bench_"):
            continue
        mod = importlib.import_module(f"benchmarks.benches.{info.name}")
        if callable(getattr(mod, "main", None)):
            yield mod


def main() -> None:
    disable_pyperf_psutil()
    runner = pyperf.Runner()

    for mod in _discover_bench_modules():
        mod.main(runner)


if __name__ == "__main__":
    main()
