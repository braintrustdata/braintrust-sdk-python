"""Individual benchmark modules.

Every ``bench_*.py`` file in this package must expose a ``main()`` function
that accepts an optional ``pyperf.Runner`` and registers its benchmarks on it.

Signature::

    def main(runner: pyperf.Runner | None = None) -> None: ...

When *runner* is ``None`` the module should create its own ``Runner`` so it
can still be executed standalone (``python -m benchmarks.benches.bench_foo``).
"""
