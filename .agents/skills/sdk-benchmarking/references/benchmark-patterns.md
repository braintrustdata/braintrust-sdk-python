# Benchmark Patterns

Use this reference when adding or updating SDK benchmarks.

## Command Cheatsheet

```bash
cd py

# Run everything
make bench

# Faster local iteration
make bench BENCH_ARGS="--fast"

# Save results for comparison
make bench BENCH_ARGS="-o /tmp/before.json"
make bench BENCH_ARGS="-o /tmp/after.json"
make bench-compare BENCH_BASE=/tmp/before.json BENCH_NEW=/tmp/after.json

# Run one module directly
python -m benchmarks.benches.bench_bt_json

# Inspect all forwarded pyperf flags
python -m benchmarks --help
```

## Module Skeleton

```python
import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks._utils import disable_pyperf_psutil


def target(value):
    return value


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("example.target[case-name]", target, "value")


if __name__ == "__main__":
    main()
```

Follow the existing `py/benchmarks/benches/bench_bt_json.py` pattern when importing repo code. The `sys.path` adjustment keeps direct module execution working from inside `py/`.

## Fixture Guidance

Put reusable builders in `py/benchmarks/fixtures.py` when:

- several benchmark cases need the same payload shape
- the payload should model realistic nested SDK inputs
- the benchmark should cover edge cases such as circular references or non-string keys

Current fixture patterns already cover:

- small, medium, and large nested payloads
- circular structures
- non-string dictionary keys
- dataclass-like and pydantic-like values

Extend those helpers before creating one-off payload factories in a new benchmark module.

## Comparison Workflow

For branch-to-branch comparisons:

```bash
cd py
git checkout main
make bench BENCH_ARGS="-o /tmp/main.json"

git checkout my-branch
make bench BENCH_ARGS="-o /tmp/branch.json"

make bench-compare BENCH_BASE=/tmp/main.json BENCH_NEW=/tmp/branch.json
```

Use `--rigorous` only when you need lower-noise final numbers; use `--fast` while iterating.
