---
name: sdk-benchmarking
description: Run, compare, and extend Braintrust Python SDK pyperf benchmarks. Use when touching hot-path code in `py/src/braintrust/` such as serialization, deep-copy, span creation, or logging; when adding or updating files under `py/benchmarks/`; or when you need baseline/branch performance measurements with `cd py && make bench` and `make bench-compare`.
---

# SDK Benchmarking

Use this skill for benchmark work in the Braintrust Python SDK repository.

Benchmark support already exists in `py/benchmarks/`. Use the current repo workflow, not commit archaeology, once you have identified the relevant benchmark surface.

## Read First

Always read:

- `AGENTS.md`
- `CONTRIBUTING.md`
- `py/Makefile`
- `py/benchmarks/__main__.py`
- `py/benchmarks/_utils.py`
- `py/benchmarks/benches/__init__.py`

Read when relevant:

- `py/benchmarks/benches/bench_bt_json.py` for the module pattern
- `py/benchmarks/fixtures.py` for shared payload builders
- `py/pyproject.toml` when benchmarking the optional `orjson` fast path (see `[project.optional-dependencies]`)
- `references/benchmark-patterns.md` in this skill for command and module templates

## Workflow

1. Identify the hot path or API surface that changed.
2. Find the nearest existing benchmark module under `py/benchmarks/benches/`.
3. Run the narrowest useful benchmark first.
4. Add or update a `bench_*.py` module only if the current suite does not cover the changed path.
5. Reuse or extend `py/benchmarks/fixtures.py` for realistic shared payloads instead of inlining bulky test data.
6. Save before/after results and compare them when the task is about regression detection or improvement claims.

## Commands

Run benchmarks from `py/`:

```bash
cd py
make bench
make bench BENCH_ARGS="--fast"
make bench BENCH_ARGS="-o /tmp/before.json"
make bench BENCH_ARGS="-o /tmp/after.json"
make bench-compare BENCH_BASE=/tmp/before.json BENCH_NEW=/tmp/after.json
python -m benchmarks.benches.bench_bt_json
```

Use `python -m benchmarks --help` for extra `pyperf` flags.

If the benchmark should measure the optional `orjson` path, install the performance extra first:

```bash
cd py
uv sync --extra performance
```

## Adding Benchmarks

Put new modules in `py/benchmarks/benches/` and name them `bench_<name>.py`.

Every benchmark module must:

- expose `main(runner: pyperf.Runner | None = None) -> None`
- create its own `pyperf.Runner()` only when `runner` is `None`
- call `disable_pyperf_psutil()` before creating that runner
- register benchmarks with stable, descriptive names via `runner.bench_func(...)`
- remain executable directly with `python -m benchmarks.benches.bench_<name>`

Do not add manual registration. `python -m benchmarks` auto-discovers every `bench_*.py` module in `py/benchmarks/benches/`.

## Fixtures

Keep reusable payload builders and synthetic objects in `py/benchmarks/fixtures.py`.

Prefer fixture helpers when:

- several benchmark cases share similar payloads
- the inputs are large enough to distract from the benchmark itself
- you need variants such as small, medium, large, circular, or non-string-key cases

Keep fixture builders deterministic and focused on representative data shapes.

## Validation

- Run the narrowest affected benchmark first.
- Use `BENCH_ARGS="--fast"` for quick local sanity checks while iterating.
- Save JSON outputs and use `make bench-compare` for baseline versus branch comparisons.
- If you changed code paths that also have correctness tests, run the smallest relevant test target in addition to the benchmark.

## Pitfalls

- Measuring import/setup overhead instead of the hot function under test.
- Inlining ad hoc payload construction in each benchmark instead of reusing fixtures.
- Forgetting the standalone `main()` pattern, which breaks auto-discovery and direct execution symmetry.
- Claiming performance changes from a single unsaved local run instead of comparing saved results.
- Benchmarking the `orjson` fast path without explicitly installing `.[performance]`.
