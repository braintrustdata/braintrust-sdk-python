# Braintrust SDK Agent Guide

This guide is for development of the Braintrust Python SDK in this repository.
If you need to learn more about Braintrust itself, see the Braintrust docs: https://www.braintrust.dev/docs

Use this file as the default playbook for work in this repository.

## Core Rules

1. **For SDK work, treat `py/` as the primary workspace.**
   - Read files under `py/`.
   - Run commands from `py/`.
   - Prefer `py/` commands over repo-root wrappers unless the task is clearly repo-level.

2. **Use `mise` as the source of truth for tools and environment.**

3. **Do not guess test commands or version coverage.**
   - `py/noxfile.py` is the source of truth for nox session names, provider/version matrices, and local reproduction commands.
   - `.github/workflows/checks.yaml` is the source of truth for which sessions run in CI, on which Python versions, and outside vs. inside the nox shard matrix.
   - For provider and integration work, also check `py/src/braintrust/integrations/versioning.py`.

4. **Keep changes narrow and validate with the smallest relevant test first.**

5. **Default bug-fix workflow: red -> green.**
   - First add or update a test that reproduces the issue.
   - Then implement the fix.
   - Only skip this if the user explicitly asks for a different approach.

6. **Prefer real integration coverage over mocks.**
   - For provider/integration behavior, prefer VCR-backed tests with checked-in cassettes.
   - This includes bugs in tracing/span shaping that happen after the SDK returns a real provider payload. If the behavior depends on the provider's actual response shape, treat it as VCR-first work, not mock-first work.
   - Be actively skeptical of mock/fake tests for provider integrations. Do not reach for mocks just because they are faster or easier to write.
   - Avoid mocks/fakes unless the code is purely local or there is no practical cassette-based option.

7. **Do not assume optional provider packages are installed.**
   - Rely on the active nox session to install what it needs.

8. **Do not add `from __future__ import annotations` unless absolutely required.**
   - It can change runtime annotation behavior in ways that break introspection.
   - Prefer quoted forward references or `TYPE_CHECKING` guards.

## Repo Map

- `py/`: main Python package, tests, examples, nox sessions, build/release workflow
- `py/src/braintrust/`: SDK source
  - top-level package files: core SDK
  - `wrappers/`: wrappers
  - `integrations/`: integrations API
  - `contrib/temporal/`: Temporal support
  - `cli/`, `devserver/`: CLI and devserver
  - `type_tests/`: static + runtime type tests
  - colocated `test_*.py`: local unit/integration tests
- `py/benchmarks/`: pyperf benchmarks
- `integrations/`: separate integration packages
- `docs/`: supporting docs

## Setup

Repo bootstrap:

```bash
mise install
make develop
```

SDK-focused setup:

```bash
cd py
make install-dev
```

Install optional provider dependencies only when needed:

```bash
cd py
make install-optional
```

## Default Workflow

When working on the SDK, prefer this sequence:

```bash
cd py
```

1. Read the relevant code and tests.
2. Check `noxfile.py` for the exact session(s) that cover the change.
3. If fixing behavior, add/update a reproducing test first.
4. Make the smallest possible change.
5. Run the narrowest affected test session first.
6. Expand coverage only as needed.
7. Before handoff, run broader hygiene checks if the change is large enough to justify them.

Common commands:

```bash
cd py
make lint
make test-core
nox -l
```

Notes:

- `cd py && make lint` runs pre-commit hooks and then `pylint`.
- `cd py && make pylint` runs only `pylint`.
- After major changes, run `cd py && make fixup` before handoff.
- The repo-root `Makefile` is a convenience wrapper; `py/Makefile` and `py/noxfile.py` are authoritative for SDK work.

## Testing Rules

### Always check the real CI target

Do not guess:

- nox session names
- supported provider versions
- which tests a provider session runs

Check `py/noxfile.py` and `.github/workflows/checks.yaml`, then reproduce with the exact local session CI uses.

### Run the smallest relevant test first

Examples:

```bash
cd py
nox -s "test_openai(latest)"
nox -s "test_openai(latest)" -- -k "test_chat_metrics"
```

### Provider and integration changes

Version-specific behavior matters in this repo.

Before changing provider/integration behavior:

1. Read the relevant session(s) in `py/noxfile.py`.
2. Read `py/src/braintrust/integrations/versioning.py`.
3. Confirm which versions, gates, fallbacks, and feature checks must keep working.
4. Do not stop at `latest` if the matrix includes older versions or version-specific branches.

### Key test facts

- `test_core` runs without optional vendor packages.
- `test_types` runs pyright, mypy, and pytest on `py/src/braintrust/type_tests/`.
- CI runs `pylint` and `test_types` via the dedicated `static_checks` workflow job on Ubuntu across the configured Python matrix, not inside the sharded `nox` job.
- The sharded `nox` workflow excludes `pylint` and `test_types`; use `py/scripts/nox-matrix.py --exclude-session ...` when reproducing shard membership locally.
- wrapper coverage is split across dedicated nox sessions by provider/version.
- `test-wheel` is a wheel sanity check and requires a built wheel first.

## Type Tests

Use `py/src/braintrust/type_tests/` when changing generic type signatures such as:

- `Eval`
- `EvalCase`
- `EvalScorer`
- `EvalHooks`

Rules:

- add or update a type test for the intended usage pattern
- name files `test_*.py`
- use absolute imports such as `from braintrust.framework import ...`

Run with:

```bash
cd py
nox -s test_types
```

## VCR and Cassettes

For provider and integration behavior, the default path is:

1. reproduce with a failing cassette-backed test
2. implement the fix
3. re-run the affected session

Do not downgrade to a mock/fake regression test just because the bug is in local post-processing of a real provider response. If the response shape is part of the behavior under test, the primary regression test should still be cassette-backed. Mock/unit tests may be added as supplemental coverage, not as the main reproduction, unless recording is genuinely impractical.

When deciding between a VCR test and a mock/fake test for provider behavior, bias heavily toward VCR. The burden of proof is on the mock: if you cannot clearly explain why a cassette-backed test is impractical, you should not be using a mock or fake as the primary regression coverage.

Cassette locations:

- `py/src/braintrust/cassettes/`
- `py/src/braintrust/wrappers/cassettes/`
- `py/src/braintrust/devserver/cassettes/`
- `py/src/braintrust/integrations/<provider>/cassettes/<version>/` for per-version integration cassettes
- `py/src/braintrust/integrations/claude_agent_sdk/cassettes/<version>/` for Claude Agent SDK subprocess transport recordings

Per-version cassette directories:

- Integration and Claude Agent SDK cassettes are stored in version-specific subdirectories (e.g. `cassettes/latest/`, `cassettes/1.71.0/`).
- Nox sessions set the `BRAINTRUST_TEST_PACKAGE_VERSION` env var, which `py/src/braintrust/integrations/conftest.py` uses to resolve the correct subdirectory.
- When running a test file directly (outside nox), the env var is absent and cassettes resolve to the base `cassettes/` directory for backward compatibility.
- Individual test files do not define their own `vcr_cassette_dir` fixtures; the shared `integrations/conftest.py` handles it.

Behavior from `py/src/braintrust/conftest.py`:

- local default: `record_mode="once"`
- CI default: `record_mode="none"`
- wheel mode skips VCR-marked tests
- fixtures inject dummy API keys and reset global state

Common commands:

```bash
cd py
nox -s "test_openai(latest)"
nox -s "test_openai(latest)" -- --disable-vcr
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

When re-recording, the nox session sets `BRAINTRUST_TEST_PACKAGE_VERSION` automatically, so cassettes land in the correct version subdirectory.

Claude Agent SDK note:

- it does not use HTTP VCR
- it talks to the bundled `claude` subprocess over stdin/stdout
- it uses transport-level cassette helpers instead
- cassettes are also stored per-version under `integrations/claude_agent_sdk/cassettes/<version>/`

Common Claude Agent SDK commands:

```bash
cd py
nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)" -- -k "test_calculator_with_multiple_operations"
```

Only re-record HTTP or subprocess cassettes when the behavior change is intentional. If unsure, ask the user.

## Benchmarks

If you touch a hot path such as serialization, deep-copy, span creation, or logging, consider benchmarks.

Quick commands:

```bash
cd py
make bench
make bench BENCH_ARGS="--fast"
make bench BENCH_ARGS="-o /tmp/before.json"
make bench BENCH_ARGS="-o /tmp/after.json"
make bench-compare BENCH_BASE=/tmp/before.json BENCH_NEW=/tmp/after.json
```

Rules:

- benchmark hot-path changes when practical
- benchmark files live in `py/benchmarks/benches/`
- new files should be named `bench_<name>.py`
- each benchmark file must expose `main(runner: pyperf.Runner | None = None)`
- shared payload builders belong in `py/benchmarks/fixtures.py`

See `py/benchmarks/benches/bench_bt_json.py` for the pattern.

## Build Notes

Build from `py/`:

```bash
cd py
make build
```

Caveat:

- `py/scripts/template-version.py` rewrites `py/src/braintrust/version.py` during build
- `py/Makefile` restores that file afterward with `git checkout`

Avoid editing `py/src/braintrust/version.py` while also running build commands.

## Editing Guidelines

- Keep tests close to the code they cover.
- Reuse existing fixtures and cassette patterns.
- Prefer extending an existing cassette-backed test over adding a new mock-heavy test.
- If a change affects examples or integrations, update the nearest example or focused test.
- For CLI/devserver changes, consider whether wheel-mode behavior also needs coverage.

## Quick Decision Guide

- **Changing SDK code?** Work from `py/`.
- **Need a test command?** Read `py/noxfile.py`.
- **Fixing a bug?** Add/update a failing test first.
- **Changing provider/integration behavior?** Use VCR-backed coverage and check version gates.
- **Changing generic typing?** Add/update a file in `py/src/braintrust/type_tests/` and run `nox -s test_types`.
- **Touching a hot path?** Consider `cd py && make bench`.
- **Preparing handoff after a major change?** Run `cd py && make fixup`.
