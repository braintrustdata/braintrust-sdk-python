# Braintrust SDK Agent Guide

Guide for contributing to the Braintrust Python SDK repository.

## Defaults

- Use `mise` as the source of truth for tools and environment.
- Prefer `py/` commands over root `make` targets when working on the SDK itself.
- Keep changes narrow and run the smallest relevant test session first.
- Do not rely on optional provider packages being installed unless the active nox session installs them.

## Repo Map

- `py/`: main Python package, tests, examples, nox sessions, release build
- `integrations/`: separate integration packages
- `internal/golden/`: compatibility and golden projects
- `docs/`: supporting docs

Important code areas in `py/src/braintrust/`:

- core SDK modules: top-level package files
- wrappers/integrations: `wrappers/`
- temporal: `contrib/temporal/`
- CLI/devserver: `cli/`, `devserver/`
- tests: colocated `test_*.py`

## Setup

Preferred repo bootstrap:

```bash
mise install
make develop
```

Package-focused setup:

```bash
cd py
make install-dev
```

Install optional provider dependencies only if needed:

```bash
cd py
make install-optional
```

## Commands

Preferred SDK workflow:

```bash
cd py
make lint
make test-core
nox -l
```

For larger or cross-cutting changes, also run `make pylint` from `py/` before handing work off.

Targeted wrapper/session runs:

```bash
cd py
nox -s "test_openai(latest)"
nox -s "test_openai(latest)" -- -k "test_chat_metrics"
```

Root `Makefile` exists as a convenience wrapper. The authoritative SDK workflow is in `py/Makefile` and `py/noxfile.py`.

## Tests

`py/noxfile.py` is the source of truth for compatibility coverage.

Key facts:

- `test_core` runs without optional vendor packages.
- wrapper coverage is split across dedicated nox sessions by provider/version.
- `pylint` installs the broad dependency surface before checking files.
- `cd py && make pylint` runs only `pylint`; `cd py && make lint` runs pre-commit hooks first and then `pylint`.
- `test-wheel` is a wheel sanity check and requires a built wheel first.

When changing behavior, run the narrowest affected session first, then expand only if needed.

## VCR

VCR cassette directories:

- `py/src/braintrust/cassettes/`
- `py/src/braintrust/wrappers/cassettes/`
- `py/src/braintrust/devserver/cassettes/`

Behavior from `py/src/braintrust/conftest.py`:

- local default: `record_mode="once"`
- CI default: `record_mode="none"`
- wheel-mode skips VCR-marked tests
- test fixtures inject dummy API keys and reset global state

Common commands:

```bash
cd py
nox -s "test_openai(latest)"
nox -s "test_openai(latest)" -- --disable-vcr
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

Only re-record cassettes when the behavior change is intentional. If in doubt, ask the user.

## Build Notes

Build from `py/`:

```bash
cd py
make build
```

Important caveat:

- `py/scripts/template-version.sh` rewrites `py/src/braintrust/version.py` during build.
- `py/Makefile` restores that file afterward with `git checkout`.

Avoid editing `py/src/braintrust/version.py` while also running build commands.

## Editing Guidance

- Keep tests near the code they cover.
- Reuse existing fixtures and cassette patterns.
- If a change affects examples or integrations, update the nearest example or focused test.
- For CLI/devserver changes, consider whether wheel-mode behavior also needs coverage.
