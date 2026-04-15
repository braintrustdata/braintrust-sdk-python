---
name: sdk-ci-triage
description: Triage and reproduce Braintrust Python SDK CI failures. Use when asked why CI failed, to fix broken CI on a PR, to inspect a failing GitHub Actions job, or to map a failing matrix job back to the exact local nox session, provider version, or workflow step that must be reproduced.
---

# SDK CI Triage

Use this skill when the task is about GitHub Actions failures in the Braintrust Python SDK repo.

This repo's CI is heavily matrixed. Do not guess which local command matches a failing job. Start from the workflow, map the failing job to the exact nox shard or workflow step, then reproduce the narrowest failing command locally.

## Read First

Always read:

- `AGENTS.md`
- `.github/workflows/checks.yaml`
- `py/noxfile.py`
- `py/scripts/nox-matrix.py`
- `py/scripts/session-weights.json`

Read when relevant:

- `.github/workflows/adk-py-test.yaml`
- `.github/workflows/langchain-py-test.yaml`
- `py/src/braintrust/conftest.py` for VCR behavior
- the provider test file named in the failing logs
- the provider integration package under `py/src/braintrust/integrations/<provider>/`
- `py/Makefile` when the failure is in lint, build, or wheel-related steps

## CI Shape In This Repo

`checks.yaml` currently has these main categories:

- `lint`: pre-commit diff-based checks
- `ensure-pinned-actions`: workflow hygiene
- `static_checks`: Ubuntu-only Python matrix for `pylint` and `test_types`
- `smoke`: install/import matrix across Python and OS
- `nox`: provider and core test matrix, sharded through `py/scripts/nox-matrix.py`
- `adk-py`: reusable workflow for ADK coverage
- `langchain-py`: reusable workflow for LangChain coverage
- `upload-wheel`: build wheel sanity check

The most common failure source is still the `nox` matrix job, but `pylint` and `test_types` failures now surface through `static_checks`, not through `nox`.

## Standard Workflow

1. Identify the failing PR, run, or job.
2. Inspect the failing job logs with `gh`.
3. Determine which workflow branch failed:
   - `lint`
   - `static_checks`
   - `smoke`
   - `nox`
   - reusable workflow (`adk-py`, `langchain-py`)
   - `upload-wheel`
4. For `nox` failures, map the matrix job to the exact nox session and pinned provider version from the logs.
5. For `static_checks` failures, identify whether `pylint` or `test_types` failed under the reported Python version.
6. Reproduce the narrowest failing command locally.
7. Fix the bug.
8. Re-run the narrowest failing command first.
9. Expand only if shared code changed.

Do not start by running the whole suite locally unless the failure genuinely spans many sessions.

## GitHub CLI Playbook

Use `gh` first; do not manually browse logs when the CLI can answer the question directly.

Common starting points:

```bash
gh pr checks <pr-number> --repo braintrustdata/braintrust-sdk-python
gh run list --repo braintrustdata/braintrust-sdk-python --limit 10
gh run view <run-id> --repo braintrustdata/braintrust-sdk-python
gh run view <run-id> --repo braintrustdata/braintrust-sdk-python --log-failed
gh run view --job <job-id> --repo braintrustdata/braintrust-sdk-python --log-failed
```

When you need structured data for job names, statuses, and ids:

```bash
gh api repos/braintrustdata/braintrust-sdk-python/actions/runs/<run-id>/jobs \
  --jq '.jobs[] | {name: .name, status: .status, conclusion: .conclusion, id: .id}'
```

When you already know the job id and want raw logs:

```bash
gh api repos/braintrustdata/braintrust-sdk-python/actions/jobs/<job-id>/logs
```

## How To Triage By Job Type

### `nox` matrix jobs

Job names look like this:

```text
nox (3.10, ubuntu-24.04, 0)
```

That means:

- Python `3.10`
- OS `ubuntu-24.04`
- shard `0` out of 4

The workflow runs:

```bash
mise exec python@<python-version> -- python ./py/scripts/nox-matrix.py <shard> 4 \
  --exclude-session pylint \
  --exclude-session test_types
```

Use a dry run first to see which sessions belong to the shard:

```bash
mise exec python@3.10 -- python ./py/scripts/nox-matrix.py 0 4 --dry-run \
  --exclude-session pylint \
  --exclude-session test_types
```

Then inspect the failing logs to find the exact session name, for example:

```text
nox > Running session test_google_genai(1.30.0)
...
nox > Session test_google_genai(1.30.0) failed.
```

Now reproduce the exact session locally from `py/`:

```bash
cd py
nox -s "test_google_genai(1.30.0)"
```

If the logs show a narrower failing test, use `-- -k` on the session:

```bash
cd py
nox -s "test_google_genai(1.30.0)" -- -k "interactions"
```

Do not substitute `latest` for an older pinned version from CI.

### `lint`

The lint job currently runs pre-commit against the PR diff:

```bash
mise exec -- pre-commit run --from-ref origin/<base> --to-ref HEAD
```

For local reproduction, use the same shape when possible:

```bash
mise exec -- pre-commit run --from-ref origin/main --to-ref HEAD
```

If the failure is really from SDK linting, also check:

```bash
cd py
make lint
make pylint
```

### `static_checks`

The `static_checks` job is an Ubuntu-only Python matrix that runs `pylint` and `test_types` together for each configured Python version.

Local equivalents:

```bash
mise exec python@3.10 -- nox -f ./py/noxfile.py -s pylint test_types
```

If only one of the two sessions failed in CI, narrow locally to that specific session:

```bash
mise exec python@3.10 -- nox -f ./py/noxfile.py -s pylint
mise exec python@3.10 -- nox -f ./py/noxfile.py -s test_types
```

### `smoke`

The smoke job validates install + import across OS and Python versions.

Local equivalents:

```bash
mise exec python@3.10 -- uv sync --project ./py --all-extras
mise exec python@3.10 -- uv run --active --no-project python -c 'import braintrust'
```

Use this path when CI fails before tests even start.

### `upload-wheel`

Reproduce from `py/`:

```bash
cd py
make install-build-deps
make build
```

Remember that build rewrites `py/src/braintrust/version.py` temporarily.

### Reusable workflows (`adk-py`, `langchain-py`)

Read the reusable workflow file and map it back to the exact nox or make command it runs. Then reproduce that command locally rather than guessing from the job name alone.

## Common Failure Patterns In This Repo

### 1. Version-matrix mismatch

This is the most common CI triage mistake.

Symptoms:

- import errors on older provider versions
- missing attributes or methods in pinned versions
- tests written against `latest` behavior only

Checklist:

- read the relevant version tuple in `py/noxfile.py`
- reproduce the exact pinned session from CI
- inspect `py/src/braintrust/integrations/versioning.py` when version routing is involved
- prefer feature detection or declarative version gating over ad hoc branching

### 2. Optional dependency assumptions

Symptoms:

- local machine passes because extra packages are installed
- CI fails in `test_core` or a focused provider session

Checklist:

- do not rely on packages outside the active nox session
- read the session in `py/noxfile.py` to see what gets installed
- remember `test_core` intentionally runs without vendor packages

### 3. VCR / cassette issues

Symptoms:

- cassette mismatch
- tests that work locally only with real credentials
- CI failures in VCR-marked provider tests

Checklist:

- read `py/src/braintrust/conftest.py`
- remember local default is `record_mode="once"`, CI is `record_mode="none"`
- only re-record cassettes when behavior intentionally changed
- prefer a narrow `-- -k <test>` rerun under the exact nox session

### 4. Auto-instrument / import-order regressions

Symptoms:

- subprocess auto-instrument tests fail
- import-before-setup versus setup-before-import behavior diverges
- patcher idempotence regressions

Checklist:

- inspect `py/src/braintrust/auto.py`
- inspect `py/src/braintrust/integrations/auto_test_scripts/`
- run the narrowest affected provider session first

### 5. OS-specific failures

Symptoms:

- Windows-only path or shell issues
- Linux-only import/install behavior

Checklist:

- use the exact job's Python version and read the failing job name carefully
- avoid assuming POSIX-only shell behavior in Python-facing logic
- inspect path normalization and file handling code first when Windows alone fails

## Reproduction Rules

- Reproduce the exact failing session before broadening.
- Use the exact provider version shown in CI logs.
- Use `cd py` for SDK test commands.
- Use `mise` as the source of truth for Python versions and environment.
- If the failure is in `nox`, do not jump straight to `make test-core` unless logs suggest shared/core fallout.

Preferred progression:

```bash
# 1. Inspect the failing shard
mise exec python@3.10 -- python ./py/scripts/nox-matrix.py 0 4 --dry-run \
  --exclude-session pylint \
  --exclude-session test_types

# 2. Reproduce the exact session
cd py
nox -s "test_google_genai(1.30.0)"

# 3. Narrow further if needed
nox -s "test_google_genai(1.30.0)" -- -k "interactions"
```

## What To Report Back

When answering a CI-triage question, report:

1. the exact failing workflow job
2. the exact failing nox session or command
3. the root cause, not just the symptom
4. the smallest local reproduction command
5. the likely fix path

Good example structure:

```text
The failing job is `nox (3.10, ubuntu-24.04, 0)`.
Within that shard, the failing session is `test_google_genai(1.30.0)`.
The root cause is that the tests import a symbol that does not exist in google-genai 1.30.0, even though it exists in newer versions.
You can reproduce it locally with `cd py && nox -s "test_google_genai(1.30.0)"`.
The fix is to gate the behavior for older versions or stop assuming the newer API exists in the pinned minimum version.
```

## Pitfalls

Avoid these common mistakes:

- guessing the session from the provider name without checking `py/noxfile.py`
- forgetting that CI excludes `pylint` and `test_types` from the sharded `nox` job
- reproducing with `latest` when CI failed on an older pinned version
- running from repo root when the real SDK command belongs in `py/`
- fixing the symptom in tests without understanding the provider-version contract
- re-recording cassettes when the behavior change was not intentional
- broadening to many sessions before the exact failing command is green
- assuming local globally-installed packages match the nox environment
