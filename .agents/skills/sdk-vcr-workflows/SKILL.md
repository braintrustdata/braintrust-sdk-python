---
name: sdk-vcr-workflows
description: Work with Braintrust Python SDK VCR and cassette-backed tests. Use when adding or updating cassette-backed provider tests, deciding whether to re-record cassettes, debugging VCR failures, converting mock-heavy coverage to VCR, or handling cassette hygiene for provider integrations and Claude Agent SDK transport recordings.
---

# SDK VCR Workflows

This repo prefers real recorded provider coverage over mocks. The normal bug-fix path: reproduce with a failing cassette-backed test, implement the fix, and re-record only when the behavior change is intentional. `AGENTS.md` rules (work from `py/`, `mise`, red → green, no guessed session names) apply throughout.

## Read First

- `py/pyproject.toml` `[tool.braintrust.matrix]` (provider version pins, what `latest` resolves to) and `py/noxfile.py` (session names)
- `py/src/braintrust/conftest.py` (record modes, dummy keys, media sanitizers) and `py/src/braintrust/integrations/conftest.py` (version-aware cassette dirs)
- the target test file and its `cassettes/<version>/` directory

Open these references when the task needs them:
- [references/recording-failures.md](references/recording-failures.md) — a live recording failed (auth, quota, retired model, partial run), or a cassette-backed test fails and you need the debugging order.
- [references/discovery-engine.md](references/discovery-engine.md) — recording `google_discoveryengine` (REST VCR + gRPC helper, gcloud ADC setup).

## How VCR Works Here

- Record mode: `once` locally, `none` in CI. A local pass that silently recorded new traffic can still fail in CI.
- Cassettes are per version: `integrations/<provider>/cassettes/<version>/` (e.g. `latest/`, `1.71.0/`). Nox sets `BRAINTRUST_TEST_PACKAGE_VERSION`; `integrations/conftest.py` resolves the directory. Without the env var (raw pytest), cassettes fall back to base `cassettes/` — so always record through nox.
- Never add per-test `vcr_cassette_dir` / `cassette_library_dir` fixtures.
- Other locations: `py/src/braintrust/cassettes/`, `py/src/braintrust/wrappers/cassettes/`, `py/src/braintrust/devserver/cassettes/`. Keep cassettes next to their tests and move them with the tests.

## Standard Workflow

1. Identify the exact failing test, then reproduce with the exact nox session and version from `py/noxfile.py`.
2. For a bug, add or update a focused failing cassette-backed test first.
3. Decide whether the test can reuse an existing cassette or needs a new one.
4. Record only the narrowest affected test, then inspect the cassette diff.
5. Re-run the provider session in replay (`--vcr-record=none`). Expand only if shared code changed.

## Choosing Coverage

Use a cassette whenever the provider's request or response shape is part of the behavior: tracing/span contents derived from real payloads, streaming aggregation, version-specific provider behavior, or replacing older mock-heavy tests.

Mocks or fakes only for: a narrow error path recordings cannot provoke, purely local patcher resolution or version routing, or internal code where traffic adds nothing. Hand-built stand-ins for provider objects (stub clients, fake tools, response dicts) count as fakes. "The bug is in local post-processing", "a mock is faster", or "a mock already exists nearby" are not reasons. If you choose a mock for provider behavior, state why a cassette is impractical.

## Commands

```bash
cd py
nox -s "test_<provider>(<version>)"
nox -s "test_<provider>(<version>)" -- -k "test_name"
nox -s "test_<provider>(<version>)" -- --vcr-record=all -k "test_name"   # narrow re-record
nox -s "test_<provider>(<version>)" -- --vcr-record=none                 # replay-only check
nox -s "test_<provider>(<version>)" -- --disable-vcr

# Claude Agent SDK: subprocess transport recordings, not HTTP VCR
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)" -- -k "test_name"
```

Claude Agent SDK talks to the bundled `claude` subprocess over stdin/stdout; `integrations/claude_agent_sdk/_test_transport.py` records and resolves per-version cassettes. Do not force HTTP VCR patterns onto it.

## Re-recording Rules

Re-record only when behavior intentionally changed: the traced request/response shape changed, the provider behavior under test changed, the old cassette covered the wrong scenario, or you are adding a new focused regression. If unsure, ask the user.

Never re-record to mask an unexplained failure, fix a flake you don't understand, paper over a version mismatch, or as a broad suite refresh when one scenario changed. A large cassette diff needs an explanation.

When a live recording fails: on auth errors stop and ask for a key; never hand-write a cassette; delete cassettes that captured 429/402 error responses; restore hand-authored or drifted cassettes with `git checkout`. Then check `git status` for ` D` entries. Details and traps: [references/recording-failures.md](references/recording-failures.md).

## Cassette Hygiene

- One focused cassette per scenario; reuse existing patterns in the provider package.
- After deleting or renaming tests, run `make check-unused-cassettes INTEGRATIONS="<provider>"` and remove orphans. After dropping matrix versions, run `make check-stale-cassettes`.
- Sanitize binary bodies. For image tests that don't depend on pixels or byte size, replace generated images with a tiny valid image, keeping response structure, usage metadata, header scrubbing, and correct content-length (see `sanitize_generated_images` in `py/src/braintrust/conftest.py`). Keep original media when its contents, format, or size is under test.
- Prefer fixing brittle assertions over cassette churn. Assert meaningful span structure, not incidental provider noise.

## Related Skills

- `sdk-integrations` — integration implementation, patchers, tracing, span shape.
- `sdk-ci-triage` — the entry point is a GitHub Actions failure, even if the fix involves VCR.
- `sdk-dependency-updates` — re-recording `latest` cassettes after a provider version bump.

## Validation Checklist

- Exact provider session and version reproduced; red → green for bugs.
- Cassette-backed test over mocks/fakes, or a stated reason why not.
- Narrowest recording; cassette diff inspected; no ` D` entries left behind.
- Provider session re-run in replay mode after recording.
- Subprocess auto-instrument coverage checked if patching or `auto_instrument()` changed.
