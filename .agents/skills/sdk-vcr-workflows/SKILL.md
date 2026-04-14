---
name: sdk-vcr-workflows
description: Work with Braintrust Python SDK VCR and cassette-backed tests. Use when adding or updating cassette-backed provider tests, deciding whether to re-record cassettes, debugging VCR failures, converting mock-heavy coverage to VCR, or handling cassette hygiene for provider integrations and Claude Agent SDK transport recordings.
---

# SDK VCR Workflows

Use this skill for VCR and cassette work in the Braintrust Python SDK repo.

This repo prefers real recorded integration coverage over mocks for provider behavior. Do not treat cassette work as an afterthought. The normal bug-fix path here is: reproduce with a failing cassette-backed test, implement the fix, and only re-record when the behavior change is intentional.

## Read First

Always read:

- `AGENTS.md`
- `py/noxfile.py`
- `py/src/braintrust/conftest.py`
- the target provider test file under `py/src/braintrust/integrations/<provider>/` or `py/src/braintrust/wrappers/`

Read when relevant:

- the provider integration package under `py/src/braintrust/integrations/<provider>/`
- the provider cassette directory under `py/src/braintrust/integrations/<provider>/cassettes/`
- `py/src/braintrust/cassettes/`
- `py/src/braintrust/wrappers/cassettes/`
- `py/src/braintrust/devserver/cassettes/`
- `py/src/braintrust/wrappers/claude_agent_sdk/` for subprocess transport recordings instead of HTTP VCR
- `py/src/braintrust/integrations/auto_test_scripts/` when auto-instrument subprocess coverage and cassettes interact

Do not guess cassette locations or session names. Check the current tree and `py/noxfile.py` first.

## Repo Rules To Follow

These rules must stay aligned with `AGENTS.md`:

- Work from `py/` for SDK tasks.
- Use `mise` as the source of truth for tools and environment.
- Do not guess nox session names or provider/version coverage.
- Default bug-fix workflow is red -> green.
- Prefer VCR-backed provider tests over mocks or fakes whenever practical.
- Treat mock/fake tests for provider behavior as an exception that requires justification, not as a neutral alternative.
- Only re-record HTTP or subprocess cassettes when the behavior change is intentional. If unsure, ask the user.
- Do not assume optional provider packages are installed outside the active nox session.

## How VCR Works In This Repo

Behavior from `py/src/braintrust/conftest.py` is the source of truth.

Current defaults:

- local default: `record_mode="once"`
- CI default: `record_mode="none"`
- wheel mode skips VCR-marked tests
- fixtures inject dummy API keys and reset global state

Implications:

- A test that passes locally by silently recording new traffic may still fail in CI if the cassette is missing or stale.
- CI will not save you by recording fresh traffic. If the cassette is wrong, CI should fail.
- Reproducing a CI VCR failure locally usually means running the exact nox session named in `py/noxfile.py`, not raw pytest in whatever environment happens to exist.

## Standard Workflow

1. Identify the exact provider test or cassette-backed failure.
2. Read `py/noxfile.py` and reproduce with the exact provider session and version.
3. If fixing a bug, add or update a focused failing cassette-backed test first.
4. Decide whether the intended behavior should reuse an existing cassette or requires a new or updated one.
5. Re-record only the narrowest affected test when behavior intentionally changes.
6. Re-run the exact provider session after recording.
7. Expand only if shared integration code changed.

Do not re-record a broad provider suite when one focused test is enough.

Do not default to mocks/fakes just to avoid recording work. Extra setup effort is not, by itself, a good reason to abandon cassette-backed coverage.

## Choosing The Right Coverage Style

### Prefer cassette-backed tests when:

- validating real provider request or response shape
- testing tracing/span contents derived from actual provider payloads
- checking streaming aggregation against real wire behavior
- confirming version-specific provider behavior that mocks could misrepresent
- replacing older fake- or mock-heavy provider tests

### Use mocks/fakes only when:

- injecting a narrow error path that is hard to provoke via recordings
- testing purely local patcher resolution or applicability logic
- validating behavior that does not need real provider traffic
- the code under test is entirely internal and a cassette would add little value

If the task is about provider behavior and you can reasonably record it, prefer VCR.

Do not use mocks/fakes as the primary regression test for provider behavior merely because:
- the bug appears in local tracing or post-processing code
- a mock is faster to write
- recording requires adding a new cassette
- an existing mock test already exists nearby

In those situations, the right move is usually to add or update a focused cassette-backed test and keep any mock/unit test only as supplemental coverage.

## Cassette Locations

Common locations in this repo:

- `py/src/braintrust/cassettes/`
- `py/src/braintrust/wrappers/cassettes/`
- `py/src/braintrust/devserver/cassettes/`
- `py/src/braintrust/integrations/<provider>/cassettes/`
- `py/src/braintrust/wrappers/claude_agent_sdk/cassettes/` for Claude Agent SDK subprocess transport recordings

Keep cassettes next to the tests they support. When migrating or moving tests, move the cassettes with them.

## Commands

Run from `py/` unless the task is clearly repo-level.

Common provider VCR commands:

```bash
cd py
nox -s "test_openai(latest)"
nox -s "test_openai(latest)" -- -k "test_openai_chat_metrics"
nox -s "test_openai(latest)" -- --disable-vcr
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

Generic reproduction shape:

```bash
cd py
nox -s "test_<provider>(<version>)"
nox -s "test_<provider>(<version>)" -- -k "test_name"
nox -s "test_<provider>(<version>)" -- --vcr-record=all -k "test_name"
```

Claude Agent SDK transport-recording commands:

```bash
cd py
nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)" -- -k "test_calculator_with_multiple_operations"
```

## Re-recording Rules

Re-record only when the behavior change is intentional.

Good reasons to re-record:

- the traced request or response shape intentionally changed
- the provider behavior you are validating intentionally changed
- the old cassette covered the wrong scenario and the test now targets the correct one
- a new focused cassette-backed regression test is being added

Bad reasons to re-record:

- masking an unexplained failure
- making a flaky test pass without understanding the difference
- updating many cassettes when only one focused scenario changed
- papering over version mismatches or incorrect assumptions about old provider releases

If a cassette diff is large, verify that the behavioral change is truly intended before keeping it.

## Narrow Recording Strategy

When you need new traffic, record the smallest thing possible:

1. use the exact session from `py/noxfile.py`
2. target a single test with `-k` when practical
3. record only the affected scenario
4. inspect the cassette diff before moving on

Preferred shape:

```bash
cd py
nox -s "test_google_genai(latest)" -- --vcr-record=all -k "test_interactions_create_and_get"
```

Avoid broad commands like re-recording an entire provider suite unless the change genuinely affects the whole suite.

## Debugging VCR Failures

When a cassette-backed test fails, check these in order:

1. **Wrong nox session or provider version**
   - Did you reproduce under the exact session from `py/noxfile.py`?
   - Are you accidentally testing `latest` when CI pins an older version?

2. **Cassette missing or stale**
   - Does the expected cassette file exist in the right directory?
   - Did the test move without moving the cassette?
   - Does the cassette still match the intended request shape?

3. **Behavior changed but cassette did not**
   - Did the code intentionally change the payload, headers, streaming sequence, or traced output?
   - If yes, re-record narrowly.

4. **Behavior did not change, but the test became too strict**
   - Prefer fixing brittle assertions before churning cassettes.
   - Assert the meaningful span structure, not incidental provider noise.

5. **Local-only pass, CI fail**
   - Remember CI uses `record_mode="none"`.
   - A local silent record can hide a missing cassette or wrong request shape.

6. **Optional dependency or auth confusion**
   - Do not rely on globally installed provider packages.
   - Use the active nox session.
   - Read `py/src/braintrust/conftest.py` before assuming credential behavior.

## Cassette Hygiene

Keep cassettes reviewable and intentional.

Prefer:

- one focused cassette per scenario or regression when practical
- reusing existing cassette patterns in the same provider package
- keeping fixture names stable and descriptive
- moving cassettes with tests when code moves from wrappers to integrations

Be careful about:

- duplicate old cassettes left behind after moves or renames
- storing unnecessary raw binary content when sanitization is possible
- overfitting tests to incidental details in recordings
- broad cassette churn from unfocused recording commands

When the provider returns binary HTTP responses or generated media, sanitize the recordings as needed so the repo does not keep unnecessary raw file bytes.

## Claude Agent SDK Exception

Claude Agent SDK coverage is cassette-backed, but not through HTTP VCR.

Important differences:

- it talks to the bundled `claude` subprocess over stdin/stdout
- it uses transport-level cassette helpers instead of HTTP request recording
- use `BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all` when re-recording

Do not try to force ordinary HTTP VCR patterns onto Claude Agent SDK subprocess tests.

## Relationship To Other Skills

- Use `sdk-integrations` when the main task is integration implementation, patchers, tracing, or provider package structure.
- Use this skill when the main difficulty is cassette-backed test design, VCR behavior, re-recording decisions, or cassette hygiene.
- Use `sdk-ci-triage` when the entry point is a GitHub Actions failure, even if the fix eventually involves VCR.
- Use `sdk-wrapper-migrations` when moving a legacy wrapper into `py/src/braintrust/integrations/` while preserving tests and cassettes.

## Validation Checklist

- Read `py/noxfile.py` before choosing commands.
- Reproduce under the exact provider session and version.
- Use red -> green when fixing a bug.
- Prefer a focused cassette-backed test over mocks/fakes.
- If you choose a mock/fake test for provider behavior, be able to state exactly why a cassette-backed test is impractical.
- Re-record only when behavior intentionally changed.
- Record the narrowest affected test first.
- Inspect the cassette diff before finishing.
- Re-run the relevant provider session after recording.
- If patching or `auto_instrument()` changed, also check the relevant subprocess auto-instrument coverage.

## Common Mistakes

Avoid these failures:

- running raw pytest in an ad hoc environment instead of the exact nox session
- re-recording against `latest` when CI covers an older pinned version
- letting local `record_mode="once"` hide a missing or stale cassette
- replacing meaningful assertions with cassette churn
- using mocks for provider behavior that should be validated from real recordings
- treating local tracing/span-shaping bugs as mock-first when the trigger is a real provider payload
- forgetting that Claude Agent SDK uses subprocess transport recordings, not HTTP VCR
- leaving duplicate stale cassettes behind after moving tests or renaming scenarios
- broad re-records that create unnecessary review noise
