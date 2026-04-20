---
name: sdk-dependency-updates
description: "Review and refresh Braintrust Python SDK dependency update PRs, especially automated `chore(deps): daily dependency update` PRs from `.github/workflows/dependency-updates.yml`. Use when Codex needs to inspect `py/pyproject.toml` and `py/uv.lock`, verify whether `needs-cassette-rerecord` is actually required, map changed matrix keys to exact nox sessions and cassette directories, re-record only affected `latest` cassettes, and validate playback before merge."
---

# SDK Dependency Updates

Use this skill for dependency bump PRs in this repo, especially the daily automated PR opened by `.github/workflows/dependency-updates.yml`.

These PRs should usually stay narrow:

- `py/pyproject.toml`
- `py/uv.lock`

If provider SDK packages changed, a human often needs to refresh affected `latest` cassettes before merge.

## Read First

Always read:

- `AGENTS.md`
- `.github/workflows/dependency-updates.yml`
- `.github/workflows/checks.yaml`
- `py/pyproject.toml`
  Focus on `[tool.braintrust.matrix]` and `[tool.braintrust.cassette-dirs]`.
- `py/noxfile.py`
- `py/scripts/update-matrix-latest.py`
- `py/scripts/determine-dependency-update-labels.py`
- `py/src/braintrust/conftest.py`
- `py/src/braintrust/integrations/conftest.py`

Read when relevant:

- PR metadata from `gh pr view <pr>`
- changed provider integration code under `py/src/braintrust/integrations/<provider>/`
- changed provider tests under `py/src/braintrust/integrations/<provider>/test_*.py`
- affected cassette directories under `py/src/braintrust/integrations/<provider>/cassettes/latest/`
- `py/src/braintrust/integrations/versioning.py` when version-gated behavior may matter
- `py/scripts/check-stale-cassettes.py` when versions were added or removed

If the task becomes mostly about VCR mechanics, use `sdk-vcr-workflows`.
If the entry point is a failing GitHub Actions job, use `sdk-ci-triage`.
If the dependency bump exposes a real provider bug that needs code changes, use `sdk-integrations`.

## What The Automation Does

The daily workflow does this:

1. run `python scripts/update-matrix-latest.py` from `py/`
2. run `uv lock --upgrade`
3. run `python scripts/determine-dependency-update-labels.py`
4. open a PR titled `chore(deps): daily dependency update`
5. apply:
   - `auto-merge-candidate` when only infra/test deps changed
   - `needs-cassette-rerecord` when provider SDK packages changed

Treat the workflow and label script as the source of truth, but still verify the actual diff. Do not trust the label blindly.

## Core Rules

- Work from `py/` for SDK commands.
- Use `mise` as the source of truth for tools and Python versions.
- Do not guess session names; read `py/noxfile.py`.
- Do not guess whether re-recording is required; inspect the actual diff in `py/pyproject.toml` and `py/uv.lock`.
- Keep dependency PR changes narrow. Avoid touching SDK code unless the user explicitly asks.
- Refresh only affected `latest` cassettes. Do not churn older version directories unless older pins changed or the user asked.
- If a changed matrix key has no cassette-dir mapping, do not invent cassette work. Run the exact targeted validation that exists for that package.
- Always review the cassette diff itself, not just pass/fail status.

## Classify The Change First

Before re-recording anything, classify the changed matrix keys.

### 1. Infra-only keys

These do not imply provider cassette refresh work:

- `pytest-matrix`
- `braintrust-core`

These PRs are usually low-risk if CI passes and the diff stays narrow.

### 2. Keys with versioned integration cassettes

These appear in `[tool.braintrust.cassette-dirs]` and usually require targeted `latest` cassette refresh when their package version changes:

- `anthropic`
- `cohere`
- `openai`
- `openai-agents`
- `litellm`
- `claude-agent-sdk`
- `agno`
- `agentscope`
- `pydantic-ai-integration`
- `pydantic-ai-wrap-openai`
- `google-genai`
- `dspy`
- `google-adk`
- `langchain-core`
- `openrouter`
- `mistralai`

### 3. Matrix keys without versioned integration cassettes

These are still provider/runtime bumps, but do not map to `integrations/*/cassettes/latest/`:

- `autoevals`
- `temporalio`

For these, run the exact nox session and review the diff, but do not invent cassette refresh instructions that the repo does not use.

## Map Matrix Keys To Actual Sessions

Do not assume package names, session names, and integration directory names line up perfectly.

Useful mismatches to remember:

- `google-adk` -> `test_google_adk(latest)` -> `py/src/braintrust/integrations/adk/cassettes/latest/`
- `langchain-core` -> `test_langchain(latest)` -> `py/src/braintrust/integrations/langchain/cassettes/latest/`
- `mistralai` -> `test_mistral(latest)` -> `py/src/braintrust/integrations/mistral/cassettes/latest/`
- `pydantic-ai-integration` -> `test_pydantic_ai_integration(latest)` and `test_pydantic_ai_logfire(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `pydantic-ai-wrap-openai` -> `test_pydantic_ai_wrap_openai(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `claude-agent-sdk` -> `test_claude_agent_sdk(latest)` -> transport cassettes under `py/src/braintrust/integrations/claude_agent_sdk/cassettes/latest/`, not HTTP VCR

For everything else, confirm the mapping in `py/noxfile.py` and `py/pyproject.toml`.

## Standard Workflow

### 1. Inspect the PR and diff

Start with metadata and changed files:

```bash
gh pr view <pr-number> --repo braintrustdata/braintrust-sdk-python \
  --json title,body,labels,files,headRefName,baseRefName
gh pr checks <pr-number> --repo braintrustdata/braintrust-sdk-python
git diff origin/main...HEAD -- py/pyproject.toml py/uv.lock
```

Confirm:

- the PR is the dependency-update workflow output
- changed files are still narrowly scoped
- whether the label matches the actual dependency diff

### 2. Reproduce the workflow's classification locally

From `py/`, use the same script the workflow uses:

```bash
cd py
python scripts/determine-dependency-update-labels.py
```

Then inspect `pyproject.toml` and `uv.lock` yourself. The script is authoritative for labels, but your review is authoritative for deciding what work to do next.

### 3. Decide the minimum work

Default to the smallest scope that matches the diff:

- one changed provider package: refresh or validate only that provider's `latest` coverage
- several changed provider packages: handle each provider independently
- polluted or flaky subset after a broad refresh: delete and re-record only those files

When providers are independent, parallel subagents are a good fit: one provider/session per subagent.

### 4. Delete before re-recording

For cassette refresh work, prefer deleting affected `latest` cassettes before re-recording so stale responses are not silently reused.

Examples:

```bash
cd py
rm -rf src/braintrust/integrations/langchain/cassettes/latest
rm -rf src/braintrust/integrations/pydantic_ai/cassettes/latest
```

For surgical cleanup, delete only the polluted cassette files first.

Do not delete older version cassette directories unless the task requires it.

### 5. Re-record with the exact nox session

Use the exact session from `py/noxfile.py`.

Examples:

```bash
cd py
nox -s "test_openai_agents(latest)" -- --vcr-record=all
nox -s "test_litellm(latest)" -- --vcr-record=all
nox -s "test_langchain(latest)" -- --vcr-record=all
nox -s "test_pydantic_ai_integration(latest)" -- --vcr-record=all
nox -s "test_pydantic_ai_wrap_openai(latest)" -- --vcr-record=all
nox -s "test_pydantic_ai_logfire(latest)" -- --vcr-record=all
```

For Claude Agent SDK transport recordings:

```bash
cd py
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
```

Prefer nox for recording so `BRAINTRUST_TEST_PACKAGE_VERSION=latest` is set correctly and cassettes land in the correct version directory.

### 6. Avoid cassette pollution

Dependency PR cassette refreshes are vulnerable to local-environment leakage.

Before recording, be suspicious of local Braintrust auth or URL env vars. If recordings start capturing unrelated Braintrust traffic, neutralize local config and re-record.

Useful environment shape for clean provider cassette recording:

```bash
unset BRAINTRUST_API_URL
unset BRAINTRUST_APP_URL
unset BRAINTRUST_ORG_NAME
export BRAINTRUST_API_KEY=___TEST_API_KEY__
```

If you must run outside nox, also set:

```bash
export BRAINTRUST_TEST_PACKAGE_VERSION=latest
```

Default to nox instead of raw pytest when possible.

### 7. Scan for pollution and review diffs

These are usually pollution, not meaningful provider behavior:

- `https://staging-api.braintrust.dev/logs3`
- `403 ForbiddenError` from Braintrust logging endpoints
- `api.braintrust.dev/version`
- `staging-api.braintrust.dev/version`
- unrelated Braintrust telemetry/version traffic in provider cassettes

Quick scan:

```bash
rg -n "staging-api\.braintrust\.dev|api\.braintrust\.dev/version|/logs3|braintrust\.dev/version" \
  py/src/braintrust/integrations/*/cassettes/latest
```

Also inspect the cassette diff. Expected churn includes:

- provider SDK version metadata changes
- ids, timestamps, session ids, request ids
- harmless response-shape additions from the upstream SDK
- binary output churn in audio/media cassettes
- cache-related metadata such as prompt caching fields

Be skeptical of:

- brand-new endpoint families unrelated to the provider under test
- Braintrust API traffic inside provider cassettes
- large request-shape changes that do not match the dependency bump
- missing interactions after a delete/re-record pass
- diffs that suggest the wrong test or wrong package version was recorded

### 8. Validate playback

After recording, run playback using the same session names without forcing record mode.

Examples:

```bash
cd py
nox -s "test_openai_agents(latest)"
nox -s "test_litellm(latest)"
nox -s "test_langchain(latest)"
nox -s "test_pydantic_ai_integration(latest)"
nox -s "test_pydantic_ai_wrap_openai(latest)"
nox -s "test_pydantic_ai_logfire(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=none nox -s "test_claude_agent_sdk(latest)"
```

If you only touched a subset, run the narrowest affected playback first.

When matrix versions changed in a way that could orphan cassette directories, also run:

```bash
cd py
make check-stale-cassettes
```

If the PR touched broader CI or workflow wiring, expand validation based on `.github/workflows/checks.yaml`.

## Report Back

When you finish, report:

1. whether the PR was infra-only or needed targeted refresh work
2. which matrix keys or provider packages changed
3. which nox sessions you ran
4. which cassette directories or files you deleted and re-recorded
5. any suspicious diff items you investigated
6. whether you found and removed pollution
7. the final playback or validation result

## Common Mistakes

Avoid these mistakes:

- trusting the PR label without checking the diff
- re-recording every provider when only a few matrix keys changed
- assuming every provider bump has versioned cassettes
- recording older version cassettes when only `latest` moved
- forgetting that `claude_agent_sdk` uses transport recordings, not HTTP VCR
- recording with raw pytest outside nox and landing cassettes in the wrong directory
- reviewing only pass/fail status and not the cassette diff
- touching SDK code in a dependency-refresh PR without an explicit reason
- skipping playback after a successful re-record
