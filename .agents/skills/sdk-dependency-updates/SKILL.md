---
name: sdk-dependency-updates
description: "Review and refresh Braintrust Python SDK dependency update PRs, especially automated `chore(deps): weekly dependency update` PRs. Use when Codex needs to inspect `py/pyproject.toml` and `py/uv.lock`, reproduce the workflow's label decision, decide whether provider `latest` cassettes need re-recording, check that bumped pins still install on CI's lowest Python, research whether minor/major bumps need SDK changes, run the exact nox sessions, and validate playback before merge."
---

# SDK Dependency Updates

Use this skill for dependency bump PRs in this repo, especially the weekly automation from `.github/workflows/dependency-updates.yml`.

Keep the work narrow. These PRs should usually stay limited to:

- `py/pyproject.toml`
- `py/uv.lock`
- affected `latest` cassette files when a re-record is justified

If the task becomes mainly about VCR mechanics, use `sdk-vcr-workflows`.
If the entry point is a failing GitHub Actions job, use `sdk-ci-triage`.
If the bump exposes an SDK or integration bug that needs code changes, use `sdk-integrations`.

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

- `py/src/braintrust/integrations/versioning.py` for version-gated behavior
- PR metadata from `gh pr view <pr>`
- changed integration tests and cassette directories for affected providers
- [references/session-mapping.md](./references/session-mapping.md) for matrix-key classification, non-obvious session/cassette mappings, shared-directory constraints, and recording gotchas

## Core Rules

- Work from `py/` for SDK commands.
- Use `mise` as the source of truth for tools and Python versions.
- Check out the actual PR branch with `gh pr checkout <pr-number>`. Do not work from a synthetic `pr-<n>` branch created via `git fetch origin pull/<n>/head:...`; it has no upstream and blocks pushing refreshed cassettes back to the PR.
- Do not guess session names; read `py/noxfile.py`.
- Do not trust the PR label blindly; verify the actual diff in `py/pyproject.toml` and `py/uv.lock`.
- Re-record only affected `latest` cassettes. Do not churn older version directories unless older pins changed or the user asked.
- Review the cassette diff itself, not just pass/fail status.

## Workflow

### 1. Inspect the PR and check out the real branch

Start with metadata and changed files, then switch to the PR branch with upstream tracking:

```bash
gh pr view <pr-number> --repo braintrustdata/braintrust-sdk-python \
  --json title,body,labels,files,headRefName,baseRefName
gh pr checks <pr-number> --repo braintrustdata/braintrust-sdk-python
gh pr checkout <pr-number>
git diff origin/main...HEAD -- py/pyproject.toml py/uv.lock
```

Confirm:

- the PR is the dependency-update workflow output
- the branch tracks `origin/<headRefName>`
- the diff is still narrowly scoped
- the label matches the actual dependency change

### 2. Reproduce the workflow's classification locally

From `py/`, run the same script the workflow uses:

```bash
cd py
python scripts/determine-dependency-update-labels.py
```

Treat the script as the source of truth for workflow labels. Treat your manual diff review as the source of truth for what work to do next.

### 3. Classify the changed keys

Classify each changed matrix key before recording anything:

- infra-only keys: no cassette refresh work
- keys with versioned integration cassettes: usually targeted `latest` playback or re-record work
- keys without versioned integration cassettes: targeted validation only

Use [references/session-mapping.md](./references/session-mapping.md) for the current classification and mappings.

### 4. Check Python-version floors for every bumped pin

Check every changed `latest` pin, patch bumps included: a patch release can raise `Requires-Python` (for example, llama-index-core `0.14.25 -> 0.14.26` moved from `>=3.10` to `>=3.11`). Local playback on the default Python will not catch this. CI fails at install time on the lowest Python in `.github/workflows/checks.yaml`.

Compare the old and new pins' `Requires-Python`:

```bash
cd py
git diff origin/main...HEAD -- pyproject.toml \
  | sed -nE 's/^([-+])latest = "([^"]+)"/\1 \2/p' | sed -E 's/\[[^]]*\]//' \
  | while read -r sign req; do
      name=${req%%==*}; version=${req#*==}
      printf '%s %-30s %-12s %s\n' "$sign" "$name" "$version" \
        "$(curl -s "https://pypi.org/pypi/$name/$version/json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["requires_python"])')"
    done
```

For each pin whose floor rises above CI's lowest Python:

- Add a `latest`-only rule in `py/nox_session_constraints.py`, for example `"test_<name>": {"versions": {"latest": {"min_python": "3.11"}}}`, merged into any existing entry. Older pins keep covering the lower Python.
- Make sure the nox session calls `_skip_if_incompatible(session)`. Matrix-registered sessions already do; hand-written ones may not.
- Check sessions that install another key's `latest` pin through `_install_matrix_dep(..., LATEST)` (grep `noxfile.py`). For example, `test_mcp_interop` installs langchain-core `latest`.
- If the package is in `[tool.braintrust.vendor-packages]`, the `pylint` session installs its `latest` pin too. Add it to the pylint session's minimum-Python handling: either lint lower Pythons against an older matrix pin, or mark the integration unlintable there.
- Verify that the CI sharder drops the session on the lowest Python:

  ```bash
  cd py
  python scripts/nox-matrix.py --dry-run --python-version 3.10 --os ubuntu-latest --exclude-static-checks 0 1
  ```

- Reproduce `pylint` on the real lowest Python. `mise exec python@3.10 -- nox` still runs on the `py/.venv` interpreter, so use a separate envdir:

  ```bash
  env -u VIRTUAL_ENV uvx --python "$(mise where python@3.10)/bin/python3.10" --from nox \
    nox -f ./py/noxfile.py -s pylint --envdir <scratch>/nox310
  ```

### 5. Apply the bump-severity policy

Default policy:

- patch bump: skip re-recording and rely on targeted playback
- minor bump: re-record affected `latest` cassettes
- major bump: re-record and expect follow-on SDK or integration work

For `0.y.z` packages, treat the middle segment as the minor version:

- `0.14.3 -> 0.14.4` is a patch bump
- `0.14.3 -> 0.15.0` is a minor bump
- `0.x -> 1.x` is a major bump

Override the default and re-record a patch bump when:

- the user explicitly asks
- existing `latest` playback fails against the new pin
- release notes call out wire-format or response-shape changes
- the integration is known-fragile and has recent cassette drift

### 6. Research minor and major bumps for SDK changes

For every minor or major bump of a package an integration instruments, check whether the SDK needs to change. Passing playback does not prove the integration still works:

- Tests can pass while tracing silently loses data, such as a token metric or an attachment.
- A test can fail for a reason that re-recording will not fix.

Seen in practice:

- **A patch target was no longer called.** pipecat 1.16 moved segmented STT from `STTService.process_generator` to a new `SegmentedSTTService._push_segment_frames`. The wrapper never ran, and provider spans lost their parent.
- **Returned types changed.** dspy 3.5 moved adapters and LMs onto vendored `lm15` dataclasses. Images were logged as raw base64 and LM spans lost token metrics.
- **A field was removed upstream.** harbor 0.26 dropped `TrialResult.failure_cause`. The plugin tolerated it, but the tests asserted it.
- **A version cap blocked the new pin.** `livekit_agents` had `max_version = "1.8.10"`, so the integration silently stopped patching 1.8.12. `update-matrix-latest.py` does not read integration `max_version` caps, so grep `max_version` under `py/src/braintrust/integrations/` for every bumped key.

Method:

- Install old and new versions side by side, with no dependencies, in separate scratch dirs:

  ```bash
  uv pip install --no-deps --target <scratch>/<pkg>-<old> <pkg>==<old>
  uv pip install --no-deps --target <scratch>/<pkg>-<new> <pkg>==<new>
  ```

  Treat them as untrusted: run any Python that reads them with `-I`.
- Read the integration's `patchers.py`, `tracing.py` and `integration.py`, then diff the upstream code they touch:
  - every patch target: does it still exist, does its signature still bind the same argument names, and is it still on the call path?
  - every private attribute or helper the integration reads
  - the types and shapes it serializes: response and usage fields, streaming event types, content blocks
- Look for new public surfaces worth tracing, and new response or usage fields not yet captured.
- Check `min_version`/`max_version` on the integration and any gates in `py/src/braintrust/integrations/versioning.py`. Fixes must keep every matrix version working: prefer feature detection, and remember patchers skip targets that don't exist.
- For many packages, fan the research out in parallel with one subagent per group of packages. Scope each to scratch dirs: no nox runs or edits in the PR worktree while cassettes are being recorded there.

Classify each package's findings:

- **Required**: breaks or silent data loss. Give file:line evidence in both the SDK and the upstream package.
- **Recommended**: new surfaces or fields worth tracing.
- **Nothing needed**: say what was checked.

Decide what to do with required findings:

- **Small and contained** (a new patcher target, a version-gated test assertion, a removed cap): fix it in the dependency PR, with the failing test as the reproduction. Validate on `latest` and on every older matrix pin for that key.
- **Large** (an integration rework): keep the previous `latest` pin in the PR. Open an issue with the findings and fix plan, and link it from the PR description. Expect the weekly job to bump the pin again until the fix lands.

Treat recommended findings as follow-ups. Do not expand the dependency PR for them.

### 7. Decide the minimum validation scope

Default to the smallest scope that matches the diff:

- one changed provider package: validate or refresh only that provider
- several changed providers: handle each provider independently
- shared cassette directory: group those sessions together and record serially

When several matrix keys share a directory, delete and refresh that directory once, then run every affected session that writes to it.

### 8. Re-record only when justified

For cassette refresh work, delete the affected `latest` directory first so stale responses are not silently reused:

```bash
cd py
rm -rf src/braintrust/integrations/<provider>/cassettes/latest
```

Before recording, neutralize local Braintrust env vars so provider cassettes do not capture unrelated Braintrust traffic:

```bash
unset BRAINTRUST_API_URL
unset BRAINTRUST_APP_URL
unset BRAINTRUST_ORG_NAME
export BRAINTRUST_API_KEY=___TEST_API_KEY__
```

Use the exact nox session from `py/noxfile.py`. Prefer nox over raw pytest so `BRAINTRUST_TEST_PACKAGE_VERSION=latest` is set correctly.

Examples:

```bash
cd py
nox -s "test_openai_agents(latest)" -- --vcr-record=all
nox -s "test_litellm(latest)" -- --vcr-record=all
nox -s "test_langchain(latest)" -- --vcr-record=all
```

For Claude Agent SDK transport recordings:

```bash
cd py
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
```

Provider API keys are expected to exist in this local environment. If a session fails fast with an auth error, stop and ask the user instead of fabricating cassettes.

### 9. Review diffs and scan for pollution

Be suspicious of:

- Braintrust API traffic inside provider cassettes
- missing interactions after a delete-and-record pass
- brand-new endpoint families unrelated to the provider under test
- large request-shape changes that do not match the dependency bump

Quick scan:

```bash
rg -n "staging-api\.braintrust\.dev|api\.braintrust\.dev/version|/logs3|braintrust\.dev/version" \
  py/src/braintrust/integrations/*/cassettes/latest
```

If recording appears to succeed but produces no real cassette updates, read [references/session-mapping.md](./references/session-mapping.md) for provider-specific cache gotchas.

### 10. Validate playback

After recording, rerun playback without forcing record mode:

```bash
cd py
nox -s "test_openai_agents(latest)"
nox -s "test_litellm(latest)"
nox -s "test_langchain(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=none nox -s "test_claude_agent_sdk(latest)"
```

If matrix versions changed in a way that could orphan cassette directories, also run:

```bash
cd py
make check-stale-cassettes
```

If the PR changed broader CI or workflow wiring, expand validation based on `.github/workflows/checks.yaml`.

## Report Back

Always report:

1. whether the PR was infra-only or needed targeted refresh work
2. which matrix keys or provider packages changed
3. which nox sessions you ran
4. which cassette directories or files you deleted and re-recorded
5. any suspicious diff items you investigated
6. whether you found and removed pollution
7. any pins that raise `Requires-Python` above CI's lowest Python, and how they were gated
8. for each minor or major bump: the required SDK changes (fixed in the PR, or held back with an issue link), the recommended follow-ups, or "nothing needed"
9. the final playback or validation result
