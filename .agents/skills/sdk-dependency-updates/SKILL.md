---
name: sdk-dependency-updates
description: "Review and refresh Braintrust Python SDK dependency update PRs, especially automated `chore(deps): weekly dependency update` PRs. Use when Codex needs to inspect `py/pyproject.toml` and `py/uv.lock`, reproduce the workflow's label decision, decide whether provider `latest` cassettes need re-recording, check that bumped pins still install on CI's lowest Python, research whether minor/major bumps need SDK changes, run the exact nox sessions, validate playback before merge, and open a follow-up issue listing the SDK gaps the bumps exposed."
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

Helper scripts (run from the repo root; each takes `--help`):

- [scripts/check_labels.py](./scripts/check_labels.py): the workflow's label decision for a committed PR branch (step 2)
- [scripts/check_requires_python.py](./scripts/check_requires_python.py): bumped pins that newly exclude CI's lowest Python (step 4)
- [scripts/cassette_shape_diff.py](./scripts/cassette_shape_diff.py): endpoint, status and count changes in re-recorded cassettes (step 10)

## Core Rules

- Work from `py/` for SDK commands.
- Use `mise` as the source of truth for tools and Python versions.
- Check out the actual PR branch with `gh pr checkout <pr-number>`. Do not work from a synthetic `pr-<n>` branch created via `git fetch origin pull/<n>/head:...`; it has no upstream and blocks pushing refreshed cassettes back to the PR.
- Do not guess session names; read `py/noxfile.py`.
- Do not trust the PR label blindly; verify the actual diff in `py/pyproject.toml` and `py/uv.lock`.
- Re-record only affected `latest` cassettes. Do not churn older version directories unless older pins changed or the user asked.
- Review the cassette diff itself, not just pass/fail status.
- Report every failure (playback sweep, recording, validation) to the user, with its root cause and a proposed fix, and discuss it before acting. Do not silently restore a cassette, skip a test, or hold a pin to make a failure go away. Fix a failure only after the user approves; the default proposal is to fix the root cause (SDK, test, or recording setup), not to work around it.

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

The script reads the uncommitted working-tree diff (`git diff`). The workflow runs it before committing. On a checked-out PR branch the changes are already committed, so it prints `changed=false`. Reproduce its decision against the merge base instead:

```bash
python3 .agents/skills/sdk-dependency-updates/scripts/check_labels.py   # --base origin/main --head HEAD
```

Treat the script as the source of truth for workflow labels. Treat your manual diff review as the source of truth for what work to do next.

### 3. Classify the changed keys

Classify each changed matrix key before recording anything:

- infra-only keys: no cassette refresh work
- keys with versioned integration cassettes: usually targeted `latest` playback or re-record work
- keys without versioned integration cassettes: targeted validation only

`[tool.braintrust.cassette-dirs]` in `py/pyproject.toml` is the source of truth for which keys own versioned cassettes. Use [references/session-mapping.md](./references/session-mapping.md) for non-obvious session mappings and shared directories.

### 4. Check Python-version floors for every bumped pin

Check every changed `latest` pin, patch bumps included: a patch release can raise `Requires-Python` (for example, llama-index-core `0.14.25 -> 0.14.26` moved from `>=3.10` to `>=3.11`). Local playback on the default Python will not catch this. CI fails at install time on the lowest Python in `.github/workflows/checks.yaml`.

Compare the old and new pins' `Requires-Python` against the lowest `python-version` in `checks.yaml`:

```bash
py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/check_requires_python.py
```

It marks pins that newly exclude the lowest CI Python with `NEW`, and exits 1 if there are any. Pins that already excluded it are listed as `already excluded`; they should already be gated, but confirm when a key is new to the matrix.

For each pin marked `NEW`:

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

### 5. Sweep playback and baseline failures before recording

Automated PRs are opened with `GITHUB_TOKEN`, which does not trigger CI. Until someone pushes to the branch, local playback is the only signal. Before deleting or recording anything, replay every `latest` session against the new pins, not just the changed keys: `uv.lock` churn and cross-pin installs reach further than the matrix diff. Replay costs no API calls and finds most real breaks.

- Run with `CI=true` (forces `record_mode="none"`) and with `BRAINTRUST_API_KEY` unset, as CI does.
- Sessions can run in parallel; give each a separate log. Don't run two sessions that share an external service at once. For example, livekit tests use one local LiveKit server.
- For every failure, re-run the same session on a clean `origin/main` worktree. Use the same env and its own `.nox` directory. A failure there too comes from local env or was already broken; a pass on `main` is a regression from the bump.
- Report each failure with that comparison, its root cause and a proposed fix (see Core Rules), and wait for approval before fixing.

Sessions that skip on the default interpreter need a nox install under an older Python. Check `max_python_exclusive` in `py/nox_session_constraints.py`; for example, sessions capped below 3.14 skip on 3.14:

```bash
uv venv --python 3.13 <scratch>/venv313
VIRTUAL_ENV=<scratch>/venv313 uv pip install nox packaging
cd py && <scratch>/venv313/bin/nox -s "<session>(latest)"
```

When driving nox from a script, put the real `uv` binary ahead of the mise shims on `PATH`. The shim re-applies `mise.toml`'s `py/.venv`, and session installs then land there instead of in `.nox/<session>`. The symptom is `ModuleNotFoundError: No module named 'braintrust'` in the session sanity check.

### 6. Apply the bump-severity policy

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

### 7. Research minor and major bumps for SDK changes

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

Report required findings to the user and agree on the approach before changing anything:

- **Small and contained** (a new patcher target, a version-gated test assertion, a removed cap): fix it in the dependency PR, with the failing test as the reproduction. Validate on `latest` and on every older matrix pin for that key.
- **Large** (an integration rework): keep the previous `latest` pin in the PR. Open an issue with the findings and fix plan, and link it from the PR description. Expect the weekly job to bump the pin again until the fix lands.

Treat recommended findings as follow-ups. Do not expand the dependency PR for them; they go in the follow-up issue (step 13). Keep each finding's file paths, upstream evidence and proposed fix as you go, since the issue needs them.

### 8. Decide the minimum validation scope

Default to the smallest scope that matches the diff:

- one changed provider package: validate or refresh only that provider
- several changed providers: handle each provider independently
- shared cassette directory: group those sessions together and record serially

When several matrix keys share a directory, delete and refresh that directory once, then run every affected session that writes to it.

### 9. Re-record only when justified

Set up the recording environment. Provider keys live in the repo-root `.env`, which mise loads through `_.file`; non-interactive shells such as scripts and agent tool calls do not load it. Leave `BRAINTRUST_API_KEY` unset, as CI does. A placeholder key such as `___TEST_API_KEY__` points the SDK at `api.braintrust.ai` and breaks playback for tests that talk to Braintrust.

```bash
set -a; . ./.env; set +a
unset BRAINTRUST_API_URL BRAINTRUST_APP_URL BRAINTRUST_ORG_NAME BRAINTRUST_API_KEY CI GITHUB_ACTIONS
```

Delete every affected `latest` directory first, all at once, before any nox session starts. Deleting a directory while another session runs `uv pip install -e .` breaks that install. Deleting also stops stale responses from being silently reused.

```bash
cd py
rm -rf src/braintrust/integrations/<provider>/cassettes/latest
```

Then run the exact nox session from `py/noxfile.py` without a record-mode flag. Prefer nox over raw pytest so `BRAINTRUST_TEST_PACKAGE_VERSION=latest` is set. With `CI` unset, the default mode is `once`, which on an empty directory records every cassette fresh. Unlike `--vcr-record=all`, it does not re-overwrite a cassette that several tests share.

```bash
cd py
nox -s "test_litellm(latest)"
nox -s "test_langchain(latest)"
nox -s "test_btx_openai(latest)" -- --vcr-record=all   # btx spec cassettes need explicit record mode
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
```

Groups of sessions that write to different directories can record in parallel. Sessions that share a directory record serially, in one pass.

Provider API keys are expected to exist in this local environment. If a session fails fast with an auth error, stop and ask the user instead of fabricating cassettes.

#### Handle failed recordings

A test that fails while recording still writes its cassette, holding the failed run. Never keep that cassette. Report every recording failure to the user with its root cause and a proposed fix, and fix it once the user approves. Diagnose whether it comes from the bump:

- **Not from the bump:** the test or recording setup is fragile. Propose a fix to the test so it records reliably, then delete its cassette and re-record. Typical causes and fixes:
  - nondeterministic model output that a test asserts on: make the input unambiguous and constrain the output (for example a `Literal` schema field, or a larger test image instead of a 1x1 pixel);
  - state shared across tests, such as a process-wide HTTP client reused on a closed event loop: isolate it in a fixture;
  - upstream latency, such as a test that polls a long-running job past its deadline.
- **From the bump:** fix the SDK or test so the test passes, then keep the new recording.

Restoring the committed cassette with `git checkout -- <file>` or holding the pin (see step 7) is a fallback, not the default. Use it only when the user chooses it, for example when a fix needs credentials or quota that aren't available.

#### Tests that need a real Braintrust org

Some tests record real Braintrust API traffic on purpose: a login to `www.braintrust.dev/api/apikey/login` plus attachment uploads or experiment fetches. To find them:

```bash
cd py
git grep -l 'apikey/login' -- 'src/braintrust/**/cassettes/latest/*'
```

With no key, these tests error during setup with a 401. Restore their cassettes, or re-record them with a real `BRAINTRUST_API_KEY` from `.env`, still with `BRAINTRUST_API_URL`/`BRAINTRUST_APP_URL` unset so they hit the same production hosts as the committed copies. Then confirm the key does not appear in the recorded files:

```bash
grep -cF "$BRAINTRUST_API_KEY" <cassette files>   # expect 0
```

### 10. Review diffs and scan for pollution

Be suspicious of:

- Braintrust API traffic inside provider cassettes, apart from the real-org tests above
- missing interactions after a delete-and-record pass
- brand-new endpoint families unrelated to the provider under test
- large request-shape changes that do not match the dependency bump

Quick scan:

```bash
rg -l "staging-api\.braintrust\.dev|api\.braintrust\.ai|api\.braintrust\.dev/version|/logs3|braintrust\.dev/version" \
  py/src/braintrust/integrations/*/cassettes/latest
```

Any hit outside the real-org files found in step 9 is pollution.

Then compare each re-recorded cassette's structure with the committed copy, not just its text: the set of `(method, URL, status)` interactions and their count. Expect only regenerated resource IDs in URLs. Investigate added or removed endpoints, non-2xx statuses that are new, count changes, and added or deleted files.

```bash
py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/cassette_shape_diff.py   # optionally: <integration> ...
```

If recording appears to succeed but produces no real cassette updates, read [references/session-mapping.md](./references/session-mapping.md) for provider-specific cache gotchas.

### 11. Validate playback

After recording, re-run every affected session in replay-only mode, the way CI runs it:

```bash
cd py
CI=true nox -s "test_litellm(latest)"
CI=true nox -s "test_langchain(latest)"
CI=true BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=none nox -s "test_claude_agent_sdk(latest)"
```

Also re-run:

- every session that shares a re-recorded directory;
- every session you changed code or constraints for, on `latest` and on each older matrix pin for that key;
- `test_core`.

If matrix versions changed in a way that could orphan cassette directories, also run:

```bash
cd py
make check-stale-cassettes
```

If the PR changed broader CI or workflow wiring, expand validation based on `.github/workflows/checks.yaml`.

### 12. Push and update the PR

Pushing to the bot branch from your own account is what first starts CI on the PR, including the lowest-Python shards. Watch that run, and triage failures with `sdk-ci-triage`.

Update the PR description to cover:

- a table of `latest` bumps by type (minor with cassettes, playback only, patch), with any held-back pins and their issue links
- cassette dispositions: re-recorded, recorded against a real org, kept as committed (and why)
- SDK and test changes made for the new pins, plus any Python-floor gating
- the sessions validated locally and how they ran (replay-only, no Braintrust key)
- a `## Follow-ups` section linking the follow-up issue from step 13, or saying the research found nothing

Held-back pins come back: the weekly job bumps every `latest` pin again on its next run. Revert held pins on each new weekly PR until the linked fix lands.

### 13. Open a follow-up issue for SDK gaps

After the PR is updated, open one GitHub issue that collects the gaps the bumps exposed: new upstream surfaces, parameters and fields the SDK doesn't trace yet, and silent data loss that wasn't fixed in the PR. This is the list of SDK work to do next. Example: #1170 (for PR #1162). Earlier ones: #858, #1001, #1088.

Check for an existing follow-up issue for this PR first (`gh issue list --search "Follow-ups from" --state all`). Skip the issue if step 7 found nothing to follow up, and say so in the PR.

Title: `Follow-ups from the <YYYY-MM-DD> dependency bumps: new upstream surfaces to trace`. Use the date of the weekly PR.

Body:

- An opening paragraph: these came out of research on the minor/major bumps in the `<date>` weekly dependency update (#<pr>); say whether any of them blocks that PR; and note that each fix must keep the older matrix pins working, so prefer feature detection.
- One `### <package> <new version>` section per package with findings, in the order of the PR's bump table.
- Under each section, one `- [ ]` task per gap, which can be fixed in its own PR:
  - **a bold one-line summary of the fix** (for example "Log `background_options`.")
  - what changed upstream, with the upstream file path (for example `openai/types/responses/response_create_params.py`)
  - where the SDK misses it, with the SDK path and function (for example `_RESPONSES_METADATA_PARAMS` in `py/src/braintrust/integrations/openai/tracing.py`)
  - the proposed fix, including any version gate or fallback for older pins
  - whether it needs a new or re-recorded cassette, or the wire shape is unconfirmed and needs a recording
- Items that aren't tasks go in plain bullets in the same section: things to watch for when re-recording next time, and `Optional:` surfaces that are low priority or experimental.
- Put gaps you held a pin back for (step 7, **Large**) in their own issue, not this one; this issue can link to it.

Create it with the `python` label, then link it from the PR's `## Follow-ups` section:

```bash
gh issue create --repo braintrustdata/braintrust-sdk-python --label python \
  --title "Follow-ups from the <date> dependency bumps: new upstream surfaces to trace" \
  --body-file <scratch>/followups.md
```

When a later PR fixes an item, tick its box and append `(Merged in #<pr>.)`.

## Report Back

Always report:

1. whether the PR was infra-only or needed targeted refresh work
2. which matrix keys or provider packages changed
3. which nox sessions you ran
4. which cassette directories or files you deleted and re-recorded, and which you restored from git (and why)
5. any suspicious diff items you investigated
6. whether you found and removed pollution
7. any pins that raise `Requires-Python` above CI's lowest Python, and how they were gated
8. for each minor or major bump: the required SDK changes (fixed in the PR, or held back with an issue link), the recommended follow-ups, or "nothing needed"
9. every playback-sweep, recording and validation failure: whether it also fails on `main`, its root cause, the fix proposed, and what the user decided
10. the final playback or validation result
11. the follow-up issue link (step 13), or why none was opened
