# Session Mapping

Read this file when:

- a changed matrix key does not map cleanly to a session or cassette directory
- several changed keys may share one cassette directory
- recording "succeeds" but produces empty, stale, or suspiciously small cassette diffs

## Matrix-key classification

### Infra-only keys

These do not imply provider cassette refresh work:

- `pytest-matrix`
- `braintrust-core`

### Keys with versioned integration cassettes

`[tool.braintrust.cassette-dirs]` in `py/pyproject.toml` maps each integration cassette directory to the matrix keys that own it. It is the source of truth. Don't keep a copy of the list here. Print it with:

```bash
cd py
python3 -c 'import tomllib; [print(d, k) for d, k in tomllib.load(open("pyproject.toml", "rb"))["tool"]["braintrust"]["cassette-dirs"].items()]'
```

### Keys without versioned integration cassettes

Any other matrix key, such as `autoevals`, `temporalio` or `mcp`, still needs targeted validation but has no `integrations/*/cassettes/latest/` directory to refresh.

## Non-obvious mappings

- `google-adk` -> `test_google_adk(latest)` -> `py/src/braintrust/integrations/adk/cassettes/latest/`
- `langchain-core` -> `test_langchain(latest)` -> `py/src/braintrust/integrations/langchain/cassettes/latest/`
- `mistralai` -> `test_mistral(latest)` -> `py/src/braintrust/integrations/mistral/cassettes/latest/`
- `pydantic-ai-integration` -> `test_pydantic_ai_integration(latest)` and `test_pydantic_ai_logfire(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `pydantic-ai-wrap-openai` -> `test_pydantic_ai_wrap_openai(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `claude-agent-sdk` -> `test_claude_agent_sdk(latest)` -> transport cassettes under `py/src/braintrust/integrations/claude_agent_sdk/cassettes/latest/`

For obvious keys, still confirm the exact session name in `py/noxfile.py`.

## Shared-directory rule

A `latest` directory is shared when:

- `cassette-dirs` maps it to more than one key, or
- several nox sessions run tests from the same integration directory.

Grep `py/noxfile.py` for `INTEGRATION_DIR}/<dir>`. Sessions registered through `_register_matrix_sessions` and the btx spec sessions don't match that grep, so check those entries too.

Current shared directories:

- `openai/cassettes/latest/` (including `btx/`): `test_openai`, `test_openai_http2_streaming`, `test_openai_ddtrace`, `test_btx_openai`
- `pydantic_ai/cassettes/latest/`: `test_pydantic_ai_integration`, `test_pydantic_ai_logfire`, `test_pydantic_ai_wrap_openai` (keys `pydantic-ai-integration`, `pydantic-ai-wrap-openai`)
- `langchain/cassettes/latest/`: `test_langchain`, `test_deepagents` (keys `langchain-core`, `deepagents`)
- `bedrock_runtime/cassettes/latest/`: keys `boto3`, `botocore`

Do not split a shared directory across workers or independent recording passes. Delete it once, then run every session that writes to it, serially. Deleting `openai/cassettes/latest` also deletes `btx/`, so re-record `test_btx_openai(latest)` as well.

## Recording gotchas

### dspy cache

`dspy` uses `~/.dspy_cache` via `diskcache`. Clear it before `--vcr-record=all` or the SDK may return cached results and VCR will capture nothing meaningful:

```bash
rm -rf ~/.dspy_cache
```

Symptom: `test_dspy(latest)` appears to pass during recording, but the resulting cassette has no interactions or stale interactions and playback later fails with "no matching cassette".

### Claude Agent SDK

`claude_agent_sdk` does not use HTTP VCR. It records stdin/stdout transport traffic and must be run with:

```bash
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
```

### Pollution symptoms

These are usually local-environment leakage, not meaningful provider behavior:

- `https://staging-api.braintrust.dev/logs3`
- `403 ForbiddenError` from Braintrust logging endpoints
- `api.braintrust.dev/version`
- `staging-api.braintrust.dev/version`
- any `api.braintrust.ai` host (a placeholder `BRAINTRUST_API_KEY` was set)

If these show up in provider cassettes, neutralize local Braintrust env vars and re-record. Tests that log in to a real Braintrust org are the exception: their cassettes contain `www.braintrust.dev/api/apikey/login` and other `api.braintrust.dev` calls on purpose. Find them with `git grep -l 'apikey/login' -- 'py/src/braintrust/**/cassettes/latest/*'`.
