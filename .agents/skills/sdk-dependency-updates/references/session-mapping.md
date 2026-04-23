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

These appear in `[tool.braintrust.cassette-dirs]` and usually map to targeted `latest` playback or re-record work:

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

### Keys without versioned integration cassettes

These still need targeted validation, but do not map to `integrations/*/cassettes/latest/`:

- `autoevals`
- `temporalio`

## Non-obvious mappings

- `google-adk` -> `test_google_adk(latest)` -> `py/src/braintrust/integrations/adk/cassettes/latest/`
- `langchain-core` -> `test_langchain(latest)` -> `py/src/braintrust/integrations/langchain/cassettes/latest/`
- `mistralai` -> `test_mistral(latest)` -> `py/src/braintrust/integrations/mistral/cassettes/latest/`
- `pydantic-ai-integration` -> `test_pydantic_ai_integration(latest)` and `test_pydantic_ai_logfire(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `pydantic-ai-wrap-openai` -> `test_pydantic_ai_wrap_openai(latest)` -> `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`
- `claude-agent-sdk` -> `test_claude_agent_sdk(latest)` -> transport cassettes under `py/src/braintrust/integrations/claude_agent_sdk/cassettes/latest/`

For obvious keys, still confirm the exact session name in `py/noxfile.py`.

## Shared-directory rule

Do not split these across workers or independent recording passes:

- `pydantic-ai-integration`
- `pydantic-ai-wrap-openai`

Both write into `py/src/braintrust/integrations/pydantic_ai/cassettes/latest/`. Delete that directory once, then run all affected sessions serially.

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

If these show up in provider cassettes, neutralize local Braintrust env vars and re-record.
