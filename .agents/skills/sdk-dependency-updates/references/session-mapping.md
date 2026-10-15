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

## Provider-specific re-record gotchas

Check this list before deleting a `latest` directory. For general rules on failed recordings, see the "When Recording Fails" section of `sdk-vcr-workflows`.

### Never re-record (restore with `git checkout`)

- **Instructor:** `TestInstructorOpenAISpans.test_instructor_openai_retries_then_succeeds.yaml` is hand-authored to force a validation retry.
- **LiteLLM:** `test_litellm_text_completion_metrics[*]` calls `gpt-3.5-turbo-instruct`, which has been retired (404).
- **OpenAI:** `test_openai_images_edit` is flaky against moderation (`public-figure` block).
- **Anthropic:** `test_anthropic_messages_sync_server_tool_spans` expects `https://www.braintrust.dev/docs` among the live web search results, which the search no longer returns (since Oct 2026).
- **Anthropic:** `test_anthropic_beta_messages_create_captures_compaction_metadata`: the live API now stops with `max_tokens` instead of compaction for this prompt.

### Recording traps

- **Harbor:** `test_atif_import_*` cassettes record Braintrust API traffic, not Harbor traffic. Run `test_harbor` with `BRAINTRUST_API_KEY` unset; `___TEST_API_KEY__` short-circuits login and points at `api.braintrust.ai`, so playback fails with a host mismatch. Do not re-record without a real org.
- **huggingface_hub:** chat tests pin `provider="cerebras"` and text-generation pins `featherless-ai`, for models HF no longer routes there, so re-recording fails with "not supported by provider". The old cassettes still play back. As of 2026-10-13 the local HF token also returns 402 "no remaining credits", so do not try to re-record huggingface_hub.
- **Mistral:** the local `MISTRAL_API_KEY` has hit persistent 429 `rate_limited` (code 1300) for hours at a time. If it does, ask the user for another key instead of retrying in a loop.
- **Anthropic:** `test_anthropic_beta_sessions_events_send_and_stream` runs a live Managed Agents session and can hang while recording. Exclude it with `-k "not test_anthropic_beta_sessions_events_send_and_stream"` and keep its old cassette.
- **Anthropic:** `test_anthropic_messages_reasoning_tokens_metrics` has several parametrizations that share two cassette names and assert against `vcr_cassette.responses[0]`. Under `--vcr-record=all` later cases fail on token counts. Delete just those two cassettes and re-run in default `once` mode.
- **pipecat:** `test_pipecat` skips on Python 3.14 (no onnxruntime wheels), so it records nothing there. Drive nox from a Python 3.13 environment before deleting its cassettes.
- **OpenAI BTX:** `rm -rf integrations/openai/cassettes/latest` also deletes `latest/btx/`, which only `nox -s "test_btx_openai(latest)" -- --vcr-record=all` re-creates. Run it after re-recording openai.

### Coupled sessions

- **dspy depends on the litellm pin:** `test_dspy(latest)` installs the litellm matrix `latest` pin, so a litellm bump can break dspy playback even when the dspy pin is unchanged. Re-record dspy whenever litellm bumps, after `rm -rf ~/.dspy_cache`.
- In general, grep `py/noxfile.py` for `_install_matrix_dep` to find sessions that install a bumped pin without their own matrix key changing.

### Already handled in tests (do not undo)

- **dspy >= 3.4:** `dspy.LM(engine="auto")` routes OpenAI through raw sockets that bypass VCR. `test_dspy.py` pins `engine="litellm"` when the parameter exists. Local playback can falsely pass from `~/.dspy_cache`, so clear it before judging dspy playback.
- **litellm Windows wheels:** vendored tiktoken files have CRLF endings, which causes a runtime fetch of `cl100k_base.tiktoken` on Windows. `test_litellm.py` sets `ignore_hosts` for `openaipublic.blob.core.windows.net`.
