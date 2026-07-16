---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes. Do not use when migrating an existing legacy wrapper from `py/src/braintrust/wrappers/` into the integrations API; use `sdk-wrapper-migrations` for that.
---

# SDK Integrations

Work under `py/src/braintrust/integrations/`. Use `sdk-wrapper-migrations` instead if the task is moving a real implementation from `py/src/braintrust/wrappers/<provider>/` into the integrations API.

## Spec

Authoritative source for span shape, allowed fields, and instrumentation rules: https://github.com/braintrustdata/braintrust-spec/blob/main/docs/instrumentation-guide.md

Read it first for anything involving span type/name, `input`/`output`/`metadata`/`metrics`/`error`/`context` shape, tool calls, prompt provenance, streaming, reasoning models, prompt caching, multimodal, completion-vs-agentic, or new attribute namespacing. Do not invent new top-level span fields, metric keys, or span types; update the spec first.

## Read First

Always: `integrations/base.py`, `integrations/versioning.py`, `integrations/__init__.py`, `integrations/utils.py`, `pyproject.toml`, `noxfile.py`.

Existing integration: `integrations/<provider>/{__init__,integration,patchers,tracing,test_*}.py`.

Situational:
- `auto.py` + `integrations/auto_test_scripts/` — anything touching `auto_instrument()` or import timing
- `conftest.py`, `integrations/conftest.py` — VCR / per-version cassette resolution
- `integrations/test_utils.py` — shared attachment / multimodal helpers
- `integrations/{adk,anthropic,google_genai}/tracing.py` — attachment + multimodal reference implementations

Import-order and subprocess regressions usually only surface via `auto_test_scripts/`. Do not skip it.

## Version And CI Routing

Never guess versions or session names. The routing chain:

- `pyproject.toml` `[tool.braintrust.matrix]` — supported provider versions, what `latest` resolves to
- `pyproject.toml` `[tool.braintrust.cassette-dirs]` — versioned cassette directory ownership
- `integrations/versioning.py` — supported version helpers, gates
- `noxfile.py` — session names, package install, `BRAINTRUST_TEST_PACKAGE_VERSION`
- `.github/workflows/checks.yaml` — CI matrix + shard membership

When changing version-gated behavior: identify every matrix version, check `min_version`/`max_version`/`superseded_by`/feature-detect branches, test the narrowest affected version first, re-record cassettes only for versions whose observable provider behavior intentionally changed.

## Pick A Reference

Start from the nearest existing integration:

- **ADK** — direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, manual `wrap_*()`, context propagation, inline data → `Attachment`
- **Agno** — multi-target patching, related patchers, version-conditional fallbacks with `superseded_by`
- **Anthropic** — compact constructor patching, small public surface, multimodal image-vs-document attachment payloads
- **Google GenAI** — multimodal tracing, generated media, output-side attachments, nested materialization that preserves non-attachment values

## Default Workflow

1. Read the spec section(s) covering the affected span shape.
2. Read the provider package + shared primitives.
3. Decide completion-style vs agentic-style (see [Span Design](#span-design)).
4. Pick the public API surface to patch — prefer stable public entry points over internal helpers.
5. Define span shape (`type`, `name`, `input`, `output`, `metadata`, `metrics`, `error`) before writing patchers.
6. Implement patchers, then tracing helpers.
7. For behavior changes: add/update a failing test first, then fix (red → green). Cassette-backed if the provider payload triggers the bug.
8. Run the narrowest nox session first. Expand only if shared code changed.

Do not start by wiring patchers and defer span-shape decisions.

## Route The Task

### New integration

1. Create `integrations/<provider>/` with `__init__.py`, `integration.py`, `patchers.py`, `tracing.py`, `test_<provider>.py`, and `cassettes/<version>/` (one dir per matrix version + `latest/`).
2. Export from `integrations/__init__.py`.
3. Add the provider session in `noxfile.py`.
4. Update `auto.py` only if it should participate in `auto_instrument()`; add a subprocess test in `auto_test_scripts/` when it does.
5. Add the `_INSTRUMENTATION = "<provider>-auto"` shadow (see [Span origin](#span-origin)) and assert it in the test suite.

### Existing integration update

Change only affected patchers, tracing, exports, tests, cassettes. Preserve the public `setup_*()` / `wrap_*()` surface. Don't touch `auto.py` / `integrations/__init__.py` / `noxfile.py` unless required. Check whether the change also needs a subprocess auto-instrument test update.

### `auto_instrument()` only

Update `auto.py`. Prefer `_instrument_integration(...)` over a custom `_instrument_*` helper. Add/update the subprocess auto-instrument test.

### Entry-point parity

The three entry points must emit equivalent spans:
- `setup_<provider>()` — package-level patching
- `wrap_*()` helpers — manual wrapping
- `auto_instrument()` — import-order-sensitive discovery

If a behavior change lands in one, check the other two. Validate `auto_instrument()` with a subprocess test, not just in-process.

## Package Layout

Ownership:
- `__init__.py` — public exports, `setup_<provider>()`, `wrap_*()` helpers
- `integration.py` — `BaseIntegration` subclass + patcher registration (keep thin)
- `patchers.py` — patchers and manual wrapping helpers
- `tracing.py` — request/response normalization, metadata extraction, stream handling, error logging
- `test_*.py` — provider behavior tests
- `cassettes/` — VCR recordings

If logic is shared across integrations, put it in `integrations/utils.py`, not copies. Check `utils.py` and neighboring integrations before writing a local helper.

## Integration Rules

Declarative setup on the `BaseIntegration` subclass: `name`, `import_names`, `patchers`. Use `min_version`/`max_version` only when feature detection isn't enough. Prefer `detect_module_version(...)` / `version_satisfies(...)` / `make_specifier(...)`.

Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids; don't hide duplicates.

**Non-invasiveness:**
- Provider errors MUST propagate. Never swallow or rewrap.
- Preserve return types, iterator/async-iterator semantics, generator/subclass behavior.
- Sync and async traced schemas MUST stay aligned when both exist.
- Setup / teardown / patching MUST be idempotent (rely on the base patcher marker).
- Contain instrumentation failures. Extraction/normalization/logging errors must be logged or ignored, never raised into the user's call path.
- Treat provider inputs/results/events/headers as untrusted. Avoid arbitrary attribute access; don't mutate third-party objects.
- Only instrument AI-generation-relevant ops (LLM calls, embeddings, tool exec, media gen, agent runs). No unrelated CRUD.

## Span Design

### Type + name

- `llm` — one provider API call
- `task` — parent for an agent run, pipeline step, or named operation
- `tool` — model-initiated tool/function execution

Use spec-recommended names (`Chat Completion`, `anthropic.messages.create`, `generate_content`). For new providers, pick a stable provider-specific name.

### Completion-style vs agentic-style

- **Completion** (single request/response, user runs any tool calls): one `llm` span, no children. Tool calls appear in `output`.
- **Agentic** (SDK/framework runs the tool loop): one parent `task` wrapping child `llm` (per model call) + `tool` (per tool exec) spans in execution order.

Completion-style *frameworks* (LiteLLM, OpenRouter, etc.) still emit `llm` spans with `metadata.provider` set to the underlying provider, NOT the framework name. Agentic frameworks (Vercel AI SDK w/ tools, LangChain agents, OpenAI Agents SDK, Claude Agent SDK, ADK, Agno) emit the full parent+children tree.

### Payload format

Canonical shape is OpenAI Chat Completions format. Providers with dedicated UI normalizers (OpenAI, Anthropic, Google) MAY preserve provider-native payloads if `metadata.provider` is set correctly; all other providers MUST convert to OpenAI shape. If a provider passes system prompt separately, insert it as a `role: "system"` entry.

### Fields

- `input` — meaningful user request (messages / prompt / provider-native input)
- `output` — meaningful provider result (normalized, not opaque SDK instances)
- `metadata` — see below
- `metrics` — spec-listed keys only (see [Metrics](#metrics))
- `error` — pass the `Exception` instance directly to span logging; do not pre-format

**Every `llm` span MUST include `metadata.model` and `metadata.provider`** (`provider` = whose pricing applies, even when going through a gateway or framework).

**Tool definitions go in `metadata.tools`** (OpenAI-shaped regardless of underlying provider), NOT in `input` messages. Preserve provider-native built-in tool types (Anthropic `computer_use`, OpenAI `web_search`) as opaque JSON-serializable entries. Never log executable tool handlers.

**Prompt provenance goes in `metadata.prompt`** (`id`, `project_id`, `version`, `variables`, plus `prompt_session_id` for playground calls). Not in `input`, request payloads, or `metadata.tools`. Strip carrier fields like `span_info` before logging.

**`metadata` allowlist-per-provider:** capture the spec-defined keys plus provider-specific detail fields (request/response ids, safety data, model-family flags) chosen deliberately. Do not dump entire raw request/response objects. Redact secrets. If a provider-specific field ends up broadly useful, promote it to the spec.

### Token metrics

The integration that directly owns the model/provider API response owns token accounting. Orchestration/framework integrations should not log token metrics when underlying provider integrations create leaf spans with usage. Agentic parent `task` spans MAY aggregate across children when the framework doesn't delegate to a separately instrumented provider client. Do not add "if OpenAI is patched, skip metrics" checks — define clear ownership instead.

### Shaping guidance

- Flatten positional args into named fields; normalize SDK objects into dicts/lists/scalars; drop noisy transport fields.
- Aggregate streaming chunks into one final `output` + stream-specific `metrics`. One `llm` span per API call, not per chunk.
- Do not over-serialize. Braintrust serializes at send/log time. Integration tracing only needs readable Python dicts/lists/scalars and materialized attachments.
- Keep wrapper bodies thin: prepare traced input, open span, call provider, normalize result, log.
- Do not wrap `span.log(...)` / `span.set_attributes(...)` in broad try/except. Braintrust span methods are boundary-safe.

### Span origin

Every span an integration creates MUST carry `context.span_origin.instrumentation.name = "<provider>-auto"` (e.g. `openai-auto`, `anthropic-auto`, `adk-auto`, `google-genai-auto`). Match the JS SDK exactly for cross-SDK filtering.

`start_span(...)` and all provider-level `start_span` methods accept `internal: SpanInternalOptions | None`, a TypedDict reserved for SDK internals — external callers should not use it. Integrations pass `internal={"instrumentation": "<provider>-auto"}`. When unset, spans fall back to the channel default (`braintrust-python-logger` / `braintrust-python-otel`).

The name does NOT propagate through the parent/child edge — each `start_span` call decides independently. User-owned spans nested inside a wrapped call (e.g. a `@traced` scorer) must NOT inherit the integration's name.

Standard pattern at the top of `tracing.py` (or `callbacks.py` / `plugin.py`):

```python
from braintrust.logger import start_span as _bt_start_span

_INSTRUMENTATION = "<provider>-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)
```

Module-level `start_span(...)` calls flow through automatically. Calls that go through a `Logger` / `Experiment` / parent-`Span` instance (`logger.start_span(...)`, `parent.start_span(...)`) need `internal={"instrumentation": _INSTRUMENTATION}` explicitly. Tests assert `span["context"]["span_origin"]["instrumentation"]["name"] == "<provider>-auto"`; `SpanImpl._instrumentation` exposes the resolved value.

## Metrics

Only emit spec-listed keys:
- `tokens`, `prompt_tokens`, `completion_tokens` — required for LLM spans; values MUST be non-negative
- `time_to_first_token` — required for streaming; SDK measures from request start to first chunk
- `completion_reasoning_tokens` — required when provider reports reasoning tokens (o-series etc.)
- `prompt_cached_tokens`, `prompt_cache_creation_tokens`, `prompt_cache_creation_5m_tokens`, `prompt_cache_creation_1h_tokens` — provider prompt caching
- `prompt_audio_tokens`, `completion_audio_tokens`, `completion_image_tokens` — audio/image models when provider reports
- `start`, `end` — standard span timing

For streaming, still produce one span per API call with accumulated `input`/`output`. Capture usage from stream metadata (e.g. OpenAI `stream_options.include_usage`) when surfaced.

For reasoning models, capture the full output structure (reasoning summaries + message blocks) and include prior reasoning in `input` for multi-turn calls.

## Multimodal And Attachments

Inline binary media (images, PDFs, audio, video) MUST be replaced by a Braintrust attachment at the leaf position. Do not add a separate top-level `attachments` list.

- Prefer `_materialize_attachment(...)` in `integrations/utils.py`. Don't reimplement base64/file decoding.
- Convert raw `bytes` / base64 / data URLs / file inputs / generated media → `braintrust.logger.Attachment`. Preserve remote URLs as strings — do not fetch to create an attachment.
- Payload shape after materialization:
  - images → `{"image_url": {"url": attachment}}`
  - non-image media/documents/files → `{"file": {"file_data": attachment, "filename": resolved.filename}}`
- Do not force non-image payloads through `image_url`.
- If materialization fails, keep the original value; never drop, null it out, or raise.
- Preserve non-attachment values while walking nested payloads unless intentionally normalizing.
- Keep MIME type, size, safety data, filenames, provider ids next to the attachment.
- Providers with UI normalizers (OpenAI, Anthropic, Google) — MAY preserve the provider-native structure and only replace the raw media leaf.
- Generated media → log the attachment in `output`, not `metadata`. For streaming, aggregate into one final `output` attachment.

## Patcher Rules

One patcher per coherent target. Prefer:

- `FunctionWrapperPatcher` — one import path / one constructor / one method surface
- `CompositeFunctionWrapperPatcher` — one logical surface across multiple related targets
- `CallbackPatcher` — setup side effects after applicability succeeds

Use `target_module` for patch targets outside the module named by `import_names` (deep or optional submodules). Use `superseded_by` for version-conditional fallbacks — not custom target-selection logic. Use lower `priority` only when ordering matters (e.g. context propagation before tracing).

Manual wrapping helpers stay thin:

```python
def wrap_agent(Agent: Any) -> Any:
    return AgentPatcher.wrap_target(Agent)
```

Every patcher needs: stable `name`, clean existence checks, version gating only when necessary, idempotence via the base patcher marker. Prefer patching stable public API surfaces; internal helpers need version-specific maintenance.

## Testing Rules

Tests live in the provider package. Default bug-fix workflow: red → green — add/update a failing test first, then fix.

**Prefer VCR-backed real provider coverage with `@pytest.mark.vcr`.** The burden of proof is on mocks: use them only for narrow error injection, purely local version-routing logic, patcher existence checks, or provider-independent helpers where the response shape is not part of the contract. Do not swap a cassette-backed regression for a mock just because the fix lives in `tracing.py` or a serializer — if a real payload triggers the bug, the primary regression stays cassette-backed.

Assert on emitted spans (not just provider return values):
- span `type` and `name`
- `input` shape (spec-required messages/prompt/config fields)
- `output` shape (normalized, not opaque SDK instances)
- `metadata.model`, `metadata.provider`, `metadata.tools`/`tool_choice`/`prompt` when present
- required `metrics` keys (tokens for LLM; `time_to_first_token` for streaming)
- parent/child structure for agentic APIs (parent `task`, child `llm`+`tool` in execution order)
- `context.span_origin.instrumentation.name == "<provider>-auto"`
- attachments: images under `image_url.url`, non-images under `file.file_data`, `Attachment` objects (not raw bytes/base64)
- error propagation, error logging
- setup/teardown/wrapping idempotence, patcher resolution when relevant

For streaming, assert both the provider iterator/async-iterator still works AND the final span has aggregated `output` + stream-specific `metrics`.

Cassettes live in `integrations/<provider>/cassettes/<version>/` (e.g. `cassettes/latest/`, `cassettes/0.48.0/`). Nox sets `BRAINTRUST_TEST_PACKAGE_VERSION` so cassettes land correctly. Do not add per-test `vcr_cassette_dir` / `cassette_library_dir` fixtures — `integrations/conftest.py` handles it. Re-record only when behavior intentionally changed. Sanitize cassettes when the provider returns binary bodies.

Confirm the exact session name from `noxfile.py` — don't assume it matches the folder.

## Commands

```bash
cd py && nox -s "test_<session>(latest)"
cd py && nox -s "test_<session>(latest)" -- -k "test_name"
cd py && nox -s "test_<session>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core
cd py && make lint
```

## Validation Checklist

- Narrowest provider session first.
- Subprocess auto-instrument test if patchers / setup / import timing / `auto.py` changed.
- `make test-core` if shared integration code changed.
- `make lint` before handoff if shared files or repo wiring changed.

## Common Mistakes

- Treating a wrapper migration as fresh integration work (use `sdk-wrapper-migrations`).
- Changing shared primitives when provider-local code should own the behavior.
- Combining unrelated patch targets into one patcher.
- Forgetting repo wiring for new providers (`integrations/__init__.py`, `noxfile.py`, sometimes `auto.py`).
- Forgetting subprocess auto-instrument tests, or async/streaming coverage.
- Re-recording cassettes when behavior didn't intentionally change.
- Custom `_instrument_*` helper where `_instrument_integration()` fits.
- Missing `target_module` for deep/optional patch targets.
- Inventing new top-level span fields / metric keys / span types (metadata is looser — see allowlist-per-provider).
- Tool definitions in `input` instead of `metadata.tools`; prompt provenance outside `metadata.prompt`.
- Missing `internal={"instrumentation": "<provider>-auto"}` on integration spans, or leaking it onto user spans nested inside.
- Framework name in `metadata.provider` for a completion-style framework wrapper.
- Per-chunk spans for streamed calls (should be one accumulated span per API call).
- Instrumentation errors escaping into the user's call path, or swallowing real provider errors.
- Double-counting tokens across orchestration + provider spans, or provider-specific ownership checks instead of clear rules.
- Over-serializing (JSON dumps/loads, recursive conversion) in tracing code.
- `try`/`except` around Braintrust span-logging methods — they're boundary-safe.
- Forcing non-image attachments through `image_url` shims; dropping unrecognized file inputs; re-serializing non-attachment values.
- Patching internal helpers when a stable public API surface would work.
