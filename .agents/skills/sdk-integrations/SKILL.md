---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes. Do not use when migrating an existing legacy wrapper from `py/src/braintrust/wrappers/` into the integrations API; use `sdk-wrapper-migrations` for that.
---

# SDK Integrations

Use this skill for integration work under `py/src/braintrust/integrations/`.

Use `sdk-wrapper-migrations` instead when the provider already has a real implementation under `py/src/braintrust/wrappers/<provider>/` and the task is to move that implementation into the integrations API.

## Quick Start

Before editing:

1. Read the Braintrust instrumentation spec (see [Spec](#spec)).
2. Read the shared integration primitives.
3. Read the target provider package.
4. Pick the nearest existing integration as a reference.
5. Confirm provider versions and nox sessions from source files, not memory.
6. Decide the span shape before writing patchers.
7. Run the narrowest provider nox session first.

Do not design a new integration shape from scratch if an existing provider already matches the problem. Do not invent span fields, metrics, or metadata keys that are not already in the spec.

## Spec

The authoritative source for what integrations must capture, how spans must be structured, and which fields are allowed is the Braintrust instrumentation guide:

- https://github.com/braintrustdata/braintrust-spec/blob/main/docs/instrumentation-guide.md

Read the spec first when the task involves:

- span type, name, or hierarchy decisions
- what to put in `input`, `output`, `metadata`, `metrics`, `error`, or `context`
- tool calls, tool results, or available tool definitions
- prompt provenance metadata
- streaming, reasoning models, or prompt caching
- multimodal or attachment payloads
- distinguishing completion-style vs agentic-style APIs
- naming or namespacing new attributes

Instrumentation MUST NOT capture data or emit metric/metadata keys that are not explicitly allowed by the spec. If the task needs a new field, update the spec before shipping the SDK change.

## Read First

Always read:

- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/versioning.py`
- `py/src/braintrust/integrations/__init__.py`
- `py/src/braintrust/integrations/utils.py`
- `py/pyproject.toml` for provider matrix pins and cassette directory mappings
- `py/noxfile.py`

Read these when working on an existing integration:

- `py/src/braintrust/integrations/<provider>/__init__.py`
- `py/src/braintrust/integrations/<provider>/integration.py`
- `py/src/braintrust/integrations/<provider>/patchers.py`
- `py/src/braintrust/integrations/<provider>/tracing.py`
- `py/src/braintrust/integrations/<provider>/test_*.py`

Read these when relevant:

- `py/src/braintrust/auto.py` for `auto_instrument()` changes
- `py/src/braintrust/conftest.py` for VCR behavior
- `py/src/braintrust/integrations/conftest.py` for per-version cassette directory resolution
- `py/src/braintrust/integrations/auto_test_scripts/` for subprocess auto-instrument coverage
- `py/src/braintrust/integrations/test_utils.py` when touching shared attachment materialization or multimodal payload shaping
- `py/src/braintrust/integrations/adk/test_adk.py`, `py/src/braintrust/integrations/anthropic/test_anthropic.py`, and `py/src/braintrust/integrations/google_genai/test_google_genai.py` for attachment-focused test layout patterns
- `py/src/braintrust/integrations/adk/tracing.py`, `py/src/braintrust/integrations/anthropic/tracing.py`, and `py/src/braintrust/integrations/google_genai/tracing.py` when handling multimodal content, binary inputs, generated media, or attachment materialization behavior

Do not forget `auto.py` and `auto_test_scripts/`. Import-order and subprocess regressions often only show up there.

## Version And CI Routing

Do not guess which provider versions or sessions apply.

Use these files as the routing chain:

- `py/pyproject.toml` `[tool.braintrust.matrix]`: supported provider versions and what `latest` resolves to
- `py/pyproject.toml` `[tool.braintrust.cassette-dirs]`: versioned cassette directory ownership
- `py/src/braintrust/integrations/versioning.py`: supported version helpers and gates
- `py/noxfile.py`: actual session names, package installation, and `BRAINTRUST_TEST_PACKAGE_VERSION`
- `.github/workflows/checks.yaml`: CI matrix and which sessions run in shards or static checks

When changing version-gated behavior:

1. Identify every matrix version for the provider.
2. Check whether the integration has `min_version`, `max_version`, `superseded_by`, or feature-detection branches.
3. Test the narrowest affected version first.
4. Add or update cassettes only for versions whose observable provider behavior intentionally changed.

## Pick A Reference

Start from the nearest current integration:

- ADK: direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, manual `wrap_*()` helpers, context propagation, inline data to `Attachment`
- Agno: multi-target patching, several related patchers, version-conditional fallbacks with `superseded_by`
- Anthropic: compact constructor patching, a small public surface, and multimodal request blocks that distinguish image vs document attachment payloads
- Google GenAI: multimodal tracing, generated media, output-side `Attachment` handling, and nested attachment materialization while preserving non-attachment values

Choose the reference based on the hardest part of the task:

- patcher topology
- tracing shape
- streaming behavior
- multimodal or binary payload handling

## Default Workflow

Use this order unless the task is clearly narrower:

1. Read the spec sections that cover the affected span shape.
2. Read shared primitives and the provider package.
3. Decide whether the target API is completion-style or agentic-style (see [Span Design Rules](#span-design-rules)).
4. Decide which public surface is being patched. Prefer stable public API entry points over internal helpers; internal targets break more often across provider versions.
5. Define the span shape:
   - span `type` and `name`
   - `input`
   - `output`
   - `metadata`
   - `metrics`
   - `error` when failures matter
6. Implement or update patchers.
7. Implement or update tracing helpers.
8. Add or update focused tests.
9. For provider-behavior bugs, make the primary regression test cassette-backed when practical, even if the implementation change is in local tracing/span post-processing of the provider response.
10. Be suspicious of mock/fake coverage for integrations. Do not choose mocks because they are convenient, faster, or easier to control.
11. Run the narrowest nox session first, then expand only if shared code changed.

Do not start by wiring wrappers and only later decide what the span should contain.

For behavior changes, prefer adding or adjusting a failing test first, then implementing until it passes.

## Route The Task

### New provider integration

1. Create `py/src/braintrust/integrations/<provider>/`.
2. Use this layout unless the provider is exceptionally small:
   - `__init__.py`
   - `integration.py`
   - `patchers.py`
   - `tracing.py`
   - `test_<provider>.py`
   - `cassettes/<version>/` when the provider uses HTTP (one subdirectory per version in the nox matrix, plus `latest/`)
3. Export the integration from `py/src/braintrust/integrations/__init__.py`.
4. Add or update the provider session in `py/noxfile.py`.
5. Update `py/src/braintrust/auto.py` only if the integration should participate in `auto_instrument()`.
6. Add subprocess coverage in `py/src/braintrust/integrations/auto_test_scripts/` when `auto_instrument()` changes.
7. Add the `_INSTRUMENTATION = "<provider>-auto"` shadow described in [Span origin](#span-origin-instrumentationname) at the top of `tracing.py`, and include an assertion in the integration's test suite that `context.span_origin.instrumentation.name` matches.

### Existing integration update

1. Read the current provider package before editing.
2. Change only the affected patchers, tracing helpers, exports, tests, and cassettes.
3. Preserve the provider's public setup and `wrap_*()` surface unless the task explicitly changes it.
4. Do not touch `auto.py`, `integrations/__init__.py`, or `py/noxfile.py` unless the task requires it.
5. Even if `auto.py` does not change, check whether the behavior change also needs an auto-instrument subprocess test update.
6. Preserve existing span shape conventions unless the task is intentionally correcting them to match the spec.

### `auto_instrument()` only

1. Update `py/src/braintrust/auto.py`.
2. Prefer `_instrument_integration(...)` over a custom `_instrument_*` helper when the standard pattern fits.
3. Add the integration import near the other integration imports.
4. Add or update the relevant subprocess auto-instrument test.

### Setup, manual wrapping, and auto-instrument

Treat these as distinct entry points:

- `setup_<provider>()`: explicit package-level patching
- public `wrap_*()` helpers: manual wrapping of a provided class, function, or client
- `auto_instrument()`: import-order-sensitive discovery and setup

When changing one entry point, check whether the other two should keep equivalent span behavior. Auto-instrument and manual paths should emit equivalent spans; if a provider-facing behavior change goes into one path, it usually needs to go into the other. If `auto_instrument()` changes or could be affected by import timing, validate it with a subprocess test instead of only calling the integration in-process.

## Package Layout Rules

Keep provider-specific behavior in `py/src/braintrust/integrations/<provider>/`.

Typical ownership:

- `__init__.py`: public exports, `setup_<provider>()`, public `wrap_*()` helpers
- `integration.py`: `BaseIntegration` subclass and patcher registration
- `patchers.py`: patchers and manual `wrap_*()` helpers
- `tracing.py`: request/response normalization, metadata extraction, stream handling, error logging
- `test_*.py`: provider behavior tests
- `cassettes/`: VCR recordings for provider HTTP traffic

Keep `integration.py` thin.

If logic is genuinely shared across integrations, move it to `py/src/braintrust/integrations/utils.py` instead of copying it into multiple providers. Before adding a local helper, check `utils.py` and neighboring integrations for an existing one.

## Integration Rules

Set up the integration declaratively:

- set `name`
- set `import_names`
- set `patchers`
- set `min_version` or `max_version` only when feature detection is not enough

Prefer feature detection first and version checks second. Use:

- `detect_module_version(...)`
- `version_satisfies(...)`
- `make_specifier(...)`

Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids. Do not hide duplicates.

### Non-invasiveness and safety

Instrumentation must not change user-visible behavior:

- Errors from the provider MUST propagate to the caller. Do not swallow or rewrap them.
- Return values, iterator/async-iterator semantics, and generator/subclass behavior MUST be preserved.
- Sync and async traced schemas MUST stay aligned when the provider exposes both.
- Setup, teardown, and patching MUST be idempotent. Applying setup twice, tearing down twice, or wrapping twice must remain safe. Rely on the base patcher's idempotence marker.
- Contain instrumentation failures. Errors in extraction, normalization, or logging code must be logged or ignored, not raised into the user's call path. Only intentional provider errors should surface.
- Treat provider inputs, results, events, headers, and metadata as untrusted. Avoid attribute access that could trigger arbitrary code, and avoid mutating third-party objects.
- Limit instrumentation to AI-generation-relevant operations (LLM calls, embeddings, tool executions, media generation, agent runs). Do not instrument unrelated CRUD or platform-management APIs.

### Scope of instrumentation

Follow the spec closely for spec-defined fields — `input`, `output`, `metrics`, `error`, tool-call structure, prompt provenance, attachments. Do not invent new top-level fields, metric keys, or span types; update the spec first if you genuinely need one.

`metadata` is treated slightly more loosely: use an **allowlist per provider**, not a denylist. Capture the spec-defined keys (`model`, `provider`, `tools`, `tool_choice`, `parallel_tool_calls`, `max_tool_calls`, `prompt`, and the permitted request-config subset) plus provider-specific detail fields you deliberately choose to include — provider request/response ids, safety data, model-family flags, and similar. Do not dump entire raw request/response objects into metadata; each captured key should be an intentional decision. Redact obvious secrets. If a provider-specific field ends up broadly useful across SDKs, promote it to the spec.

Preserve provider behavior. Tracing code must not change return values, control flow, or error behavior unless the task explicitly requires it.

### Span origin (`instrumentation.name`)

Every span an integration creates MUST carry `context.span_origin.instrumentation.name` identifying which integration produced it. Naming convention: `<provider>-auto` (e.g. `openai-auto`, `anthropic-auto`, `adk-auto`, `google-genai-auto`). Match the JS SDK exactly for cross-SDK dashboards to work.

Braintrust's `start_span(...)` (and every provider-level `start_span` method) accepts an `internal: dict | None` kwarg reserved for SDK internals. Integrations pass `internal={"instrumentation": "<provider>-auto"}` to stamp `context.span_origin.instrumentation.name`. When unset, spans fall back to the channel default (`braintrust-python-logger` for direct logging, `braintrust-python-otel` for the OTel processor). The `internal` dict is intentionally opaque to signal that external callers should not use it — the keys inside are unstable.

**Only integration-created spans should carry the integration's name.** User-owned spans nested inside a wrapped provider call — for example a `@traced` scorer running inside a wrapped agent turn — must NOT inherit the integration's `instrumentation.name`. The value does not propagate through the span parent/child edge; each `start_span` call independently decides its own value.

The standard per-integration pattern is a small shadow at the top of the integration's `tracing.py` (or `callbacks.py`/`plugin.py`):

```python
from braintrust.logger import start_span as _bt_start_span

_INSTRUMENTATION = "<provider>-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)
```

Every direct `start_span(...)` call site inside the module then flows through the shadow automatically. If the integration opens spans through a `Logger`, `Experiment`, or existing `Span` instance (i.e. `logger.start_span(...)`, `parent.start_span(...)`), pass `internal={"instrumentation": _INSTRUMENTATION}` explicitly at those sites — the shadow only catches module-level `start_span` calls.

Tests should assert on the emitted span's `context.span_origin.instrumentation.name`. `SpanImpl` also exposes the resolved value as `span._instrumentation` for direct inspection in tests.

## Span Design Rules

### Span type and name

Every span must set the correct `type` and a stable `name`:

- `llm`: a single provider API call
- `task`: a parent span for an agent run, pipeline step, or named operation
- `tool`: a model-initiated tool/function execution

Use spec-recommended names where they exist (for example `Chat Completion` for OpenAI, `anthropic.messages.create` for Anthropic, `generate_content` for Google). For new providers, pick a stable, provider-specific name.

### Completion-style vs agentic-style

Decide the span shape before writing patchers:

- Completion-style APIs (single request/response, user executes any tool calls): one `llm` span per API call, no children. Tool calls the model requests appear in the span's `output`; the SDK does not create `tool` spans.
- Agentic-style APIs (the SDK/framework runs the tool loop internally): one parent `task` span wrapping child `llm` spans (one per underlying model call) and child `tool` spans (one per tool execution). Preserve execution order.

For frameworks: completion-style frameworks (for example LiteLLM) must produce provider-shaped `llm` spans and set `metadata.provider` to the underlying provider (`openai`, `anthropic`, etc.), not the framework name. Agentic-style frameworks (Vercel AI SDK with tools, LangChain agents, OpenAI Agents SDK, Claude Agent SDK, ADK, Agno) must produce the full parent-plus-children tree.

### Canonical payload format

Braintrust uses the OpenAI Chat Completions message format as the canonical representation for LLM `input` and `output`. The UI parses, displays, and diffs spans assuming this format.

- For providers that have a dedicated Braintrust UI normalizer (currently OpenAI, Anthropic, Google), instrumentation MAY preserve the provider-native payload. Set `metadata.provider` correctly so the UI can normalize.
- For any other provider, instrumentation MUST convert payloads into the OpenAI Chat Completions format so the UI can render them. If the provider exposes a system prompt as a separate parameter, insert it into the messages array as a `role: "system"` entry.

### Input, output, metadata, metrics, error

Build readable spans. Do not dump raw `args` and `kwargs` unless the provider API already exposes a clean schema.

Use this rubric:

- `input`: the meaningful user request (messages, prompt, or provider-native input structure)
- `output`: the meaningful provider result (choices, message, or provider-native response structure)
- `metadata`: only spec-allowed fields such as `model`, `provider`, `tools`, `tool_choice`, `parallel_tool_calls`, `max_tool_calls`, `prompt`, and the permitted request-config subset (`temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `stop`, `response_format`)
- `metrics`: only spec-allowed keys (see [Metrics](#metrics))
- `error`: exceptions or failure information; pass the `Exception` instance directly to span logging rather than pre-formatting it into strings

Every `llm` span MUST include `metadata.model` (the resolved model id from the response, including any version suffix when the API returns one) and `metadata.provider` (the provider whose pricing applies, even when the caller went through a gateway or framework).

Tool definitions are request configuration, not conversation content: the schemas passed to the model MUST go in `metadata.tools` (OpenAI-shaped, regardless of the underlying provider), not in the `input` messages. Preserve provider-native built-in tool types (for example Anthropic `computer_use`, OpenAI `web_search`) as opaque JSON-serializable entries in `metadata.tools`. Do not log executable tool handlers or closures.

Prompt provenance for Braintrust-managed prompts MUST go in `metadata.prompt` (`id`, `project_id`, `version`, `variables`, and `prompt_session_id` for playground/prompt-session calls). Do not put prompt provenance in `input`, request payloads, or `metadata.tools`. Wrapper-only carrier fields such as `span_info` are internal plumbing; strip them from the provider request and do not log them.

### Token metrics

Avoid double-counting token metrics:

- the integration that directly owns the model/provider API response should own token accounting
- orchestration/framework integrations should usually not log token metrics when underlying provider integrations can create leaf spans with usage metrics; agentic parent `task` spans MAY aggregate token counts across children when the framework does not delegate to a separately instrumented provider client
- do not add fragile provider-specific ownership checks such as "if OpenAI is patched, skip metrics"; prefer a clear span ownership rule instead

### Shaping guidance

Good span shaping usually means:

- flatten positional arguments into named fields
- normalize provider SDK objects into dicts, lists, or scalars when that improves readability
- drop duplicate or noisy transport fields
- aggregate streaming chunks into one final `output` plus stream-specific `metrics`

Do not over-serialize in integration code. Braintrust handles serialization when sending/logging spans, so integration tracing helpers usually only need to shape readable Python dicts/lists/scalars and materialize attachments where appropriate. Avoid unnecessary JSON dumps/loads, recursive conversion, or stringification just to make values serializable.

Keep wrapper bodies thin: prepare traced input, open the span, call the provider, normalize the result, and log `output`/`metadata`/`metrics`.

Braintrust span logging methods are boundary-safe and should not throw during normal integration use. Do not wrap `span.log(...)`, `span.set_attributes(...)`, or similar Braintrust span methods in broad `try`/`except` blocks. Only catch exceptions around provider calls or around integration-owned conversion code when there is a specific expected failure mode and a clear fallback.

Prefer provider-local helpers in `tracing.py`, for example:

```python
def _prepare_traced_call(args: list[Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ...


def _process_result(result: Any, start: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ...
```

## Metrics

Only emit spec-listed metric keys. Do not invent metric keys.

Standard LLM metrics:

- `tokens`, `prompt_tokens`, `completion_tokens`: required for LLM spans (values MUST be non-negative)
- `time_to_first_token`: required for streaming spans; measured by the SDK from request start to the first chunk
- `completion_reasoning_tokens`: required when the provider reports reasoning tokens (for example OpenAI o-series)
- `prompt_cached_tokens`, `prompt_cache_creation_tokens`, `prompt_cache_creation_5m_tokens`, `prompt_cache_creation_1h_tokens`: capture when the provider reports prompt caching
- `prompt_audio_tokens`, `completion_audio_tokens`, `completion_image_tokens`: capture when the provider reports them for audio/image models
- `start`, `end`: standard span timing

Streaming spans still produce a single `llm` span per API call, with `input`/`output` accumulated from chunks. Capture token counts from stream usage metadata when the provider surfaces it (for example OpenAI `stream_options.include_usage`).

For reasoning models, capture the full output structure (reasoning summaries plus assistant message blocks) when the provider exposes them, and include prior reasoning blocks in the `input` for multi-turn calls.

## Multimodal And Attachments

When a request or response contains inline binary media (images, PDFs, audio, video, and similar), replace the raw media leaf in the span payload with a Braintrust attachment. Do not add a separate top-level `attachments` list.

Treat binary payloads as attachments, not logged bytes:

- prefer the shared `_materialize_attachment(...)` helper in `py/src/braintrust/integrations/utils.py` over provider-local base64 or file-decoding code
- convert provider-owned raw `bytes`, base64 payloads, data URLs, file inputs, and generated media into `braintrust.logger.Attachment` objects when Braintrust should upload the content (the SDK converts these into `braintrust_attachment` references on the wire)
- preserve normal remote URLs as strings; do not fetch remote URLs solely to create an attachment
- use the repo's existing multimodal payload shapes after materialization:
  - images -> `{"image_url": {"url": attachment}}`
  - non-image media/documents/files -> `{"file": {"file_data": attachment, "filename": resolved.filename}}`
- do not force non-image payloads through `image_url` shims
- if attachment materialization or upload fails, keep the original value instead of dropping it or replacing it with `None`, and do not raise into the user's call path
- preserve non-attachment values while walking nested payloads unless you are intentionally normalizing them for readability
- keep useful metadata such as MIME type, size, safety data, filenames, or provider ids next to the attachment
- when a provider-native payload has a dedicated UI normalizer (OpenAI, Anthropic, Google), you may preserve the provider-native structure and replace only the raw media leaf

For generated media on the output side, log the attachment inside `output` (not `metadata`). For streaming media outputs, aggregate chunks into the final `output` attachment; do not log raw chunks as separate blobs.

## Patcher Rules

Create one patcher per coherent patch target.

Prefer:

- `FunctionWrapperPatcher` for one import path or one constructor/method surface
- `CompositeFunctionWrapperPatcher` for one logical surface spread across multiple related targets
- `CallbackPatcher` for setup side effects after applicability succeeds

Use `target_module` when the patch target lives outside the module named by `import_names`, especially for optional or deep submodules.

Use `superseded_by` for version-conditional fallbacks instead of custom target-selection logic.

Use lower `priority` only when patch ordering really matters, such as context propagation before tracing.

Manual wrapping helpers should be thin:

```python
def wrap_agent(Agent: Any) -> Any:
    return AgentPatcher.wrap_target(Agent)
```

Require every patcher to have:

- a stable `name`
- clean existence checks
- version gating only when necessary
- idempotence through the base patcher marker

Prefer patching stable public API surfaces (documented client methods, top-level constructors). Reach into internal helpers only when the public surface is genuinely insufficient, and expect those patches to need version-specific maintenance.

## Testing Rules

Keep tests in the provider package.

Default bug-fix workflow: red -> green.

- First add or update a focused test that reproduces the integration bug.
- Then implement the fix.
- Only skip this when the task explicitly asks for a different approach.

Prefer VCR-backed real provider coverage with `@pytest.mark.vcr`. This includes span-shaping/tracing bugs where the bad behavior is triggered by a real provider response payload; do not treat those as mock-first just because the code path is local.

Default stance: if the behavior is provider-facing, assume mocks/fakes are the wrong tool until proven otherwise. A mock should need justification, not the other way around.

Use mocks or fakes only for cases that are hard to drive through recordings, such as:

- narrow error injection
- purely local version-routing logic
- patcher existence checks
- provider-independent helper logic where the provider response shape is not part of the contract being validated

Do not replace or skip a cassette-backed regression with a mock/fake test merely because the implementation change lives in `tracing.py`, a serializer, or another local post-processing layer. If a real provider payload is what triggers the bug, the main regression test should reflect that real payload.

Test emitted spans, not just provider return values. In particular, assert on:

- span `type` and `name`
- `input` shape (messages/prompt/config fields the spec requires)
- `output` shape (normalized provider result, not opaque SDK instances)
- `metadata.model`, `metadata.provider`, and any `metadata.tools` / `metadata.tool_choice` / `metadata.prompt` present
- required `metrics` keys (token counts for LLM spans; `time_to_first_token` for streaming)
- parent/child structure for agentic APIs (parent `task`, child `llm` and `tool` spans in execution order)
- `context.span_origin.instrumentation.name` equals the integration's `<provider>-auto` identifier
- attachment conversion for binary inputs or generated media, including that images land under `image_url.url`, non-image payloads land under `file.file_data`, and traced payloads contain `Attachment` objects rather than raw bytes or base64 blobs
- error propagation and error logging behavior
- idempotence of setup/teardown/wrapping
- patcher resolution and duplicate detection when relevant

Cover the surfaces that changed:

- direct `wrap_*()` behavior
- setup-time patching
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and error logging

For streaming changes, verify both:

- the provider still returns the expected iterator or async iterator
- the final logged span contains the aggregated `output` and stream-specific `metrics`

Keep VCR cassettes in `py/src/braintrust/integrations/<provider>/cassettes/<version>/` (e.g. `cassettes/latest/`, `cassettes/0.48.0/`). Nox sessions set `BRAINTRUST_TEST_PACKAGE_VERSION` automatically so cassettes land in the correct version subdirectory. Do not add per-test `vcr_cassette_dir` or `cassette_library_dir` fixtures; the shared `py/src/braintrust/integrations/conftest.py` handles version routing. Re-record only when behavior intentionally changes.

When the provider returns binary HTTP responses or generated media, sanitize cassettes as needed so fixtures do not store raw file bytes.

When choosing test commands, confirm the actual session name in `py/noxfile.py` instead of assuming it matches the provider folder.

## Commands

```bash
cd py && nox -s "test_<session>(latest)"
cd py && nox -s "test_<session>(latest)" -- -k "test_name"
cd py && nox -s "test_<session>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core
cd py && make lint
```

## Validation Checklist

- Run the narrowest provider session first.
- If the change touches patchers, setup behavior, import timing, or anything that could affect `auto_instrument()`, run the relevant subprocess auto-instrument test from `py/src/braintrust/integrations/auto_test_scripts/`.
- Run the relevant auto-instrument subprocess test if `auto.py` changed.
- Run `cd py && make test-core` if shared integration code changed.
- Run `cd py && make lint` before handoff when shared files or repo-level wiring changed.

## Common Mistakes

Avoid these failures:

- treating a wrapper migration as fresh integration work
- changing shared integration primitives when provider-local code should own the behavior
- combining unrelated patch targets into one patcher
- forgetting repo-level wiring for new providers: `integrations/__init__.py`, `py/noxfile.py`, and sometimes `auto.py`
- forgetting the subprocess auto-instrument tests
- forgetting async or streaming coverage
- re-recording cassettes when behavior did not intentionally change
- adding a custom `_instrument_*` helper where `_instrument_integration()` already fits
- forgetting `target_module` for deep or optional patch targets
- inventing new top-level span fields, metric keys, or span types beyond what the spec allows (metadata is treated more loosely — see the metadata allowlist guidance above)
- putting tool definitions in `input` instead of `metadata.tools`, or including prompt provenance anywhere other than `metadata.prompt`
- forgetting to set `instrumentation=<provider>-auto` on spans an integration opens, or leaking that identifier onto user-owned spans nested inside a wrapped call
- using the framework name as `metadata.provider` for a completion-style framework wrapper instead of the underlying provider
- producing per-chunk spans for a streamed call instead of one accumulated span per API call
- letting instrumentation errors surface into the user's call path, or swallowing real provider errors
- double-counting token metrics in both orchestration/framework spans and provider leaf spans
- adding provider-specific token ownership detection instead of defining clear metric ownership for the integration
- doing excessive serialization/stringification in tracing code even though Braintrust serializes span payloads at send/log time
- wrapping Braintrust span logging methods in broad `try`/`except` blocks even though those methods are designed not to throw
- forcing non-image attachments through `image_url` shims, dropping unrecognized file inputs, or re-serializing non-attachment values while materializing payloads
- patching internal helpers when a stable public API surface would work
