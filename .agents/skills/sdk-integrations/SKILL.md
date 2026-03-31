---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes. Do not use when migrating an existing legacy wrapper from `py/src/braintrust/wrappers/` into the integrations API; use `sdk-wrapper-migrations` for that.
---

# SDK Integrations

Use this skill for integrations API work under `py/src/braintrust/integrations/`.

If the provider already has a real implementation under `py/src/braintrust/wrappers/<provider>/` and the task is to move that implementation into the integrations API, switch to `sdk-wrapper-migrations` instead of treating it like a fresh integration.

## Pick The Nearest Example

Start from one structural reference and one patching reference instead of designing from scratch:

- ADK (`py/src/braintrust/integrations/adk/`) for direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, manual `wrap_*()` helpers, priority-based context propagation, and input-side `inline_data` to `Attachment` conversion.
- Agno (`py/src/braintrust/integrations/agno/`) for multi-target patching, version-conditional fallbacks with `superseded_by`, and providers that need several related patchers.
- Anthropic (`py/src/braintrust/integrations/anthropic/`) for constructor patching and a compact provider package with a small public surface.
- Google GenAI (`py/src/braintrust/integrations/google_genai/`) for multimodal serialization, generated media outputs, and output-side `Attachment` handling.

Match an existing repo pattern unless the target provider forces a different shape.

Choose the example based on the hardest part of the task, not just provider similarity:

- If the task is mostly about patcher topology, copy the closest patcher layout first.
- If the task is mostly about traced payload shaping, copy the closest tracing implementation first.
- If the task involves generated media or multimodal payloads, start from ADK or Google GenAI before looking at simpler text-only integrations.

## Read First

Always read:

- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/versioning.py`
- `py/src/braintrust/integrations/__init__.py`
- `py/noxfile.py`

Read when updating an existing integration:

- `py/src/braintrust/integrations/<provider>/__init__.py`
- `py/src/braintrust/integrations/<provider>/integration.py`
- `py/src/braintrust/integrations/<provider>/patchers.py`
- `py/src/braintrust/integrations/<provider>/tracing.py`
- `py/src/braintrust/integrations/<provider>/test_*.py`

Read when relevant:

- `py/src/braintrust/auto.py` for `auto_instrument()` work
- `py/src/braintrust/conftest.py` for VCR behavior
- `py/src/braintrust/integrations/auto_test_scripts/` for subprocess auto-instrument tests
- `py/src/braintrust/integrations/adk/test_adk.py` and `py/src/braintrust/integrations/anthropic/test_anthropic.py` for test layout patterns
- `py/src/braintrust/integrations/adk/tracing.py` and `py/src/braintrust/integrations/google_genai/tracing.py` when the provider accepts binary inputs, emits generated files, or otherwise needs `Attachment` objects in traced input/output

## Working Sequence

Use this order unless the task is obviously narrower:

1. Read the nearest provider package and the shared integration primitives.
2. Decide which public surface is being patched: constructor, top-level function, client method, stream method, or manual `wrap_*()` helper.
3. Decide what the span should look like before writing patchers:
   - what belongs in `input`
   - what belongs in `output`
   - what belongs in `metadata`
   - what belongs in `metrics`
4. Implement or update patchers.
5. Implement or update tracing helpers.
6. Add or update focused tests in the provider package.
7. Run the narrowest nox session first, then expand only if shared code changed.

Do not start by wiring patchers and only later asking what the logged span should contain. The traced shape should drive the tracing helper design from the start.

## Route The Task

### New provider integration

1. Create `py/src/braintrust/integrations/<provider>/`.
2. Add the normal split unless the provider is exceptionally small:
   - `__init__.py`
   - `integration.py`
   - `patchers.py`
   - `tracing.py`
   - `test_<provider>.py`
   - `cassettes/` when the provider uses HTTP
3. Export the integration from `py/src/braintrust/integrations/__init__.py`.
4. Add or update the provider session in `py/noxfile.py`.
5. Update `py/src/braintrust/auto.py` only if the integration should participate in `auto_instrument()`.
6. Add subprocess coverage in `py/src/braintrust/integrations/auto_test_scripts/` when `auto_instrument()` changes.

### Existing integration update

1. Read the current provider package before editing.
2. Change only the affected patchers, tracing helpers, exports, tests, and cassettes.
3. Preserve the provider's public setup and `wrap_*()` surface unless the task explicitly changes it.
4. Keep repo-level changes narrow; do not touch `auto.py`, `integrations/__init__.py`, or `py/noxfile.py` unless the task actually requires it.
5. Preserve existing span shape conventions unless the task is intentionally improving or correcting them.

### `auto_instrument()` only

1. Update `py/src/braintrust/auto.py`.
2. Use `_instrument_integration(...)` instead of adding a custom `_instrument_*` helper when the integration fits the standard pattern.
3. Add the integration import near the other integration imports.
4. Add or update the relevant subprocess auto-instrument test.

## Package Layout

Keep provider-local code inside `py/src/braintrust/integrations/<provider>/`.

Typical file ownership:

- `__init__.py`: export the integration class, `setup_<provider>()`, and public `wrap_*()` helpers
- `integration.py`: define the `BaseIntegration` subclass and register patchers
- `patchers.py`: define patchers and manual `wrap_*()` helpers
- `tracing.py`: keep provider-specific tracing, stream handling, normalization, and metadata extraction
- `test_*.py`: keep provider behavior tests next to the integration
- `cassettes/`: keep VCR recordings next to the integration tests when the provider uses HTTP

Keep `integration.py` thin. Put provider behavior in provider-local modules, not in shared integration primitives, unless the shared abstraction is genuinely missing.

## Integration Rules

Set the integration class up declaratively:

- set `name`
- set `import_names`
- set `patchers`
- set `min_version` or `max_version` only when feature detection is not enough

Keep span creation, metadata extraction, stream aggregation, error logging, and output normalization in `tracing.py`.

Preserve provider behavior. Do not let tracing-only code change provider return values, control flow, or error behavior except where the task explicitly requires it.

Generate structured spans. Do not pass raw `args` and `kwargs` straight into traced spans unless the provider API already exposes the exact stable schema you want to log. Instead:

- Build a provider-shaped `input` object that names the important request fields explicitly, for example model, messages/contents, prompt, config, tools, or options.
- Build an `output` object that captures the useful response payload in normalized form instead of logging opaque SDK objects.
- Put secondary facts in `metadata`, such as provider ids, finish reasons, model versions, safety attributes, or normalized request/response annotations that are useful but not the primary payload.
- Put timings and token/accounting values in `metrics`, such as `start`, `end`, `duration`, `time_to_first_token`, `prompt_tokens`, `completion_tokens`, and `tokens`.
- Drop noisy transport-level or duplicate fields rather than mirroring the full raw call surface.
- Add small provider-local helpers in `tracing.py` to extract `input`, `output`, `metadata`, and `metrics` before opening or closing spans.

Aim for spans that are readable in the UI without requiring someone to reverse-engineer the provider SDK's calling convention from positional arguments.

Shape spans by semantics, not by the provider SDK object model:

- `input` is the meaningful request a human would describe, not the raw Python call signature.
- `output` is the meaningful result, not a provider response class dumped wholesale.
- `metadata` is for supporting context that helps interpretation but is not the main payload.
- `metrics` is for timings, token counts, and similar numeric accounting.

Good span shaping usually means:

- flattening positional arguments into named fields
- omitting duplicate values that appear in both request and response objects
- normalizing provider-specific classes into dicts/lists/scalars
- aggregating streaming chunks into one final `output` plus stream-specific `metrics`
- preserving useful provider identifiers without leaking transport noise

Use provider-local helper functions instead of building spans inline inside wrappers. A good pattern is:

```python
def _prepare_traced_call(args: list[Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ...


def _process_result(result: Any, start: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ...
```

Keep wrapper bodies thin: prepare input, open the span, call the provider, normalize the result, then log `output`, `metadata`, and `metrics`.

When deciding where a field belongs:

- Put it in `input` if the caller intentionally supplied it.
- Put it in `output` if it is the core result a user would care about.
- Put it in `metadata` if it explains the result but is not the result itself.
- Put it in `metrics` if it is numeric operational accounting or timing.
- Put it in `error` if the call failed and you want the span to record the exception or failure message instead of pretending the failure is ordinary output.

Distinguish span payload fields from span setup fields:

- Treat `input`, `output`, `metadata`, `metrics`, and `error` as the main logged payload fields.
- Treat `name` plus `type` or `span_attributes` as span identity/classification, not as payload.
- Use `parent` only when you need to attach the span to an explicit exported parent instead of relying on current-span context.
- Use `start_time` when the true start happened before the wrapper got control and you need accurate duration or time-to-first-token accounting.

Examples:

- prompt/model/tools/config belong in `input`
- generated text, tool calls, embeddings summary, generated images summary, or normalized message content belong in `output`
- provider request ids, finish reasons, safety annotations, cached-hit indicators, or model revision identifiers belong in `metadata`
- token counts, elapsed time, time-to-first-token, retry counts, or billable character counts belong in `metrics`
- exceptions, provider errors, and wrapper failures belong in `error`

Across the current integrations, `input`/`output`/`metadata`/`metrics` are the common structured logging fields, and `error` is the main additional event field used during failures. Other values should usually live inside one of those containers unless they are truly span-level controls like `name`, `type`, `span_attributes`, `parent`, or `start_time`.

Treat provider-owned binary payloads as attachments, not raw logged bytes. When traced input or output contains inline media, generated files, or other uploadable content:

- Convert raw `bytes` into `braintrust.logger.Attachment` objects in provider-local tracing helpers instead of logging raw bytes or large base64 blobs.
- Use the repo's existing message/content shapes when embedding attachments in traced payloads. For multimodal content this is often `{"image_url": {"url": attachment}}`, even when the MIME type is not literally an image.
- Preserve ordinary remote URLs as strings. Only convert provider-owned binary content or data-URL style payloads that Braintrust should upload to object storage.
- Keep structured metadata alongside the attachment, such as MIME type, size, safety attributes, or provider ids, so spans stay inspectable without reading the blob.

Prefer feature detection first and version checks second. Use:

- `detect_module_version(...)`
- `version_satisfies(...)`
- `make_specifier(...)`

Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids; do not silently paper over duplicates.

If a provider surface has both sync and async variants, try to keep the traced schema aligned across both paths. Differences in implementation are fine; differences in logged shape should be intentional.

## Patcher Rules

Create one patcher per coherent patch target. Split unrelated targets into separate patchers.

Use `FunctionWrapperPatcher` for one import path or one constructor/method surface, for example:

- `ProviderClient.__init__`
- `client.responses.create`

Use `CompositeFunctionWrapperPatcher` when several closely related targets should appear as one patcher, for example:

- sync and async variants of the same method
- the same logical surface patched across multiple modules

Set `target_module` when the patch target lives outside the module named by `import_names`, especially for optional or deep submodules. Failed `target_module` imports should make the patcher skip cleanly through `applies()`.

Use `superseded_by` for version-conditional mutual exclusion. Express fallback relationships declaratively instead of reproducing `hasattr` logic in custom `applies()` methods whenever possible.

Expose manual wrapping helpers through `wrap_target()`:

```python
def wrap_agent(Agent: Any) -> Any:
    return AgentPatcher.wrap_target(Agent)
```

Use lower `priority` values only when ordering matters, such as context propagation before tracing patchers.

Require every patcher to provide:

- a stable `name`
- version gating only when needed
- clean existence checks
- idempotence through the base patcher marker

## Testing

Keep integration tests in the provider package.

Use `@pytest.mark.vcr` for real provider network behavior. Prefer recorded provider traffic over mocks or fakes. Use mocks or fakes only for cases that are hard to drive through recordings, such as:

- narrow error injection
- local version-routing logic
- patcher existence checks

Write tests against the emitted span shape, not just the provider return value. A tracing change is incomplete if the provider call still works but the logged span becomes noisy, incomplete, or inconsistent.

Cover the surfaces that changed:

- direct `wrap_*()` behavior
- `setup()` patching for newly created clients or classes
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and error logging
- patcher resolution and duplicate detection
- attachment conversion for binary inputs or generated media, including assertions that traced payloads contain `Attachment` objects rather than raw bytes
- span structure, including assertions that the traced span exposes meaningful `input`, `output`, `metadata`, and `metrics` rather than opaque raw call arguments

For span assertions, prefer checking the specific normalized fields that matter:

- the `input` contains the expected model/messages/prompt/config fields
- the `output` contains normalized provider results rather than opaque SDK instances
- the `metadata` contains finish reasons, ids, or annotations in the expected place
- the `metrics` contain the expected timing or token fields when the provider returns them
- binary payloads are represented as `Attachment` objects where applicable

If a change affects streaming, verify both:

- intermediate behavior still returns the provider's expected iterator or async iterator
- the final logged span contains the aggregated `output` and stream-specific `metrics`

Keep VCR cassettes in `py/src/braintrust/integrations/<provider>/cassettes/`. Re-record only when the behavior change is intentional.

When the provider returns binary HTTP responses or generated media, make cassette sanitization part of the change if needed so recorded fixtures do not store raw file bytes.

When choosing commands, confirm the real session name in `py/noxfile.py` instead of assuming it matches the provider folder. Examples in this repo include `test_agno`, `test_anthropic`, and `test_google_adk`.

## Commands

```bash
cd py && nox -s "test_<session>(latest)"
cd py && nox -s "test_<session>(latest)" -- -k "test_name"
cd py && nox -s "test_<session>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core
cd py && make lint
```

## Validation

- Run the narrowest provider session first.
- Run the relevant auto-instrument subprocess test if `auto.py` changed.
- Run `cd py && make test-core` if shared integration code changed.
- Run `cd py && make lint` before handoff when shared files or repo-level wiring changed.

## Pitfalls

- Treating a wrapper migration as fresh integration work.
- Changing shared integration primitives when the provider-specific package should own the behavior.
- Combining unrelated targets into one patcher.
- Forgetting repo-level touch points for new providers: `integrations/__init__.py`, `py/noxfile.py`, and sometimes `auto.py`.
- Forgetting async or streaming coverage.
- Re-recording cassettes when behavior did not intentionally change.
- Adding a custom `_instrument_*` helper where `_instrument_integration()` already fits.
- Forgetting `target_module` for deep or optional submodule patch targets.
