---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes. Do not use when migrating an existing legacy wrapper from `py/src/braintrust/wrappers/` into the integrations API; use `sdk-wrapper-migrations` for that.
---

# SDK Integrations

Use this skill for integration work under `py/src/braintrust/integrations/`.

Use `sdk-wrapper-migrations` instead when the provider already has a real implementation under `py/src/braintrust/wrappers/<provider>/` and the task is to move that implementation into the integrations API.

## Quick Start

Before editing:

1. Read the shared integration primitives.
2. Read the target provider package.
3. Pick the nearest existing integration as a reference.
4. Decide the span shape before writing patchers.
5. Run the narrowest provider nox session first.

Do not design a new integration shape from scratch if an existing provider already matches the problem.

## Read First

Always read:

- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/versioning.py`
- `py/src/braintrust/integrations/__init__.py`
- `py/src/braintrust/integrations/utils.py`
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
- `py/src/braintrust/integrations/auto_test_scripts/` for subprocess auto-instrument coverage
- `py/src/braintrust/integrations/test_utils.py` when touching shared attachment materialization or multimodal payload shaping
- `py/src/braintrust/integrations/adk/test_adk.py`, `py/src/braintrust/integrations/anthropic/test_anthropic.py`, and `py/src/braintrust/integrations/google_genai/test_google_genai.py` for attachment-focused test layout patterns
- `py/src/braintrust/integrations/adk/tracing.py`, `py/src/braintrust/integrations/anthropic/tracing.py`, and `py/src/braintrust/integrations/google_genai/tracing.py` when handling multimodal content, binary inputs, generated media, or attachment materialization behavior

Do not forget `auto.py` and `auto_test_scripts/`. Import-order and subprocess regressions often only show up there.

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

1. Read shared primitives and the provider package.
2. Decide which public surface is being patched.
3. Define the span shape:
   - `input`
   - `output`
   - `metadata`
   - `metrics`
   - `error` when failures matter
4. Implement or update patchers.
5. Implement or update tracing helpers.
6. Add or update focused tests.
7. Run the narrowest nox session first, then expand only if shared code changed.

Do not start by wiring wrappers and only later decide what the span should contain.

## Route The Task

### New provider integration

1. Create `py/src/braintrust/integrations/<provider>/`.
2. Use this layout unless the provider is exceptionally small:
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
4. Do not touch `auto.py`, `integrations/__init__.py`, or `py/noxfile.py` unless the task requires it.
5. Even if `auto.py` does not change, check whether the behavior change also needs an auto-instrument subprocess test update.
6. Preserve existing span shape conventions unless the task is intentionally correcting them.

### `auto_instrument()` only

1. Update `py/src/braintrust/auto.py`.
2. Prefer `_instrument_integration(...)` over a custom `_instrument_*` helper when the standard pattern fits.
3. Add the integration import near the other integration imports.
4. Add or update the relevant subprocess auto-instrument test.

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

If logic is genuinely shared across integrations, move it to `py/src/braintrust/integrations/utils.py` instead of copying it into multiple providers.

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

Preserve provider behavior. Tracing code must not change return values, control flow, or error behavior unless the task explicitly requires it.

Keep sync and async traced schemas aligned when the provider exposes both.

## Span Design Rules

Build readable spans. Do not dump raw `args` and `kwargs` unless the provider API already exposes a clean schema.

Use this rubric:

- `input`: the meaningful user request
- `output`: the meaningful provider result
- `metadata`: supporting context such as ids, finish reasons, safety data, model revisions
- `metrics`: timing and numeric accounting such as token counts or elapsed time
- `error`: exceptions or failure information

Good span shaping usually means:

- flatten positional arguments into named fields
- normalize provider SDK objects into dicts, lists, or scalars
- drop duplicate or noisy transport fields
- aggregate streaming chunks into one final `output` plus stream-specific `metrics`

Keep wrapper bodies thin: prepare traced input, open the span, call the provider, normalize the result, and log `output`/`metadata`/`metrics`.

Prefer provider-local helpers in `tracing.py`, for example:

```python
def _prepare_traced_call(args: list[Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ...


def _process_result(result: Any, start: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ...
```

Treat binary payloads as attachments, not logged bytes:

- prefer the shared `_materialize_attachment(...)` helper in `py/src/braintrust/integrations/utils.py` over provider-local base64 or file-decoding code
- convert provider-owned raw `bytes`, base64 payloads, data URLs, file inputs, and generated media into `braintrust.logger.Attachment` objects when Braintrust should upload the content
- preserve normal remote URLs as strings
- use the repo's existing multimodal payload shapes after materialization:
  - images -> `{"image_url": {"url": attachment}}`
  - non-image media/documents/files -> `{"file": {"file_data": attachment, "filename": resolved.filename}}`
- do not force non-image payloads through `image_url` shims
- if attachment materialization fails, keep the original value instead of dropping it or replacing it with `None`
- preserve non-attachment values while walking nested payloads unless you are intentionally normalizing them for readability
- keep useful metadata such as MIME type, size, safety data, filenames, or provider ids next to the attachment

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

## Testing Rules

Keep tests in the provider package.

Default bug-fix workflow: red -> green.

- First add or update a focused test that reproduces the integration bug.
- Then implement the fix.
- Only skip this when the task explicitly asks for a different approach.

Prefer VCR-backed real provider coverage with `@pytest.mark.vcr`. Use mocks or fakes only for cases that are hard to drive through recordings, such as:

- narrow error injection
- local version-routing logic
- patcher existence checks

Test emitted spans, not just provider return values.

Cover the surfaces that changed:

- direct `wrap_*()` behavior
- setup-time patching
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and error logging
- patcher resolution and duplicate detection when relevant
- attachment conversion for binary inputs or generated media, including assertions that images land under `image_url.url`, non-image payloads land under `file.file_data`, and traced payloads contain `Attachment` objects rather than raw bytes or base64 blobs
- span structure, especially `input`, `output`, `metadata`, and `metrics`

For streaming changes, verify both:

- the provider still returns the expected iterator or async iterator
- the final logged span contains the aggregated `output` and stream-specific `metrics`

Also verify, when relevant:

- the `input` contains the expected model/messages/prompt/config fields
- the `output` contains normalized provider results rather than opaque SDK instances
- the `metadata` contains finish reasons, ids, or annotations in the expected place
- binary payloads are represented as `Attachment` objects where applicable, while remote URLs and non-attachment values remain unchanged and unmaterialized file inputs are preserved rather than dropped

Keep VCR cassettes in `py/src/braintrust/integrations/<provider>/cassettes/`. Re-record only when behavior intentionally changes.

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
- forcing non-image attachments through `image_url` shims, dropping unrecognized file inputs, or re-serializing non-attachment values while materializing payloads
