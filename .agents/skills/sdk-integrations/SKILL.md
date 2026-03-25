---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes. Do not use when migrating an existing legacy wrapper from `py/src/braintrust/wrappers/` into the integrations API; use `sdk-wrapper-migrations` for that.
---

# SDK Integrations

Use this skill for integrations API work under `py/src/braintrust/integrations/`.

If the provider already has a real implementation under `py/src/braintrust/wrappers/<provider>/` and the task is to move that implementation into the integrations API, switch to `sdk-wrapper-migrations` instead of treating it like a fresh integration.

## Pick The Nearest Example

Start from one structural reference and one patching reference instead of designing from scratch:

- ADK (`py/src/braintrust/integrations/adk/`) for direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, manual `wrap_*()` helpers, and priority-based context propagation.
- Agno (`py/src/braintrust/integrations/agno/`) for multi-target patching, version-conditional fallbacks with `superseded_by`, and providers that need several related patchers.
- Anthropic (`py/src/braintrust/integrations/anthropic/`) for constructor patching and a compact provider package with a small public surface.

Match an existing repo pattern unless the target provider forces a different shape.

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

Prefer feature detection first and version checks second. Use:

- `detect_module_version(...)`
- `version_satisfies(...)`
- `make_specifier(...)`

Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids; do not silently paper over duplicates.

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

Cover the surfaces that changed:

- direct `wrap_*()` behavior
- `setup()` patching for newly created clients or classes
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and error logging
- patcher resolution and duplicate detection

Keep VCR cassettes in `py/src/braintrust/integrations/<provider>/cassettes/`. Re-record only when the behavior change is intentional.

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
