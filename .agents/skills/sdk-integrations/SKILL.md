---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API. Use for work in `py/src/braintrust/integrations/`, including new providers, patchers, tracing, `auto_instrument()` updates, integration exports, and integration tests.
---

# SDK Integrations

Use this skill for integrations API work under `py/src/braintrust/integrations/`.

Start from the nearest existing provider instead of designing from scratch:

- ADK (`py/src/braintrust/integrations/adk/`) is the best reference for direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, and public `wrap_*()` helpers.
- Anthropic (`py/src/braintrust/integrations/anthropic/`) is the best reference for constructor patching with `FunctionWrapperPatcher`.

## Workflow

1. Read the shared primitives and the nearest provider example.
2. Decide whether the task is a new provider, an existing provider update, or an `auto_instrument()` change.
3. Change only the affected integration, patchers, tracing, exports, and tests.
4. Update tests and cassettes only where behavior changed intentionally.
5. Run the narrowest provider session first, then expand only if shared code changed.

## Read First

Always read:

- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/versioning.py`

Read when relevant:

- `py/src/braintrust/auto.py` for `auto_instrument()` work
- `py/src/braintrust/conftest.py` for VCR behavior
- `py/src/braintrust/integrations/adk/test_adk.py` for integration test patterns
- `py/src/braintrust/integrations/auto_test_scripts/` for subprocess auto-instrument tests

## Package Layout

Create new providers under `py/src/braintrust/integrations/<provider>/`. Keep the existing layout for provider updates unless the current structure is the problem.

Typical files:

- `__init__.py`: export the integration class, `setup_<provider>()`, and public `wrap_*()` helpers
- `integration.py`: define the `BaseIntegration` subclass and register patchers
- `patchers.py`: define patchers and `wrap_*()` helpers
- `tracing.py`: keep provider-specific tracing, stream handling, and normalization
- `test_<provider>.py`: keep provider behavior tests next to the integration
- `cassettes/`: keep VCR recordings next to the integration tests when the provider uses HTTP

## Integration Rules

Keep `integration.py` thin. Set:

- `name`
- `import_names`
- `patchers`
- `min_version` and `max_version` only when needed

Keep provider behavior in the provider package, not in shared integration code. Put span creation, metadata extraction, stream aggregation, error logging, and output normalization in `tracing.py`.

Preserve provider behavior. Do not let tracing-only code break the provider call.

## Patcher Rules

Create one patcher per coherent patch target. If targets are unrelated, split them.

Use `FunctionWrapperPatcher` for one import path or one constructor/method surface, for example:

- `ProviderClient.__init__`
- `client.responses.create`

Use `CompositeFunctionWrapperPatcher` when several closely related targets should appear as one patcher, for example:

- sync and async variants of the same method
- the same function patched across multiple modules

Set `target_module` when the patch target lives outside the module named by `import_names`, especially for optional or deep submodules. Failed `target_module` imports should cause the patcher to skip cleanly through `applies()`.

Expose manual wrapping helpers through `wrap_target()`:

```python
def wrap_agent(Agent: Any) -> Any:
    return AgentRunAsyncPatcher.wrap_target(Agent)
```

Use lower `priority` values only when ordering matters, such as context propagation before tracing.

Patchers must provide:

- stable `name` values
- version gating only when needed
- existence checks
- idempotence through the base patcher marker

Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids instead of silently ignoring them.

## Patching Patterns

Use constructor patching when the goal is to instrument future clients created by the provider SDK. Patch the constructor, then attach traced surfaces after the real constructor runs.

Use direct method patching with `target_module` when the provider exposes a flatter API and there is no useful constructor patch point.

Keep public `wrap_*()` helpers in `patchers.py` and export them from the integration package.

## Versioning

Prefer feature detection first and version checks second.

Use:

- `detect_module_version(...)`
- `version_satisfies(...)`
- `make_specifier(...)`

## `auto_instrument()`

Update `py/src/braintrust/auto.py` only if the integration should be auto-patched.

All `auto_instrument()` parameters are plain `bool` flags. Use `_instrument_integration(...)` instead of adding a custom `_instrument_*` function:

```python
if provider:
    results["provider"] = _instrument_integration(ProviderIntegration)
```

Add the integration import near the other integration imports in `auto.py`.

## Tests

Keep integration tests in the provider package.

Use `@pytest.mark.vcr` for real provider network behavior. Prefer recorded provider traffic over mocks or fakes. Use mocks or fakes only for cases that are hard to drive through recordings, such as:

- narrow error injection
- local version-routing logic
- patcher existence checks

Cover the surfaces that changed:

- direct `wrap(...)` behavior
- `setup()` patching new clients
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and error logging
- patcher resolution and duplicate detection

Keep VCR cassettes in `py/src/braintrust/integrations/<provider>/cassettes/`. Re-record them only for intentional behavior changes.

## Commands

```bash
cd py && nox -s "test_<provider>(latest)"
cd py && nox -s "test_<provider>(latest)" -- -k "test_name"
cd py && nox -s "test_<provider>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core
cd py && make lint
```

## Validation

- Run the narrowest provider session first.
- Run `cd py && make test-core` if shared integration code changed.
- Run `cd py && make lint` before handing off broader integration changes.
- Run the relevant auto-instrument subprocess tests if `auto_instrument()` changed.

## Pitfalls

- Moving provider-specific behavior into shared integration code.
- Combining unrelated targets into one patcher.
- Forgetting async or streaming coverage.
- Re-recording cassettes when behavior did not intentionally change.
- Adding a custom `_instrument_*` helper where `_instrument_integration()` already fits.
- Forgetting `target_module` for deep or optional submodule patch targets.
