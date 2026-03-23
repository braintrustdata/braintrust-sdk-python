---
name: sdk-integrations
description: Create or update a Braintrust Python SDK integration using the integrations API. Use when asked to add an integration, update an existing integration, add or update patchers, update auto_instrument, add integration tests, or work in py/src/braintrust/integrations/.
---

# SDK Integrations

SDK integrations define how Braintrust discovers a provider, patches it safely, and keeps provider-specific tracing local to that integration. Read the existing integration closest to your task before writing a new one. If there is no closer example, `py/src/braintrust/integrations/anthropic/` is a useful reference implementation.

## Workflow

1. Read the shared integration primitives and the closest provider example.
2. Choose the task shape: new provider, existing provider update, or `auto_instrument()` update.
3. Implement the smallest integration, patcher, tracing, and export changes needed.
4. Add or update VCR-backed integration tests and only re-record cassettes when behavior changed intentionally.
5. Run the narrowest provider session first, then expand to shared validation only if the change touched shared code.

## Commands

```bash
cd py && nox -s "test_<provider>(latest)"
cd py && nox -s "test_<provider>(latest)" -- -k "test_name"
cd py && nox -s "test_<provider>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core
cd py && make lint
```

## Creating or Updating an Integration

### 1. Read the nearest existing implementation

Always inspect these first:

- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/runtime.py`
- `py/src/braintrust/integrations/versioning.py`
- `py/src/braintrust/integrations/config.py`

Relevant example implementation:

- `py/src/braintrust/integrations/anthropic/`

Read these additional files only when the task needs them:

- changing `auto_instrument()`: `py/src/braintrust/auto.py` and `py/src/braintrust/auto_test_scripts/test_auto_anthropic_patch_config.py`
- adding or updating VCR tests: `py/src/braintrust/conftest.py` and `py/src/braintrust/integrations/anthropic/test_anthropic.py`

Then choose the path that matches the task:

- new provider: create `py/src/braintrust/integrations/<provider>/`
- existing provider: read the provider package first and change only the affected patchers, tracing, tests, or exports
- `auto_instrument()` only: keep the integration package unchanged unless the option shape or patcher surface also changed

### 2. Create or extend the integration module

For a new provider, create a package under `py/src/braintrust/integrations/<provider>/`.

For an existing provider, keep the module layout unless the current structure is actively causing problems.

Typical files:

- `__init__.py`: public exports for the integration type and any public helpers
- `integration.py`: the `BaseIntegration` subclass, patcher registration, and high-level orchestration
- `patchers.py`: one patcher per patch target, with version gating and existence checks close to the patch
- `tracing.py`: provider-specific span creation, metadata extraction, stream handling, and output normalization
- `test_<provider>.py`: integration tests for `wrap(...)`, `setup()`, sync/async behavior, streaming, and error handling
- `cassettes/`: recorded provider traffic for VCR-backed integration tests when the provider uses HTTP

### 3. Define the integration class

Implement a `BaseIntegration` subclass in `integration.py`.

Set:

- `name`
- `import_names`
- `min_version` and `max_version` only when needed
- `patchers`

Keep the class focused on orchestration. Provider-specific tracing logic should stay in `tracing.py`.

### 4. Add one patcher per coherent patch target

Put patchers in `patchers.py`.

Use `FunctionWrapperPatcher` when patching a single import path with `wrapt.wrap_function_wrapper`. Good examples:

- constructor patchers like `ProviderClient.__init__`
- single API surfaces like `client.responses.create`
- one sync and one async constructor patcher instead of one patcher doing both

Keep patchers narrow. If you need to patch multiple unrelated targets, create multiple patchers rather than one large patcher.

Patchers are responsible for:

- stable patcher ids via `name`
- optional version gating
- existence checks
- idempotence through the base patcher marker

### 5. Keep tracing provider-local

Put span creation, metadata extraction, stream aggregation, error logging, and output normalization in `tracing.py`.

This layer should:

- preserve provider behavior
- support sync, async, and streaming paths as needed
- avoid raising from tracing-only code when that would break the provider call

If the provider has complex streaming internals, keep that logic local instead of forcing it into shared abstractions.

### 6. Wire public exports

Update public exports only as needed:

- `py/src/braintrust/integrations/__init__.py`
- `py/src/braintrust/__init__.py`

### 7. Update auto_instrument only if this integration should be auto-patched

If the provider belongs in `braintrust.auto.auto_instrument()`, add a branch in `py/src/braintrust/auto.py`.

Match the current pattern:

- plain `bool` options for simple on/off integrations
- `IntegrationPatchConfig` only when users need patcher-level selection

## Tests

Keep integration tests with the integration package.

Provider behavior tests should use `@pytest.mark.vcr` whenever the provider uses network calls. Avoid mocks and fakes.

Cover:

- direct `wrap(...)` behavior
- `setup()` patching new clients
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure/error logging
- patcher selection if using `IntegrationPatchConfig`

Preferred locations:

- provider behavior tests: `py/src/braintrust/integrations/<provider>/test_<provider>.py`
- version helper tests: `py/src/braintrust/integrations/test_versioning.py`
- auto-instrument subprocess tests: `py/src/braintrust/auto_test_scripts/`

If the provider uses VCR, keep cassettes next to the integration test file under `py/src/braintrust/integrations/<provider>/cassettes/`.

Only re-record cassettes when the behavior change is intentional.

Use mocks or fakes only for cases that are hard to drive through recorded provider traffic, such as narrowly scoped error injection, local version-routing logic, or patcher existence checks.

## Patterns

### Constructor patching

If instrumenting future clients created by the SDK is the goal, patch constructors and attach traced surfaces after the real constructor runs. Anthropic is an example of this pattern.

### Patcher selection

Use `IntegrationPatchConfig` only when users benefit from enabling or disabling specific patchers. Validate unknown patcher ids through `BaseIntegration.resolve_patchers()` instead of silently ignoring them.

### Versioning

Prefer feature detection first and version checks second.

Use:

- `detect_module_version(...)`
- `version_in_range(...)`
- `version_matches_spec(...)`

Do not add `packaging` just for integration routing.

## Validation

- Run the narrowest provider session first.
- Run `cd py && make test-core` if you changed shared integration code.
- Run `cd py && make lint` before handing off broader integration changes.
- If you changed `auto_instrument()`, run the relevant subprocess auto-instrument tests.

## Done When

- the provider package contains only the integration, patcher, tracing, export, and test changes required by the task
- provider behavior tests use VCR unless recorded traffic cannot cover the behavior
- cassette changes are present only when provider behavior changed intentionally
- the narrowest affected provider session passes
- `cd py && make test-core` has been run if shared integration code changed
- `cd py && make lint` has been run before handoff

## Common Pitfalls

- Leaving provider behavior in `BaseIntegration` instead of the provider package.
- Combining multiple unrelated patch targets into one patcher.
- Forgetting async or streaming coverage.
- Defaulting to mocks or fakes when the provider flow can be covered with VCR.
- Moving tests but not moving their cassettes.
- Adding patcher selection without tests for enabled and disabled cases.
- Editing `auto_instrument()` in a way that implies a registry exists when it does not.
