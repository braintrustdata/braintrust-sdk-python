---
name: sdk-wrapper-migrations
description: Migrate Braintrust Python SDK legacy wrapper implementations to the integrations API. Use when moving a provider from `py/src/braintrust/wrappers/` into `py/src/braintrust/integrations/`, preserving backward compatibility while relocating tracing, patchers, tests, cassettes, auto-instrument hooks, and test sessions.
---

# SDK Wrapper Migrations

Use this skill when a provider already exists under `py/src/braintrust/wrappers/` and needs to be migrated to the integrations API.

Use current repo examples, not old commit history:

- `py/src/braintrust/integrations/adk/` for full integration package structure, test placement, auto-instrument coverage, and wrapper delegation
- `py/src/braintrust/integrations/anthropic/` for constructor patching and a minimal compatibility wrapper

The target end state is:

- provider logic lives in `py/src/braintrust/integrations/<provider>/`
- tests and cassettes live with the integration
- `auto_instrument()` uses the integration
- the legacy wrapper becomes a thin compatibility layer

## Read First

Always read:

- the existing legacy wrapper under `py/src/braintrust/wrappers/<provider>/`
- `py/src/braintrust/integrations/anthropic/__init__.py`
- `py/src/braintrust/integrations/anthropic/integration.py`
- `py/src/braintrust/integrations/anthropic/patchers.py`
- `py/src/braintrust/integrations/anthropic/tracing.py`
- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/auto.py`
- `py/noxfile.py`

Read when relevant:

- `py/src/braintrust/integrations/auto_test_scripts/`
- the provider's existing wrapper tests and cassettes

## Workflow

1. Inventory the wrapper's public API, patch targets, tests, and cassettes.
2. Create an integration package that preserves the wrapper's behavior and public helper surface.
3. Move provider-specific tracing and patching into the integration package.
4. Move tests, cassettes, and auto-instrument subprocess coverage next to the integration.
5. Wire the integration into exports, `auto.py`, and `py/noxfile.py`.
6. Replace the wrapper with a thin re-export layer.
7. Run the narrowest provider session first, then expand if shared code changed.

## Migration Checklist

### 1. Preserve the public surface

Before moving code, list the public names exposed by the wrapper:

- setup functions
- `wrap_*()` helpers
- deprecated aliases that still need to work
- `__all__`

The integration package should own that public surface after the migration. The wrapper should only delegate to it.

### 2. Create the integration package

Create `py/src/braintrust/integrations/<provider>/` with the same split used by ADK:

- `__init__.py`: public API, setup entry point, deprecated aliases if needed
- `integration.py`: `BaseIntegration` subclass and patcher registration
- `patchers.py`: one patcher per coherent patch target, plus public `wrap_*()` helpers
- `tracing.py`: provider-specific tracing, stream handling, normalization, and helper code
- `test_<provider>.py`: provider behavior tests
- `cassettes/`: VCR recordings when the provider uses HTTP

Keep provider-specific behavior out of shared modules unless the provider truly needs a shared change.

### 3. Move tracing and patching out of the wrapper

Extract wrapper internals into:

- `tracing.py` for spans, metadata extraction, stream aggregation, and output normalization
- `patchers.py` for patcher classes and `wrap_*()` helpers
- `integration.py` for the orchestration layer only

Prefer one patcher per coherent patch target. Use composite patchers only when several related targets should be user-visible as one patcher.

### 4. Preserve setup behavior

The new integration package should preserve the wrapper's setup semantics:

- keep the same setup function names where possible
- keep deprecated aliases that users may still import
- keep logger initialization or other setup-time side effects aligned with prior behavior

The integration package is the new source of truth. Do not leave setup logic duplicated in the wrapper.

### 5. Move tests and cassettes

Move provider tests from `py/src/braintrust/wrappers/` into the integration package.

Move or rename:

- provider behavior tests to `py/src/braintrust/integrations/<provider>/`
- cassettes to `py/src/braintrust/integrations/<provider>/cassettes/`
- auto-instrument subprocess tests to `py/src/braintrust/integrations/auto_test_scripts/`

Update imports and cassette paths during the move. Preserve coverage for:

- direct `wrap_*()` behavior
- setup-time patching
- sync paths
- async paths
- streaming paths
- idempotence
- failure and logging behavior

### 6. Wire repo-level integration points

Update the minimum shared surfaces required by the migration:

- `py/src/braintrust/integrations/__init__.py`
- `py/src/braintrust/auto.py` if the provider participates in `auto_instrument()`
- `py/noxfile.py` so provider sessions run against the integration tests

Only change shared integration primitives when the provider actually needs it.

### 7. Reduce the wrapper to compatibility imports

After the integration package is working, replace the legacy wrapper implementation with a thin `__init__.py` that re-exports the migrated surface from `braintrust.integrations.<provider>`.

Keep `__all__` aligned with the pre-migration public API. Do not leave business logic, tracing helpers, or patchers behind in the wrapper package.

## Current Examples

Use ADK as the main structural reference:

- tracing moved into `py/src/braintrust/integrations/adk/tracing.py`
- patchers moved into `py/src/braintrust/integrations/adk/patchers.py`
- orchestration moved into `py/src/braintrust/integrations/adk/integration.py`
- public exports live in `py/src/braintrust/integrations/adk/__init__.py`
- wrapper tests and cassettes moved under `py/src/braintrust/integrations/adk/`
- auto-instrument subprocess coverage moved to `py/src/braintrust/integrations/auto_test_scripts/test_auto_adk.py`
- `py/src/braintrust/wrappers/adk/__init__.py` became a thin compatibility layer

Use Anthropic as the compact constructor-patching reference:

- `py/src/braintrust/integrations/anthropic/integration.py` registers sync and async constructor patchers
- `py/src/braintrust/integrations/anthropic/patchers.py` keeps one patcher per constructor target
- `py/src/braintrust/wrappers/anthropic.py` is a minimal compatibility re-export

Match those patterns unless the provider has a clear reason to differ.

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
- Run `cd py && make lint` before handoff when the migration touches shared files.
- Run the relevant auto-instrument subprocess tests if `auto.py` changed.
- Verify the old wrapper import path still works through compatibility re-exports.

## Pitfalls

- Copying wrapper code into the integration package without restructuring it around `integration.py`, `patchers.py`, and `tracing.py`.
- Leaving real logic behind in the wrapper after the migration.
- Breaking deprecated aliases or `__all__` exports that users still import.
- Moving tests without moving their cassettes or auto-instrument scripts.
- Forgetting to update `py/noxfile.py` to point at the new integration test paths.
- Changing shared integration code more broadly than the provider requires.
- Re-recording cassettes when behavior did not intentionally change.
