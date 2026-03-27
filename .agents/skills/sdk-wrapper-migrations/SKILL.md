---
name: sdk-wrapper-migrations
description: Migrate Braintrust Python SDK legacy wrapper implementations to the integrations API. Use when moving an existing provider from `py/src/braintrust/wrappers/` into `py/src/braintrust/integrations/` while preserving old import paths, public helpers, tests, cassettes, tracing behavior, auto-instrument hooks, and nox coverage.
---

# SDK Wrapper Migrations

Migrate an existing wrapper-backed provider to the integrations API without breaking the old wrapper import path.

Prefer this skill only when both of these are true:

- Find an existing provider implementation under `py/src/braintrust/wrappers/`.
- Need the end state to be an integration package under `py/src/braintrust/integrations/` plus a thin compatibility wrapper.

Use `sdk-integrations` instead when the task is integration work that does not start from a legacy wrapper.

Do not reconstruct migrations from old commit history. Start from the current tree and copy the nearest current pattern.

## Target End State

Finish with this structure:

- provider logic in `py/src/braintrust/integrations/<provider>/`
- provider tests in `py/src/braintrust/integrations/<provider>/`
- provider cassettes in `py/src/braintrust/integrations/<provider>/cassettes/` when applicable
- `auto_instrument()` pointing at the integration when the provider participates in auto patching
- the wrapper reduced to compatibility re-exports with the old public surface intact

Do not leave tracing helpers, patchers, or setup orchestration behind in the wrapper.

## Current References

Use the nearest current provider instead of inventing a layout:

- ADK: use `py/src/braintrust/integrations/adk/` as the main structural reference for package layout, patchers, tracing split, tests, cassettes, auto-test scripts, and thin wrapper delegation.
- Agno: use `py/src/braintrust/integrations/agno/` for multi-method patching, `CompositeFunctionWrapperPatcher`, raw wrapt wrappers in `tracing.py`, and version-conditional fallbacks using `superseded_by`.
- Anthropic: use `py/src/braintrust/integrations/anthropic/` for compact constructor patching and a minimal compatibility wrapper.

Match one of those patterns unless the provider has a concrete reason to differ.

## Read First

Always read:

- the existing legacy wrapper under `py/src/braintrust/wrappers/<provider>/` or `py/src/braintrust/wrappers/<provider>.py`
- `py/src/braintrust/integrations/base.py`
- `py/src/braintrust/integrations/versioning.py`
- `py/src/braintrust/auto.py`
- `py/noxfile.py`

Read these current migration examples before editing:

- `py/src/braintrust/integrations/adk/__init__.py`
- `py/src/braintrust/integrations/adk/integration.py`
- `py/src/braintrust/integrations/adk/patchers.py`
- `py/src/braintrust/integrations/adk/tracing.py`
- `py/src/braintrust/integrations/agno/patchers.py`
- `py/src/braintrust/integrations/agno/tracing.py`
- `py/src/braintrust/integrations/anthropic/__init__.py`
- `py/src/braintrust/integrations/anthropic/integration.py`
- `py/src/braintrust/integrations/anthropic/patchers.py`

Read when relevant:

- `py/src/braintrust/conftest.py` for VCR behavior
- `py/src/braintrust/integrations/auto_test_scripts/` for auto-instrument subprocess coverage
- the provider's existing wrapper tests and cassettes

## Migration Playbook

1. Inventory the wrapper before moving code.
2. Create the integration package with the public API and layout you intend to keep.
3. Move tracing helpers and wrapper logic into provider-local integration modules.
4. Move tests, cassettes, and auto-instrument subprocess coverage next to the integration.
5. Wire exports, `auto.py`, and `py/noxfile.py` to the new integration location.
6. Collapse the wrapper to compatibility imports and re-exports.
7. Run the narrowest provider test session first, then expand only if shared code changed.

## Inventory First

Before editing, list the wrapper's user-visible and repo-visible surface:

- setup entry points such as `setup_<provider>()`
- public `wrap_*()` helpers
- deprecated aliases that must still import correctly
- `__all__`
- patch targets and target modules
- sync, async, and streaming code paths
- test files, cassette directories, and auto-test scripts
- any version-routing logic, especially `hasattr`-based fallback behavior

Do not start moving files until this inventory is explicit. The migration succeeds only if the integration preserves the same behavior and import surface.

## Package Layout

Create `py/src/braintrust/integrations/<provider>/` and keep provider-specific behavior there.

Use this layout unless the provider already has a better current variant:

- `__init__.py`: export the public API, `setup_<provider>()`, and compatibility aliases
- `integration.py`: define the `BaseIntegration` subclass and register patchers
- `patchers.py`: define patchers and public `wrap_*()` helpers
- `tracing.py`: keep spans, metadata extraction, stream handling, normalization, and helper code
- `test_<provider>.py` or split test files: keep provider behavior tests next to the integration
- `cassettes/`: keep VCR recordings next to the provider tests when the provider uses HTTP

Keep `integration.py` thin. Do not move provider behavior into shared integration primitives unless the provider truly needs a shared change.

## Public API Rules

Preserve the wrapper's public surface exactly unless the task explicitly changes it.

Keep or migrate:

- setup function names
- `wrap_*()` helper names
- deprecated aliases
- `__all__`

Make the integration package the source of truth. Make the wrapper import from the integration package, not the other way around.

When the legacy wrapper is a single module such as `py/src/braintrust/wrappers/anthropic.py`, reduce that module to compatibility re-exports in place. When the wrapper is a package directory, reduce its `__init__.py` to compatibility re-exports and delete or stop importing the old implementation modules if they are no longer used.

## Patching And Tracing Rules

Move raw tracing behavior into `tracing.py`.

Keep tracing wrappers as plain wrapt wrapper functions. Do not carry wrapper-era patch-state logic into tracing code:

- no `is_patched`
- no `mark_patched`
- no `hasattr` branching to choose targets

Move patch target selection into `patchers.py`.

Prefer:

- one `FunctionWrapperPatcher` per coherent target
- `CompositeFunctionWrapperPatcher` only when several related targets should appear as one patcher
- `superseded_by` for version-conditional fallback relationships

When the legacy wrapper does "wrap `_run` if present, otherwise wrap `run`", convert that to separate patchers instead of reproducing the branching:

- point the preferred patcher at the higher-priority target directly
- point the fallback patcher at the fallback target
- set `superseded_by` on the fallback patcher

Use `py/src/braintrust/integrations/agno/patchers.py` as the reference pattern for this conversion.

Expose manual wrapping through thin public helpers in `patchers.py`, then re-export them from `__init__.py`.

## Test And Cassette Moves

Move provider tests with the implementation. Do not strand coverage under `wrappers/`.

Move or update:

- provider behavior tests into `py/src/braintrust/integrations/<provider>/`
- cassette directories into `py/src/braintrust/integrations/<provider>/cassettes/`
- auto-instrument subprocess tests into `py/src/braintrust/integrations/auto_test_scripts/` when relevant

Update imports, cassette paths, and fixtures during the move.

Preserve coverage for the changed surfaces:

- direct `wrap_*()` behavior
- setup-time patching
- sync behavior
- async behavior
- streaming behavior
- idempotence
- failure and logging behavior
- version-routing behavior when applicable

Re-record cassettes only when behavior intentionally changes.

## Repo Wiring

Update only the shared surfaces required by the migration:

- `py/src/braintrust/integrations/__init__.py` when the provider should be exported there
- `py/src/braintrust/auto.py` when `auto_instrument()` should use the integration
- `py/noxfile.py` so provider sessions point at the migrated integration tests

Prefer narrow repo-level changes. Do not broaden shared integration code unless the migration cannot work without it.

## Validation

Run the smallest relevant session first:

```bash
cd py && nox -s "test_<provider>(latest)"
cd py && nox -s "test_<provider>(latest)" -- -k "test_name"
cd py && nox -s "test_<provider>(latest)" -- --vcr-record=all -k "test_name"
```

Expand only when the migration touches shared code:

```bash
cd py && make test-core
cd py && make lint
```

Also verify:

- the old wrapper import path still works
- the old `wrap_*()` helpers still work
- deprecated aliases still resolve
- the relevant auto-instrument subprocess tests still pass if `auto.py` changed

## Migration-Specific Pitfalls

Avoid these failures:

- copying wrapper code into `integrations/` without restructuring it around `__init__.py`, `integration.py`, `patchers.py`, and `tracing.py`
- leaving business logic or tracing helpers in the wrapper after the migration
- preserving wrapper-era `hasattr` or patch-state logic in tracing wrappers instead of using patcher primitives
- re-implementing target precedence with custom branching instead of `superseded_by`
- forgetting to move cassettes or auto-test scripts with the tests
- updating tests but forgetting `py/noxfile.py`
- breaking deprecated aliases, `__all__`, or old import paths
- changing shared integration code more broadly than the provider requires
- re-recording cassettes when behavior did not intentionally change
