---
name: sdk-integrations
description: Create or update Braintrust Python SDK integrations built on the integrations API under `py/src/braintrust/integrations/`. Use when adding a new integration package, extending an existing provider integration, changing patchers, tracing, span shape, manual `wrap_*()` helpers, integration exports, `auto_instrument()` wiring, `py/noxfile.py` sessions, integration tests, or cassettes.
---

# SDK Integrations

Work under `py/src/braintrust/integrations/`. For cassette recording and VCR debugging, also use `sdk-vcr-workflows`.

## Spec

The instrumentation spec is the authority for span shape, allowed fields, metrics, and instrumentation rules. Start at https://github.com/braintrustdata/braintrust-spec/blob/main/skills/instrumentation-spec/SKILL.md. It points to `references/instrumentation-guide.md` (the full guide) and `references/features/*.md` (decision spans, MCP servers, prompt cache, attachments, token and cost metrics, and others). Fetch with:

```bash
gh api repos/braintrustdata/braintrust-spec/contents/skills/instrumentation-spec/references/instrumentation-guide.md --jq .content | base64 --decode
```

There is no `docs/instrumentation-guide.md` in that repo; it moved in July 2026. Read the spec sections and feature files that cover the span shape you are changing before you write code. Do not invent new top-level span fields, metric keys, or span types; change the spec first.

## Read First

Always: `integrations/base.py`, `integrations/versioning.py`, `integrations/__init__.py`, `integrations/utils.py`, `pyproject.toml`, `noxfile.py`.

Existing integration: `integrations/<provider>/{__init__,integration,patchers,tracing,test_*}.py`.

Situational:
- `auto.py` + `integrations/auto_test_scripts/` — anything touching `auto_instrument()` or import timing. Import-order and subprocess regressions usually only surface here.
- `conftest.py`, `integrations/conftest.py` — VCR / per-version cassette resolution
- `integrations/test_utils.py` — shared attachment / multimodal helpers

References in this skill (open when the task needs them):
- [references/span-design.md](references/span-design.md) — span type/name, completion vs agentic, payload format, `metadata` fields, token ownership, metric keys, attachments. Open before defining or changing span shape.
- [references/span-origin.md](references/span-origin.md) — the `_INSTRUMENTATION = "<provider>-auto"` pattern. Open when creating spans in a new integration or a new code path.
- [references/streaming-lifecycle.md](references/streaming-lifecycle.md) — exit paths and GC finalization for streamed responses. Open when wrapping any stream.

## Version And CI Routing

Never guess versions or session names:

- `pyproject.toml` `[tool.braintrust.matrix]` — supported provider versions, what `latest` resolves to
- `pyproject.toml` `[tool.braintrust.cassette-dirs]` — versioned cassette directory ownership
- `integrations/versioning.py` — supported version helpers, gates
- `noxfile.py` — session names, package install, `BRAINTRUST_TEST_PACKAGE_VERSION`
- `.github/workflows/checks.yaml` — CI matrix + shard membership

When changing version-gated behavior: identify every matrix version, check `min_version`/`max_version`/`superseded_by`/feature-detect branches, test the narrowest affected version first, re-record cassettes only for versions whose observable provider behavior intentionally changed.

## Pick A Reference Integration

- **ADK** — direct method patching, `target_module`, `CompositeFunctionWrapperPatcher`, manual `wrap_*()`, context propagation, inline data → `Attachment`
- **Agno** — multi-target patching, related patchers, version-conditional fallbacks with `superseded_by`
- **Anthropic** — compact constructor patching, small public surface, multimodal image-vs-document attachment payloads
- **Google GenAI** — multimodal tracing, generated media, output-side attachments, nested materialization that preserves non-attachment values

## Default Workflow

1. Read the spec section(s) covering the affected span shape.
2. Read the provider package + shared primitives.
3. Define span shape (`type`, `name`, `input`, `output`, `metadata`, `metrics`, `error`) before writing patchers. See [references/span-design.md](references/span-design.md).
4. Pick the public API surface to patch — prefer stable public entry points over internal helpers.
5. Implement patchers, then tracing helpers.
6. For behavior changes: add/update a failing test first, then fix (red → green). Cassette-backed if the provider payload triggers the bug.
7. Run the narrowest nox session first. Expand only if shared code changed.
8. Consolidate tests (see [Testing Rules](#testing-rules)) before handoff.

## Route The Task

**New integration:**
1. Create `integrations/<provider>/` with `__init__.py` (exports, `setup_<provider>()`, `wrap_*()`), `integration.py` (thin `BaseIntegration` subclass), `patchers.py`, `tracing.py`, `test_<provider>.py`, and `cassettes/<version>/` (one dir per matrix version + `latest/`).
2. Export from `integrations/__init__.py`.
3. Add the provider session in `noxfile.py` and the `[tool.braintrust.cassette-dirs]` entry in `pyproject.toml`.
4. Update `auto.py` only if it should participate in `auto_instrument()`; prefer `_instrument_integration(...)` over a custom helper, and add a subprocess test in `auto_test_scripts/`.
5. Add the span-origin shadow ([references/span-origin.md](references/span-origin.md)) and assert it in tests.

**Existing integration update:** change only affected patchers, tracing, exports, tests, cassettes. Preserve the public `setup_*()` / `wrap_*()` surface. Don't touch `auto.py` / `integrations/__init__.py` / `noxfile.py` unless required.

**Entry-point parity:** `setup_<provider>()`, `wrap_*()`, and `auto_instrument()` must emit equivalent spans. If a behavior change lands in one, check the other two. Validate `auto_instrument()` with a subprocess test, not just in-process.

Shared logic goes in `integrations/utils.py`, not copies. Check it and neighboring integrations before writing a local helper.

## Integration Rules

Declarative setup on the `BaseIntegration` subclass: `name`, `import_names`, `patchers`. Use `min_version`/`max_version` only when feature detection isn't enough. Prefer `detect_module_version(...)` / `version_satisfies(...)` / `make_specifier(...)`. Let `BaseIntegration.resolve_patchers()` reject duplicate patcher ids.

- Provider errors MUST propagate. Never swallow or rewrap.
- Preserve return types, iterator/async-iterator semantics, generator/subclass behavior. Keep sync and async traced schemas aligned.
- Setup / teardown / patching MUST be idempotent (rely on the base patcher marker).
- Instrumentation failures (extraction, normalization) must never raise into the user's call path. But do not wrap `span.log(...)` / `span.set_attributes(...)` in try/except — they are boundary-safe.
- Treat provider inputs/results/events/headers as untrusted. Don't mutate third-party objects.
- Only instrument AI-generation-relevant ops (LLM calls, embeddings, tool exec, media gen, agent runs).
- Do not over-serialize. `span.log` already runs `bt_safe_deep_copy` (pydantic, dataclasses, attachments). Log readable dicts/lists/scalars and materialized attachments; no JSON dumps/loads or recursive conversion in tracing code.
- Every `llm` span has `metadata.model` and `metadata.provider` (whose pricing applies, not the framework name).

## Patcher Rules

One patcher per coherent target: `FunctionWrapperPatcher` (one import path / method surface), `CompositeFunctionWrapperPatcher` (one logical surface across related targets), `CallbackPatcher` (setup side effects). Use `target_module` for deep or optional submodules, `superseded_by` for version-conditional fallbacks, lower `priority` only when ordering matters. Manual `wrap_*()` helpers stay thin: `return AgentPatcher.wrap_target(Agent)`.

## Testing Rules

Tests live in the provider package. Red → green for bug fixes.

- **VCR, not mocks or fakes.** Use `@pytest.mark.vcr` with real provider recordings. Mocks are only for narrow error injection, purely local version-routing logic, or patcher existence checks. A fix in `tracing.py` still gets a cassette-backed regression when a real payload triggers the bug. Hand-built stand-ins for provider objects count as fakes.
- **Assert on emitted spans**, not just provider return values: `type`, `name`, `input`/`output` shape, `metadata.model`/`provider`/`tools`, required `metrics`, parent/child order for agentic APIs, `context.span_origin.instrumentation.name`, attachments, error logging.
- **Streaming:** assert the provider iterator still works and the final span has aggregated `output` + `time_to_first_token`. Follow [references/streaming-lifecycle.md](references/streaming-lifecycle.md) for exit-path coverage.
- **Cassettes** live in `integrations/<provider>/cassettes/<version>/`; `integrations/conftest.py` resolves the version dir — never add per-test `vcr_cassette_dir` fixtures. Re-record only when behavior intentionally changed. Sanitize binary media in request and response bodies.
- **Consolidate before handoff (required, do not wait to be asked).** For every test file you touched, merge tests that differ only in inputs into one `@pytest.mark.parametrize` test, and delete tests whose assertions another test already makes (for example a sync test that a parametrized sync/async test covers). Keep the red → green regression test. Then run `make check-unused-cassettes INTEGRATIONS="<provider>"` and delete cassettes no test reads. In the handoff, list the tests merged or removed and the test count before and after.

## Commands

```bash
cd py && nox -s "test_<session>(latest)"                      # confirm the session name in noxfile.py
cd py && nox -s "test_<session>(latest)" -- -k "test_name"
cd py && nox -s "test_<session>(latest)" -- --vcr-record=all -k "test_name"
cd py && make test-core                                         # shared integration code changed
cd py && make lint                                              # shared files or repo wiring changed
```

Also run the subprocess auto-instrument test if patchers, setup, import timing, or `auto.py` changed.

## Common Mistakes

- Changing shared primitives when provider-local code should own the behavior.
- Combining unrelated patch targets into one patcher, or patching internal helpers when a public surface works.
- Forgetting repo wiring for new providers (`integrations/__init__.py`, `noxfile.py`, `cassette-dirs`, sometimes `auto.py`).
- Forgetting subprocess auto-instrument tests, or async/streaming coverage.
- Tool definitions in `input` instead of `metadata.tools`; prompt provenance outside `metadata.prompt`.
- Framework name in `metadata.provider`; per-chunk spans for streamed calls; double-counting tokens across framework + provider spans.
- Missing `internal={"instrumentation": "<provider>-auto"}` on integration spans, or leaking it onto nested user spans.
- Handing off with redundant or overlapping tests.
