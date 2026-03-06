# Contributing

Guide for contributing to the Braintrust Python SDK.

## Setup

### Prerequisites

- Python 3.10+
- [mise](https://mise.jdx.dev/) for tool installation and repo-local environment management

### Getting Started

```bash
git clone https://github.com/braintrustdata/braintrust-sdk-python.git
cd braintrust-sdk-python
mise install
make develop
```

If you use `mise activate` in your shell, entering the repo will automatically expose the configured tools. If you do not, you can still run commands explicitly with `mise exec -- ...`.

## Repo Layout

- `py/`: main Python SDK
- `integrations/`: separate integration packages such as LangChain and ADK
- `internal/golden/`: golden and compatibility projects
- `docs/`: supporting docs

Most SDK changes should happen under `py/`.

## Common Workflows

### Python SDK

```bash
cd py
make install-dev
make test-core
make lint
nox -l
```

Run a focused session:

```bash
cd py
nox -s "test_openai(latest)"
```

Run a single test subset:

```bash
cd py
nox -s "test_openai(latest)" -- -k "test_chat_metrics"
```

Install optional provider packages only when you need them:

```bash
cd py
make install-optional
```

### Repo-Level Commands

The root `Makefile` is a convenience wrapper around `py/Makefile`.

Useful root commands:

```bash
make fixup
make test-core
make lint
```

`make test-wheel` requires a built wheel first.

### Integration Packages

LangChain:

```bash
cd integrations/langchain-py
uv sync
uv run pytest src
```

ADK:

```bash
cd integrations/adk-py
uv sync
uv run pytest
```

## Testing Notes

The SDK uses [nox](https://nox.thea.codes/) for compatibility testing across optional providers and versions. `py/noxfile.py` is the source of truth for available sessions.

### VCR Tests

Many wrapper and devserver tests use VCR cassettes.

- Locally, missing cassettes can be recorded with `record_mode="once"`.
- In CI, missing cassettes fail because `record_mode="none"` is used.
- If your change intentionally changes HTTP behavior, re-record the affected cassettes and commit them.

Useful example:

```bash
cd py
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

### Fixtures

Shared test fixtures live in `py/src/braintrust/conftest.py`.

Common ones include:

- dummy API key setup for VCR-backed tests
- Braintrust global state reset between tests
- wheel-mode skipping for VCR tests

The `memory_logger` fixture from `braintrust.test_helpers` is useful for asserting on logged spans without a real Braintrust backend.

## CI

GitHub Actions workflows live in `.github/workflows/`.

Main workflows:

- `py.yaml`: SDK test matrix
- `langchain-py-test.yaml`: LangChain integration tests
- `adk-py-test.yaml`: ADK integration tests
- `lint.yaml`: pre-commit and formatting checks
- `publish-py-sdk.yaml`: PyPI release
- `test-publish-py-sdk.yaml`: TestPyPI release validation

CI uses VCR cassettes and dummy credentials, so forks do not need provider API secrets for normal test runs.

## Submitting Changes

1. Make your change in the narrowest relevant area.
2. Add or update tests.
3. Re-record cassettes if the HTTP behavior change is intentional.
4. Run the smallest relevant local checks first, then broader ones if needed.
5. Run `make fixup` before opening a PR.
6. Open a pull request against `main`.
