# Contributing

Guide for contributing to the Braintrust Python SDK.

## Repository Structure

```
braintrust-sdk-python/
├── py/                  # Python SDK
│   ├── src/braintrust/  # Source code
│   │   ├── wrappers/    # Provider wrappers (OpenAI, Anthropic, Google, etc.)
│   │   ├── contrib/     # Community integrations (Temporal, etc.)
│   │   ├── devserver/   # Local dev server / CLI
│   │   └── conftest.py  # Shared test fixtures
│   ├── noxfile.py       # Test session definitions
│   └── Makefile         # Build/test commands
├── integrations/
│   ├── langchain-py/    # LangChain integration
│   └── adk-py/          # Google ADK integration
├── internal/            # Golden tests
├── scripts/             # Dev scripts
├── docs/                # Documentation
└── Makefile             # Top-level commands
```

## Setup

### Prerequisites

- Python 3.10+ (3.9 supported but some test sessions are skipped)
- [uv](https://github.com/astral-sh/uv) (installed automatically by `make install-dev`)

### Getting Started

```bash
# Clone the repo
git clone https://github.com/braintrustdata/braintrust-sdk-python.git
cd braintrust-sdk-python

# Create venv and install all dependencies
make develop

# Activate the environment
source env.sh
```

### Python SDK Development

```bash
cd py

# Install dev dependencies
make install-dev

# Install optional provider packages (for wrapper development)
make install-optional
```

## Running Tests

### Python SDK

Tests use [nox](https://nox.thea.codes/) to run across different dependency versions.

```bash
cd py

# Run all test sessions
make test

# Run core tests only (no optional dependencies)
make test-core

# List all available sessions
nox -l

# Run a specific session
nox -s "test_openai(latest)"
nox -s "test_anthropic(latest)"
nox -s "test_temporal(latest)"

# Run a single test within a session
nox -s "test_openai(latest)" -- -k "test_chat_metrics"
```

### Integration Tests

```bash
# LangChain
cd integrations/langchain-py
uv sync
uv run pytest src

# ADK
cd integrations/adk-py
uv sync
uv run pytest
```

### Linting

```bash
# From repo root — runs pre-commit hooks (formatting, etc.)
make fixup

# Python-specific lint (pylint)
cd py && make lint
```

## VCR Cassette Testing

Tests for API provider wrappers use VCR.py to record and replay HTTP interactions. This means most tests run without real API keys.

See [docs/vcr-testing.md](docs/vcr-testing.md) for full details. Key points:

- **Locally:** VCR records new cassettes on first run (`record_mode="once"`). You need a real API key to record.
- **In CI:** VCR only replays existing cassettes (`record_mode="none"`). No API keys needed.
- **Modifying tests:** If your change alters the HTTP request a test makes, you must re-record the cassette locally with a real API key and commit it.
- **New tests:** Add `@pytest.mark.vcr`, record the cassette locally, and commit the cassette file.

## CI Overview

CI runs on GitHub Actions. All workflows are in `.github/workflows/`.

### Workflows

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **py** | `py.yaml` | PR (py/integrations changes), push to main | Runs nox test matrix across Python 3.10–3.13 on Ubuntu + Windows, plus integration tests |
| **langchain-py** | `langchain-py-test.yaml` | Called by `py.yaml` | Lint + tests for the LangChain integration |
| **adk-py** | `adk-py-test.yaml` | Called by `py.yaml` | Lint + tests for the Google ADK integration |
| **lint** | `lint.yaml` | PR | Pre-commit hooks and formatting checks |
| **publish** | `publish-py-sdk.yaml` | Tag push (`py-sdk-v*.*.*`) | Build, test wheel, publish to PyPI, create GitHub release |
| **test-publish** | `test-publish-py-sdk.yaml` | Manual dispatch | Publish to TestPyPI for pre-release validation |

### No API Key Secrets Required

CI workflows do **not** use real API key secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Tests rely on VCR cassettes with dummy API keys provided by test fixtures. This means:

- Forks can run CI without configuring any secrets.
- The `test_latest_wrappers_novcr` nox session (which disables VCR) is automatically skipped in CI.

### Test Sharding

The main `py.yaml` workflow shards nox sessions across 2 parallel jobs per Python version/OS combination using `scripts/nox-matrix.sh`.

## Test Fixtures

Key auto-applied fixtures defined in `py/src/braintrust/conftest.py`:

| Fixture | Purpose |
|---|---|
| `setup_braintrust` | Sets dummy API keys (OpenAI, Google, Anthropic) for VCR tests |
| `override_app_url_for_tests` | Points `BRAINTRUST_APP_URL` to production for consistent behavior |
| `reset_braintrust_state` | Resets global SDK state after each test |
| `skip_vcr_tests_in_wheel_mode` | Skips VCR tests when testing from an installed wheel |

The `memory_logger` fixture (from `braintrust.test_helpers`) lets you capture logged spans in-memory without a real Braintrust connection:

```python
def test_something(memory_logger):
    # ... exercise code that logs spans ...
    spans = memory_logger.pop()
    assert len(spans) == 1
```

## Submitting Changes

1. Create a branch for your changes.
2. Make your changes and add/update tests.
3. If you modified VCR tests, re-record cassettes and commit them.
4. Run `make fixup` to format and lint.
5. Run relevant test sessions to verify (e.g. `nox -s "test_openai(latest)"`).
6. Open a pull request against `main`.
