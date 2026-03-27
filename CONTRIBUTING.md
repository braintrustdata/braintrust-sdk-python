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

### Claude Agent SDK Subprocess Cassettes

`claude_agent_sdk` tests use the real SDK and bundled `claude` CLI, but they do not use VCR. Instead they record and replay the SDK/CLI JSON transport under:

- `py/src/braintrust/wrappers/claude_agent_sdk/cassettes/`

Behavior:

- Locally, subprocess cassettes default to `once`.
- In CI, subprocess cassettes default to `none`.
- Override with `BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all` when you need to re-record.

Useful examples:

```bash
cd py
nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all \
  nox -s "test_claude_agent_sdk(latest)" -- -k "test_calculator_with_multiple_operations"
```

### Fixtures

Shared test fixtures live in `py/src/braintrust/conftest.py`.

Common ones include:

- dummy API key setup for VCR-backed tests
- Braintrust global state reset between tests
- wheel-mode skipping for VCR tests

The `memory_logger` fixture from `braintrust.test_helpers` is useful for asserting on logged spans without a real Braintrust backend.

## Benchmarks

The SDK includes local performance benchmarks powered by [pyperf](https://pyperf.readthedocs.io/), located in `py/benchmarks/`. These cover hot paths like serialization and deep-copy routines.

### Running benchmarks

```bash
cd py

# Run all benchmarks
make bench

# Quick sanity check (fewer iterations)
make bench BENCH_ARGS="--fast"

# Save results for later comparison
make bench BENCH_ARGS="-o /tmp/results.json"

# Run a single benchmark module directly
python -m benchmarks.benches.bench_bt_json
```

To benchmark with the optional `orjson` fast-path installed:

```bash
cd py
python -m uv pip install -e '.[performance]'
make bench
```

### Comparing across branches

```bash
cd py

git checkout main
make bench BENCH_ARGS="-o /tmp/main.json"

git checkout my-branch
make bench BENCH_ARGS="-o /tmp/branch.json"

make bench-compare BENCH_BASE=/tmp/main.json BENCH_NEW=/tmp/branch.json
```

### Useful pyperf flags

| Flag            | Purpose                                           |
| --------------- | ------------------------------------------------- |
| `--fast`        | Fewer iterations — good for a quick sanity check  |
| `--rigorous`    | More iterations — reduces noise for final numbers |
| `-o FILE`       | Write results to a JSON file for later comparison |
| `--append FILE` | Append to an existing results file                |

Run `python -m benchmarks --help` for the full list.

### Adding a new benchmark

Drop a new `bench_<name>.py` file into `py/benchmarks/benches/`. It will be picked up automatically — no registration required.

Your module needs to expose a `main()` function that accepts an optional `pyperf.Runner`:

```python
import pyperf

from benchmarks._utils import disable_pyperf_psutil


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("my_benchmark", my_func, my_arg)


if __name__ == "__main__":
    main()
```

If your benchmark needs reusable test data, add builder functions to `py/benchmarks/fixtures.py`.

## CI

GitHub Actions workflows live in `.github/workflows/`.

Main workflows:

- `checks.yaml`: merged SDK checks workflow, including lint, pinned-action validation, the Python test matrix, wheel build, and the `checks-passed` required-check aggregator
- `langchain-py-test.yaml`: LangChain integration tests
- `adk-py-test.yaml`: ADK integration tests
- `publish-py-sdk.yaml`: PyPI release
- `test-publish-py-sdk.yaml`: TestPyPI release validation

CI uses committed HTTP VCR cassettes and Claude Agent SDK subprocess cassettes, so forks do not need provider API secrets for normal replayed test runs.

## Submitting Changes

1. Make your change in the narrowest relevant area.
2. Add or update tests.
3. Re-record HTTP or Claude Agent SDK subprocess cassettes if the provider interaction change is intentional.
4. Run the smallest relevant local checks first, then broader ones if needed.
5. Run `make fixup` before opening a PR.
6. Open a pull request against `main`.
