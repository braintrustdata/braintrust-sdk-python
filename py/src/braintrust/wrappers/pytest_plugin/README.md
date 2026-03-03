# Braintrust Pytest Plugin

Automatically track pytest test results as [Braintrust](https://www.braintrust.dev) experiments. Mark tests with `@pytest.mark.braintrust`, run with `--braintrust`, and get experiment tracking with spans, pass/fail scores, and custom metrics.

## Requirements

- Python >= 3.10
- pytest >= 8

## Installation

The plugin is included with the Braintrust SDK and auto-registers via `pytest11` entry point:

```bash
pip install braintrust
```

## Quick Start

```python
import pytest

@pytest.mark.braintrust
def test_my_llm(braintrust_span):
    result = my_llm("hello")
    braintrust_span.log(input={"query": "hello"}, output=result)
    assert "greeting" in result
```

Run with:

```bash
pytest --braintrust --braintrust-project="my-project"
```

## Configuration

### CLI Options

| Option                    | Env Var                 | Description                               |
| ------------------------- | ----------------------- | ----------------------------------------- |
| `--braintrust`            | —                       | Enable experiment tracking (required)     |
| `--braintrust-project`    | `BRAINTRUST_PROJECT`    | Project name (default: test module name)  |
| `--braintrust-experiment` | `BRAINTRUST_EXPERIMENT` | Experiment name (default: auto-generated) |
| `--braintrust-api-key`    | `BRAINTRUST_API_KEY`    | API key for Braintrust                    |
| `--braintrust-no-summary` | —                       | Suppress terminal summary                 |

### Marker kwargs

```python
@pytest.mark.braintrust(
    project="my-project",      # Override project for this test/class
    input={"query": "hello"},  # Static input data
    expected={"answer": "hi"}, # Expected output
    tags=["regression"],       # Tags for the span
    metadata={"model": "gpt-4"},  # Additional metadata
)
```

## Logging Data

The `braintrust_span` fixture is a standard [`Span`](https://www.braintrust.dev/docs/reference/python#span) object. Use `span.log()` to record data:

```python
def test_example(braintrust_span):
    braintrust_span.log(
        input={"query": "hello"},
        output={"response": "world"},
        expected={"response": "world"},
        scores={"accuracy": 0.95},
        metadata={"model": "gpt-4"},
    )
```

When `--braintrust` is not passed, the fixture returns a no-op span that silently discards all logged data, so tests still pass normally.

## Experiment Grouping

By default, one experiment is created per test module. Override with:

- **CLI**: `--braintrust-project="name"` (applies to all tests)
- **Marker**: `@pytest.mark.braintrust(project="name")` (per-test or per-class)

## Data-Driven Tests

Parametrized test arguments are automatically logged as input:

```python
@pytest.mark.braintrust
@pytest.mark.parametrize("query,expected_answer", [
    ("2+2?", "4"),
    ("Capital of France?", "Paris"),
])
def test_qa(braintrust_span, query, expected_answer):
    result = my_llm(query)
    braintrust_span.log(output=result)
```

Each parametrized case becomes a separate span with `input: {"query": "2+2?", "expected_answer": "4"}`.

If you provide `input` via the marker, it takes precedence over auto-logged arguments.

## Viewing Results

After a test run, the terminal shows an experiment summary:

```
=========================SUMMARY=========================
95.50% 'pass'  score

See results for my-test-2026-03-02T15:30:00 at https://www.braintrust.dev/app/...
```

Click the URL to view detailed results in the Braintrust UI.
