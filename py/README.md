# Braintrust Python SDK

[![PyPI version](https://img.shields.io/pypi/v/braintrust.svg)](https://pypi.org/project/braintrust/)

The official Python SDK for logging, tracing, and evaluating AI applications with [Braintrust](https://www.braintrust.dev/).

## Installation

Install the SDK:

```bash
pip install braintrust
```

## Quickstart

Run a simple evaluation:

```python
from braintrust import Eval


def is_equal(expected, output):
    return expected == output


Eval(
    "Say Hi Bot",
    data=lambda: [
        {"input": "Foo", "expected": "Hi Foo"},
        {"input": "Bar", "expected": "Hello Bar"},
    ],
    task=lambda input: "Hi " + input,
    scores=[is_equal],
)
```

Then run:

```bash
BRAINTRUST_API_KEY=<YOUR_API_KEY> braintrust eval tutorial_eval.py
```

## Optional Extras

Install extras as needed for specific workflows:

```bash
pip install "braintrust[cli]"
pip install "braintrust[openai-agents]"
pip install "braintrust[otel]"
pip install "braintrust[temporal]"
pip install "braintrust[all]"
```

Available extras:

- `performance`: installs `orjson` for faster JSON serialization
- `cli`: installs optional dependencies used by the Braintrust CLI
- `openai-agents`: installs OpenAI Agents integration support
- `otel`: installs OpenTelemetry integration dependencies
- `temporal`: installs Temporal integration dependencies
- `all`: installs all optional extras

## Documentation

- Python SDK docs: https://www.braintrust.dev/docs/reference/sdks/python
- Braintrust docs: https://www.braintrust.dev/docs
- Source code: https://github.com/braintrustdata/braintrust-sdk-python/tree/main/py

## License

Apache-2.0
