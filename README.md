# Braintrust Python SDKs

[Braintrust](https://www.braintrust.dev/) is a platform for evaluating and shipping AI products. Learn more at [braintrust.dev](https://www.braintrust.dev/) and in the [docs](https://www.braintrust.dev/docs).

This repository contains Braintrust's Python SDKs and integrations, including:

- The main `braintrust` SDK package in [`./py`](./py)
- Integration packages under [`./integrations`](./integrations)
- Examples, tests, and local development tooling for Python SDK development

## Quickstart

Install the main SDK and scorer package:

```bash
pip install braintrust autoevals
```

Create `tutorial_eval.py`:

```python
from autoevals import LevenshteinScorer
from braintrust import Eval

Eval(
    "Say Hi Bot",
    data=lambda: [
        {"input": "Foo", "expected": "Hi Foo"},
        {"input": "Bar", "expected": "Hello Bar"},
    ],
    task=lambda input: "Hi " + input,
    scores=[LevenshteinScorer],
)
```

Run it:

```bash
BRAINTRUST_API_KEY=<YOUR_API_KEY> braintrust eval tutorial_eval.py
```

## Packages

| Package | Purpose | PyPI | Docs |
| --- | --- | --- | --- |
| `braintrust` | Core Python SDK for logging, tracing, evals, and CLI workflows. | [![PyPI - braintrust](https://img.shields.io/pypi/v/braintrust.svg)](https://pypi.org/project/braintrust/) | [py/README.md](py/README.md) |
| `braintrust-langchain` | LangChain callback integration for automatic Braintrust logging. | [![PyPI - braintrust-langchain](https://img.shields.io/pypi/v/braintrust-langchain.svg)](https://pypi.org/project/braintrust-langchain/) | [integrations/langchain-py/README.md](integrations/langchain-py/README.md) |
| `braintrust-adk` | Deprecated Google ADK integration package. New ADK support lives in `braintrust`. | [![PyPI - braintrust-adk](https://img.shields.io/pypi/v/braintrust-adk.svg)](https://pypi.org/project/braintrust-adk/) | [integrations/adk-py/README.md](integrations/adk-py/README.md) |

## Documentation

- Python SDK docs: https://www.braintrust.dev/docs/reference/sdks/python
- Release notes: https://www.braintrust.dev/docs/reference/release-notes

## License

Apache-2.0
