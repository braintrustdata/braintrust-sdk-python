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

## Logging With an Ingestion Key

Ingestion keys let apps that run on your users' machines, like a desktop app or a CLI, send traces to a single project without an API key.
Pass the ingestion key URL from your project settings to `init_logger`, or set `BRAINTRUST_INGESTION_KEY`:

```python
import braintrust

logger = braintrust.init_logger(ingestion_key="https://dp.example.com/ingest?ingestKey=bt-ik-...")
logger.log(input="question", output="answer")
```

The logger sends rows and attachments straight to the data plane in the URL, with the key in the `Authorization` header.
It never logs in, so it ignores `BRAINTRUST_API_KEY` and any earlier `braintrust.login()`, and it doesn't register or look up the project.
Passing an explicit `api_key` to `init_logger` uses that key instead of `BRAINTRUST_INGESTION_KEY`, and passing both `api_key` and `ingestion_key` is an error.
Feedback logged with an ingestion key can only include scores, expected values, and tags.

To export OpenTelemetry spans with an ingestion key, point a standard OTLP exporter at the `/otel/v1/traces` path under the ingestion URL and drop the query string:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="https://dp.example.com/ingest/otel/v1/traces",
    headers={"Authorization": "Bearer bt-ik-..."},
)
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
- Repo publishing guide: https://github.com/braintrustdata/braintrust-sdk-python/blob/main/docs/publishing.md
- Source code: https://github.com/braintrustdata/braintrust-sdk-python/tree/main/py

## License

Apache-2.0
