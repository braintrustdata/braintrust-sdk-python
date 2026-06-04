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

## Replay Trace Exports

Use `braintrust replay` to turn a saved trace export into a local regression
check. This is useful when you want to rerun a task or scorer against a
production trace shape without sending a new experiment to Braintrust.

```bash
braintrust replay trace.json \
  --task my_agent:answer \
  --score my_scores:answer_quality \
  --min-score answer_quality=0.85 \
  --min-score-delta answer_quality=0 \
  --fail-on-error \
  --json
```

The trace file can be JSONL, a JSON list of span rows, or a JSON object with a
`spans` field. Rows use the same fields Braintrust spans expose, including
`span_id`, `root_span_id`, `input`, `output`, `expected`, `scores`, `metrics`,
`metadata`, and `span_attributes`.

Replay tasks receive the root span input and may also accept `expected`,
`metadata`, and `trace` keyword arguments:

```python
def answer(input, trace):
    return app.answer(input["messages"])
```

Scorers use the same common arguments as eval scorers:

```python
async def answer_quality(input, output, expected, trace):
    tool_spans = await trace.get_spans(["tool"])
    return output == expected
```

The report includes current scores, baseline scores from the original root
span, score deltas, derived trace metrics, and metric deltas. Threshold flags
make the command useful in CI when an agent or scorer change should not regress
against saved production traces.

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
