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
pip install "braintrust[audio]"
pip install "braintrust[cli]"
pip install "braintrust[openai-agents]"
pip install "braintrust[otel]"
pip install "braintrust[temporal]"
pip install "braintrust[all]"
```

Available extras:

- `audio`: installs PyAV for Opus/Ogg session recording on Python 3.12+
- `performance`: installs `orjson` for faster JSON serialization
- `cli`: installs optional dependencies used by the Braintrust CLI
- `openai-agents`: installs OpenAI Agents integration support
- `otel`: installs OpenTelemetry integration dependencies
- `temporal`: installs Temporal integration dependencies
- `all`: installs all optional extras

### Audio encoding

`braintrust[audio]` requires Python 3.12+ and uses PyAV's bundled FFmpeg codecs.
On Python 3.10–3.11, the extra does not install PyAV; use PCM/WAV recording.
Each Opus/Ogg segment is an independent file with exact playable duration. Installing the extra does not
enable capture; framework adapters still require their audio opt-ins and privacy
permission. Without PyAV, default recording warns once and uses raw PCM. Explicit
Opus configuration requires the extra. PCM/WAV need no native codecs.

Audio uploads use `Attachment.upload()` directly. Span customizers and masking
apply to trace records published afterward; removing an attachment reference does
not prevent or undo its upload.

## Documentation

- Python SDK docs: https://docs.braintrust.dev/sdks/python/versions/latest
- Braintrust docs: https://docs.braintrust.dev
- Repo publishing guide: https://github.com/braintrustdata/braintrust-sdk-python/blob/main/docs/publishing.md
- Source code: https://github.com/braintrustdata/braintrust-sdk-python/tree/main/py

## License

Apache-2.0
