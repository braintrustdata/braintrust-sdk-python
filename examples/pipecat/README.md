# Pipecat voice tracing

Two small voice agents using real OpenAI services and the local Braintrust SDK:

| Example | Pipeline |
| --- | --- |
| `cascade.py` | Speech → OpenAI transcription → LLM + order lookup → synthesized speech |
| `realtime.py` | Speech → OpenAI Realtime + order lookup → speech |

Requires Python 3.11–3.13, `uv`, `OPENAI_API_KEY`, and `BRAINTRUST_API_KEY` in your environment. From the repository root, run either command:

```sh
uv run --project examples/pipecat python examples/pipecat/cascade.py
uv run --project examples/pipecat python examples/pipecat/realtime.py
```

Each command installs its dependencies, runs one conversation, and prints a trace link in the `example-pipecat` project. OpenAI usage is billable.

The bundled `order.wav` (about 82 KiB) asks “Where is my order number one zero four two?” `common.py` supplies file input and silent output so no microphone, speaker, browser, or phone setup is needed. The local `lookup_order` tool returns a fictional delivery date. Listen to the captured audio in the trace.

Instrumentation is enabled with:

```python
logger = braintrust.init_logger(project="example-pipecat")
setup_pipecat(capture_audio_attachments=True)
```

Pipecat metrics are enabled in `PipelineParams`. The trace includes turns, model calls, tool execution, metrics, and Ogg audio attachments. Set `capture_audio_attachments=False` to trace without recording audio.
