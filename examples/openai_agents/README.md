# OpenAI Agents SDK + Braintrust

Calls `braintrust.auto_instrument()` to install `BraintrustTracingProcessor` as the SDK's tracing processor, then runs a small `Agent` via `Runner`. The trace shows the agent run as a root span with each model call nested underneath.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
