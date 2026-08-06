# Cursor SDK + Braintrust

Calls `braintrust.auto_instrument()` to wrap Cursor's `Agent`/`AsyncAgent` runs, then sends a single prompt. The trace shows a task span for the run, an LLM span per model turn, and a tool span for each tool the agent invokes.

Cursor executes the model calls inside its own bridge subprocess, so the LLM span is reconstructed from the run's streamed events rather than from a provider client.

## Run

```bash
export BRAINTRUST_API_KEY=...
export CURSOR_API_KEY=...

uv sync
uv run python example.py
```
