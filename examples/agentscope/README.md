# AgentScope + Braintrust

Calls `braintrust.auto_instrument()` to register the AgentScope tracing hooks, then runs a `ReActAgent` against `gpt-4o-mini`. The trace shows the agent's `*.reply` task span and the underlying LLM span.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
