# Pydantic AI + Braintrust

Calls `braintrust.auto_instrument()` so the Pydantic AI agent's underlying provider calls are captured as Braintrust spans automatically. The agent itself is wrapped in `start_span` so the run gets a permalink.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
