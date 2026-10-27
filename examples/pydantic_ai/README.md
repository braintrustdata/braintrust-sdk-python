# Pydantic AI + Braintrust

Calls `braintrust.auto_instrument()`, which traces the Pydantic AI agent. The script also opens `start_span` so it can print a permalink.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
