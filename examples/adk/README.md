# Google ADK + Braintrust

Calls `braintrust.auto_instrument()` so Google ADK is patched automatically, then runs a weather agent against `gemini-3.8-flash` with one tool.

## Run

```bash
export BRAINTRUST_API_KEY=...
export GOOGLE_API_KEY=...

uv sync
uv run python example.py
```
