# Google ADK + Braintrust

Calls `braintrust.auto_instrument()` so Google ADK is patched automatically, then runs a weather agent against `gemini-2.0-flash` with one tool. The trace shows the runner span, the agent span, the LLM span, and the tool call.

## Run

```bash
export BRAINTRUST_API_KEY=...
export GOOGLE_API_KEY=...

uv sync
uv run python example.py
```
