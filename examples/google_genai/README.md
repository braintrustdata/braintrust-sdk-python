# Google GenAI + Braintrust

Calls `braintrust.auto_instrument()` to patch the Google GenAI client, then runs a single `generate_content` call against `gemini-2.0-flash-001`. The trace captures the LLM span with token metrics.

## Run

```bash
export BRAINTRUST_API_KEY=...
export GOOGLE_API_KEY=...        # or GEMINI_API_KEY

uv sync
uv run python example.py
```
