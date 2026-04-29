# OpenAI + Braintrust

Calls `braintrust.auto_instrument()` to patch `openai`, then uses `@braintrust.traced` to wrap the surrounding function so its inputs and outputs become a parent span. Every chat completion call is recorded as a child LLM span automatically.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
