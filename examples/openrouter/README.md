# OpenRouter + Braintrust

Calls `braintrust.auto_instrument()` to patch the OpenRouter SDK so chat, embeddings, and responses calls are traced. The resulting LLM spans include the provider that OpenRouter dispatched to (e.g. `metadata.provider == "openai"`).

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENROUTER_API_KEY=...

uv sync
uv run python example.py
```
