# Anthropic + Braintrust

Calls `braintrust.auto_instrument()` to patch the Anthropic SDK so both sync and async clients are traced automatically. The async script also exercises `messages.stream(...)` and `stream=True` create calls.

## Run

```bash
export BRAINTRUST_API_KEY=...
export ANTHROPIC_API_KEY=...

uv sync
uv run python sync.py
uv run python async_example.py
```
