# Cohere + Braintrust

Calls `braintrust.auto_instrument()` to patch the Cohere SDK so every chat, chat-stream, embed, rerank, and audio-transcription call is traced. Patches all four client variants — `cohere.Client`, `cohere.AsyncClient`, `cohere.ClientV2`, and `cohere.AsyncClientV2` — so any new instance is captured automatically.

## Run

```bash
export BRAINTRUST_API_KEY=...
export COHERE_API_KEY=...

uv sync
uv run python example.py
```
