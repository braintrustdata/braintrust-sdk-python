# Mistral + Braintrust

Calls `braintrust.auto_instrument()` to patch the Mistral SDK so every chat, FIM, agents, conversations, embeddings, OCR, speech, and transcription call is traced — the full surface of the SDK. Any newly-constructed `Mistral` client is captured automatically.

## Run

```bash
export BRAINTRUST_API_KEY=...
export MISTRAL_API_KEY=...

uv sync
uv run python example.py
```
