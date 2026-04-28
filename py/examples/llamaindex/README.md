# LlamaIndex + Braintrust

Calls `braintrust.auto_instrument()` to register a `BraintrustSpanHandler` on the LlamaIndex global dispatcher, then issues a single LLM completion. The trace captures the LlamaIndex span hierarchy plus the underlying provider call.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
