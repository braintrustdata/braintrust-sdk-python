# LiteLLM + Braintrust

Calls `braintrust.auto_instrument()` to wrap the global `litellm` module so every call (`completion`, `acompletion`, `embedding`, `responses`, `image_generation`, `moderation`, `transcription`, `rerank`, and their async counterparts) is traced automatically with full token metrics, timing, and costs.

If you're using LiteLLM **inside** another framework (DSPy, for example), call `auto_instrument()` **before** importing that framework — see `examples/dspy/` for the pattern.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
