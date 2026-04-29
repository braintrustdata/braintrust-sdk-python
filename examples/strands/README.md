# Strands Agents + Braintrust

Calls `braintrust.auto_instrument()` to install the Strands tracer, then runs a single Strands `Agent` against `gpt-4o-mini`. The trace shows the `*.invoke` task span, the `event_loop.cycle` span, and the LLM span underneath.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
