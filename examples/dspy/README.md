# DSPy + Braintrust

Calls `braintrust.auto_instrument()` **before** importing DSPy. That single call patches both LiteLLM (so token metrics propagate up from the underlying provider call) and DSPy's `configure()` (so the Braintrust callback gets attached automatically — no manual `BraintrustDSpyCallback` wiring required).

The example then runs a `ReAct` agent with two tools to produce a rich nested trace: the module span, every LM step, every tool call, and full token usage.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
