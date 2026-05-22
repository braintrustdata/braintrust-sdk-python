# AutoGen + Braintrust

Calls `braintrust.auto_instrument()` to install the AutoGen patcher, then runs a single `AssistantAgent` backed by `OpenAIChatCompletionClient`. The trace shows the agent's task span and the OpenAI chat completion span underneath.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
