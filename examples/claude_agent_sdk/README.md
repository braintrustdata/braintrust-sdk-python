# Claude Agent SDK + Braintrust

Calls `braintrust.auto_instrument()` to wrap `ClaudeSDKClient` and any registered tools, then runs a small query against the Claude Agent SDK subprocess. The trace shows the client span plus an LLM span per turn.

This example launches the bundled `claude` subprocess and talks to it over stdin/stdout — make sure the [Claude Agent SDK CLI](https://docs.anthropic.com/claude/docs/claude-agent-sdk) is installed and authenticated.

## Run

```bash
export BRAINTRUST_API_KEY=...
export ANTHROPIC_API_KEY=...

uv sync
uv run python example.py
```
