# LangChain + Braintrust

Calls `braintrust.auto_instrument()`, which patches `langchain_core.tracers.context` to install a global `BraintrustCallbackHandler`. After that, every chain run is traced with no `config={"callbacks": [...]}` plumbing required — the trace shows the `RunnableSequence` task span with `ChatPromptTemplate` and `ChatOpenAI` children.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```

## Anthropic prompt-cache metrics demo

Use it to verify cache reads/writes and token totals on real Braintrust spans.

```bash
# Loads BRAINTRUST_API_KEY and ANTHROPIC_API_KEY from ../../.env automatically.
uv sync
uv run python anthropic_prompt_cache.py
```

To inspect the logged spans with the Braintrust CLI:

```bash
bt projects list --json | jq '.[] | select(.name == "z-abhi-langchain-anthropic-cache-demo")'
bt view logs --object-ref project_logs:<project-id> --list-mode spans --limit 10 --json
```
