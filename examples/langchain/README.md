# LangChain + Braintrust

Calls `braintrust.auto_instrument()`, which patches `langchain_core.tracers.context` to install a global `BraintrustCallbackHandler`. After that, every chain run is traced with no `config={"callbacks": [...]}` plumbing required — the trace shows the `RunnableSequence` task span with `ChatPromptTemplate` and `ChatOpenAI` children.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
