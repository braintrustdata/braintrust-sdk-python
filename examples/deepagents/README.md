# Deep Agents + Braintrust

Deep Agents is built on LangChain and LangGraph, so Braintrust traces it through the existing LangChain integration. There is no separate `setup_deepagents()` step.

## Run

```bash
# Loads BRAINTRUST_API_KEY and OPENAI_API_KEY from ../../.env automatically.
uv sync
uv run python example.py
```

The trace includes the Deep Agents/LangGraph root span, model spans, tool spans, and Deep Agents metadata such as `ls_integration="deepagents"` and `lc_agent_name`.
