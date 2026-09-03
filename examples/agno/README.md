# Agno + Braintrust

Each script calls `braintrust.auto_instrument()` before importing `agno`, so all subsequent agent/team activity is traced. The agents use `YFinanceTools` so they actually exercise tool calls.

| Script | Shape |
| --- | --- |
| `simple_agent.py` | one agent, one question |
| `simple_agent_stream.py` | one agent, streamed |
| `async_simple_agent_stream.py` | one agent, async + streamed |
| `team_agent.py` | research + advisor team, sync |
| `async_team_agent.py` | research + advisor team, async + streamed |
| `accuracy_eval.py` | `AccuracyEval` over one agent, scored in Braintrust |
| `eval_suite.py` | an `agno.eval` suite; each `Case` becomes an experiment row |

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python simple_agent.py
```
