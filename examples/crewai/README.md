# CrewAI + Braintrust

Calls `braintrust.auto_instrument()` to register a Braintrust listener on the CrewAI event bus. Running a `Crew` produces the full scope family of spans: `crewai.kickoff` → `crewai.task` → `crewai.agent` → `crewai.llm`.

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python example.py
```
