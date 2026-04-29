# Braintrust Python SDK Examples

Each subdirectory in this folder is a self-contained [`uv`](https://docs.astral.sh/uv/) project demonstrating one Braintrust integration or feature. They are designed to be cloned, copied, or run as-is without affecting the rest of the repository.

## Layout

Every example has the same shape:

```
examples/<name>/
├── pyproject.toml   # declares deps + a local `path` source for braintrust
├── README.md        # what it shows + how to run it
└── *.py             # the example script(s)
```

The `pyproject.toml` in each example pins `braintrust` to the local SDK checkout in `py/` via:

```toml
[tool.uv.sources]
braintrust = { path = "../../py", editable = true }
```

That means `uv sync` inside any example installs the version of the SDK currently on disk — edits to `py/src/braintrust/` are picked up immediately.

## Running an example

From the repo root:

```bash
cd examples/<name>
uv sync
uv run python <script>.py
```

You will need a `BRAINTRUST_API_KEY` in your environment, plus whatever provider key the example uses (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). Each example's README lists its requirements.

## The default pattern: `auto_instrument()`

Most provider/framework examples follow the same two-line setup:

```python
import braintrust

braintrust.auto_instrument()
braintrust.init_logger(project="example-<name>")

# now import and use the library normally
```

`braintrust.auto_instrument()` patches every supported integration that's installed in the current environment — there's no need to call individual `setup_*()` or `wrap_*()` helpers. The exceptions are examples that demonstrate non-tracing concerns (`evals/`, `langsmith/`, `otel/`, `temporal/`).

If you want to see `auto_instrument()` light up multiple providers in one trace, the `openai/`, `anthropic/`, and any other provider example can be combined trivially — just install both packages in the same environment, call `auto_instrument()` once, and use both clients.

## Available examples

Unless noted otherwise, every example below uses `braintrust.auto_instrument()`.

| Directory | What it shows |
| --- | --- |
| `adk/` | Google ADK weather agent with one tool |
| `agentscope/` | AgentScope `ReActAgent` against `gpt-4o-mini` |
| `agno/` | Agno agents and teams, sync + async, streaming + non-streaming |
| `anthropic/` | Sync and async Anthropic clients |
| `autogen/` | AutoGen `AssistantAgent` backed by `OpenAIChatCompletionClient` |
| `claude_agent_sdk/` | Claude Agent SDK subprocess query |
| `cohere/` | Cohere `ClientV2` chat call |
| `crewai/` | CrewAI `Agent` + `Task` + `Crew` end-to-end |
| `dspy/` | DSPy `ReAct` agent with two tools (LiteLLM token metrics propagate) |
| `evals/` | The `Eval` framework — does **not** use `auto_instrument()` |
| `google_genai/` | Google GenAI `generate_content` against Gemini |
| `langchain/` | LangChain `prompt | model` chain — global handler installed by `auto_instrument()` |
| `langsmith/` | Migration helper for projects coming from LangSmith — uses `setup_langsmith()` |
| `litellm/` | LiteLLM `completion` |
| `llamaindex/` | LlamaIndex LLM completion |
| `mistral/` | Mistral chat completion |
| `openai/` | OpenAI chat completion plus `@braintrust.traced` for the surrounding function |
| `openai_agents/` | OpenAI Agents SDK `Runner` running an agent |
| `openrouter/` | OpenRouter chat completion routed to OpenAI |
| `otel/` | OpenTelemetry interop — `BraintrustSpanProcessor`, filtering, distributed tracing |
| `pydantic_ai/` | Pydantic AI agent run inside a `start_span` for a permalink |
| `strands/` | Strands `Agent` against `gpt-4o-mini` |
| `temporal/` | Distributed Temporal workflow tracing via `BraintrustPlugin` |

## Why no committed `uv.lock`?

Examples are meant to demonstrate the **current** SDK against **current** provider releases, not snapshot a specific point in time. Committing lockfiles would create constant churn (a new lockfile for every transitive bump on PyPI) and would actively work against the demo goal — someone running an example next quarter wants it to work with whatever's on PyPI today, not a 3-month-old pin. Lockfiles are gitignored.

If you want a reproducible snapshot for your own use, run `uv lock` locally; it just won't be checked in.

## Adding a new example

1. Create `examples/<name>/` with the layout above.
2. In `pyproject.toml`, declare deps under `[project] dependencies` and add the local `braintrust` source under `[tool.uv.sources]` (see any existing example).
3. Write a short README explaining required env vars and how to run it.
4. Verify with `uv sync && uv run python <script>.py` from inside the example dir.
5. The `pylint` nox session in `py/noxfile.py` automatically lints `examples/`, so any imports your example needs must be in the `lint` dependency group of `py/pyproject.toml`.
