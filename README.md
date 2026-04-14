# Braintrust Python SDKs

[Braintrust](https://www.braintrust.dev/) is a platform for evaluating and shipping AI products. Learn more at [braintrust.dev](https://www.braintrust.dev/) and in the [docs](https://www.braintrust.dev/docs).

This repository contains Braintrust's Python SDKs and integrations, including:

- The main `braintrust` SDK package in [`./py`](./py)
- Built-in integrations under [`py/src/braintrust/integrations`](py/src/braintrust/integrations) and related compatibility packages under [`./integrations`](./integrations)
- Examples, tests, and local development tooling for Python SDK development

## Quickstart

Install the main SDK and scorer package:

```bash
pip install braintrust autoevals
```

Create `tutorial_eval.py`:

```python
from autoevals import LevenshteinScorer
from braintrust import Eval

Eval(
    "Say Hi Bot",
    data=lambda: [
        {"input": "Foo", "expected": "Hi Foo"},
        {"input": "Bar", "expected": "Hello Bar"},
    ],
    task=lambda input: "Hi " + input,
    scores=[LevenshteinScorer],
)
```

Run it:

```bash
BRAINTRUST_API_KEY=<YOUR_API_KEY> braintrust eval tutorial_eval.py
```

## Packages

| Package | Purpose | PyPI | Docs |
| --- | --- | --- | --- |
| `braintrust` | Core Python SDK for logging, tracing, evals, CLI workflows, and built-in integrations. | [![PyPI - braintrust](https://img.shields.io/pypi/v/braintrust.svg)](https://pypi.org/project/braintrust/) | [py/README.md](py/README.md) |

## Integrations included in `braintrust`

| Integration | Auto-instrumented | Min version |
| --- | --- | --- |
| [OpenAI](py/src/braintrust/oai.py) | Yes | `openai>=1.71` |
| [Anthropic](py/src/braintrust/integrations/anthropic/) | Yes | `anthropic>=0.48.0` |
| [LiteLLM](py/src/braintrust/integrations/litellm/) | Yes | `litellm>=1.74.0` |
| [Pydantic AI](py/src/braintrust/integrations/pydantic_ai/) | Yes | `pydantic_ai>=1.10.0` |
| [Google GenAI](py/src/braintrust/integrations/google_genai/) | Yes | `google-genai>=1.30.0` |
| [Google ADK](py/src/braintrust/integrations/adk/) | Yes | `google-adk>=1.14.1` |
| [LangChain](py/src/braintrust/integrations/langchain/) | Yes | `langchain-core>=0.3.28` |
| [OpenRouter](py/src/braintrust/integrations/openrouter/) | Yes | `openrouter>=0.6.0` |
| [Agno](py/src/braintrust/integrations/agno/) | Yes | `agno>=2.1.0` |
| [AgentScope](py/src/braintrust/integrations/agentscope/) | Yes | `agentscope>=1.0.0` |
| [Claude Agent SDK](py/src/braintrust/integrations/claude_agent_sdk/) | Yes | `claude_agent_sdk>=0.1.10` |
| [DSPy](py/src/braintrust/integrations/dspy/) | Yes | `dspy>=2.6.0` |
| [pytest plugin](py/src/braintrust/wrappers/pytest_plugin/README.md) | No | `pytest>=8` |

## Documentation

- Python SDK docs: https://www.braintrust.dev/docs/reference/sdks/python
- Release notes: https://www.braintrust.dev/docs/reference/release-notes

## License

Apache-2.0
