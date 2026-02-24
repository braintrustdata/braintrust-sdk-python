# Braintrust SDK

Python client for Braintrust, plus wrapper libraries for OpenAI, Anthropic, and other AI providers.

## Structure

```
├── py/             # Python SDK (see py/CLAUDE.md)
├── integrations/   # Python integrations (adk-py, langchain-py)
├── internal/       # Golden tests
└── scripts/        # Dev scripts
```

## Quick Reference

| Task          | Command       |
| ------------- | ------------- |
| Run all tests | `make test`   |
| Lint/format   | `make fixup`  |
| Python lint   | `make pylint` |

## Setup

```bash
make develop      # Create venv and install deps
source env.sh     # Activate environment
```
