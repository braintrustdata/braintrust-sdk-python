# braintrust-langchain (DEPRECATED)

[![PyPI version](https://img.shields.io/pypi/v/braintrust-langchain.svg)](https://pypi.org/project/braintrust-langchain/)

SDK for integrating [Braintrust](https://braintrust.dev) with [LangChain](https://langchain.com/). This package provides a callback handler to automatically log LangChain executions to Braintrust.

> **This package is deprecated.** The LangChain integration is now included in the main [`braintrust`](https://pypi.org/project/braintrust/) package.

## Migration

1. Remove `braintrust-langchain` from your dependencies
2. Install or upgrade `braintrust`:
   ```bash
   pip install --upgrade braintrust
   ```
3. Update your imports:
   ```python
   # Before
   from braintrust_langchain import BraintrustCallbackHandler, set_global_handler

   # After (option 1: same explicit callback API from the main package)
   from braintrust.integrations.langchain import BraintrustCallbackHandler, set_global_handler

   # After (option 2: helper that initializes Braintrust and installs the global handler)
   from braintrust.integrations.langchain import setup_langchain

   setup_langchain(project_name="your-project-name")
   ```

The callback API is the same - no code changes are needed beyond the package/import path.

---

## Installation

```bash
pip install braintrust-langchain
```

## Requirements

- Python >= 3.10
- LangChain >= 0.3.27

## Quickstart

First, make sure you have your Braintrust API key set in your environment:

```bash
export BRAINTRUST_API_KEY="your-api-key"
```

```python
import asyncio
from braintrust import init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


async def main():
    # Initialize the logger with your project
    init_logger(project="your-project-name")

    # Create the callback handler and set it globally for all LangChain components
    handler = BraintrustCallbackHandler()
    set_global_handler(handler)

    # Initialize your LangChain components
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    model = ChatOpenAI()

    # Create a simple chain
    chain = prompt | model

    # Use LangChain as normal - all calls will be logged to Braintrust
    response = await chain.ainvoke({"number": "2"})


if __name__ == "__main__":
    asyncio.run(main())
```

## Passing Handlers Explicitly

If you'd prefer to pass the callback handler to specific LangChain calls instead of setting it globally, you can do so using the `callbacks` config option:

```python
async def main():
    handler = BraintrustCallbackHandler()

    # Pass the handler to specific calls
    response = await chain.ainvoke(
        {"number": "2"},
        config={"callbacks": [handler]}
    )

    # Or initialize components with the handler
    model = ChatOpenAI(callbacks=[handler])
```

### Supported Features

The callback handler supports logging for:

- LLM calls (including streaming)
- Chat model interactions
- Chain executions
- Tool/Agent usage
- Memory operations
- State management (LangGraph)

Review the [LangChain documentation](https://python.langchain.com/docs/modules/callbacks/) for more information on how to use callbacks.

## Documentation

- Braintrust docs: https://www.braintrust.dev/docs
- Braintrust Python SDK docs: https://www.braintrust.dev/docs/reference/sdks/python
- LangChain callback docs: https://python.langchain.com/docs/modules/callbacks/
