# Braintrust SDK

[Braintrust](https://www.braintrust.dev/) is a platform for evaluating and shipping AI products. To learn more about Braintrust or sign up for free,
visit our [website](https://www.braintrust.dev/) or check out the [docs](https://www.braintrust.dev/docs).

This repository contains the Python SDK for Braintrust. The SDK includes utilities to:

- Log experiments and datasets to Braintrust
- Run evaluations (via the `Eval` framework)
- Manage an on-premises installation of Braintrust

## Quickstart: Python

Install the library with pip.

```bash
pip install braintrust autoevals
```

Then, create a file named `eval_tutorial.py` with the following code:

```python
from braintrust import Eval
from autoevals import LevenshteinScorer

Eval(
  "Say Hi Bot",
  data=lambda: [
      {
          "input": "Foo",
          "expected": "Hi Foo",
      },
      {
          "input": "Bar",
          "expected": "Hello Bar",
      },
  ],  # Replace with your eval dataset
  task=lambda input: "Hi " + input,  # Replace with your LLM call
  scores=[LevenshteinScorer],
)
```

Then, run the following command:

```bash
BRAINTRUST_API_KEY=<YOUR_API_KEY> \
  braintrust eval eval_tutorial.py
```

## Integrations

Braintrust provides integrations with several popular AI development tools and platforms:

- **LangChain Python**: Integration for logging LangChain Python executions to Braintrust. [Learn more](integrations/langchain-py)

## Documentation

For more information, check out the [docs](https://www.braintrust.dev/docs):

- [Python](https://www.braintrust.dev/docs/reference/sdks/python)
