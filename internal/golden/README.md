# Golden Tests

These test files validate the Braintrust SDK's integration with different AI providers by running comprehensive test suites that cover various LLM features.

## Test Files

Each test suite validates:

- Basic and multi-turn completions
- System prompts
- Streaming responses
- Image and document inputs
- Temperature and sampling parameters
- Stop sequences and metadata
- Tool use and function calling
- Mixed content types

## Running Tests

Run a specific golden suite from its directory, for example:

```bash
cd langchain-py-v1
python langchain.py
```

```bash
cd pydantic-ai-v1
python pydantic_ai_test.py
```


## Requirements

Before running a suite, ensure you have the appropriate API keys set as environment variables for that provider, along with `BRAINTRUST_API_KEY` if you want to log traces to Braintrust.

## Contributing

### Adding a New Provider

To add tests for a new AI provider:

1. Use an existing golden suite as a reference implementation
2. Ensure all test cases are covered with provider-specific adaptations
3. Follow the naming convention already used by the surrounding suites

### Adding New Feature Coverage

When adding a new feature (like reasoning, extended context, or new modalities):

1. Add the test case to existing golden test files
2. Ensure consistency in test structure and naming across providers
3. Update this README to document the new feature coverage

This helps maintain broad feature coverage across the remaining golden suites.
