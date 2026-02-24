# Golden Tests

These test files validate the Braintrust SDK's integration with different AI providers by running comprehensive test suites that cover various LLM features.

## Test Files

- `genai-py-v1/google_genai.py` - Tests for Google Generative AI integration

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

### Python Tests

```bash
cd genai-py-v1
python google_genai.py
```

## Requirements

Before running the tests, ensure you have the appropriate API keys set as environment variables:

- `BRAINTRUST_API_KEY` to log to braintrust
- `GOOGLE_API_KEY` for Google Generative AI tests

The tests will automatically log traces to Braintrust projects named `golden-python-genai`.

## Contributing

### Adding a New Provider

To add tests for a new AI provider:

1. Use `google_genai.py` as a reference implementation
2. Provide it as context to an LLM and ask it to create an equivalent file for the new provider
3. Ensure all test cases are covered with provider-specific adaptations
4. Follow the naming convention: `<provider>.py`

### Adding New Feature Coverage

When adding a new feature (like reasoning, extended context, or new modalities):

1. Add the test case to existing golden test files
2. Ensure consistency in test structure and naming across providers
3. Update this README to document the new feature coverage

This ensures comprehensive validation across all supported providers and maintains test parity.
