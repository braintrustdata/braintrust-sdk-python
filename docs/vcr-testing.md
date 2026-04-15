# VCR And Cassette Testing

This repo uses two recording/replay mechanisms for provider-backed tests:

- [VCR.py](https://github.com/kevin1024/vcrpy) for HTTP interactions
- a Claude Agent SDK subprocess cassette transport for `claude_agent_sdk` tests

Both approaches let CI replay committed request/response traffic without making live provider calls.

## How It Works

HTTP tests decorated with `@pytest.mark.vcr` record requests/responses into YAML cassette files. Subsequent runs replay from the cassette instead of making real API calls.

Claude Agent SDK tests do not use VCR because the SDK communicates with the bundled `claude` subprocess over a JSON stdin/stdout protocol rather than direct HTTP from the test process. Those tests instead record the raw transport conversation into JSON cassette files and replay that transport stream in CI.

### Cassette Locations

| Test suite | Cassette directory |
|---|---|
| Python SDK (wrappers) | `py/src/braintrust/wrappers/cassettes/` |
| Claude Agent SDK subprocess tests | `py/src/braintrust/wrappers/claude_agent_sdk/cassettes/` |
| Langchain integration | `integrations/langchain-py/src/tests/cassettes/` |

## Local Development vs CI

The key difference between local and CI is the cassette record mode.

- VCR uses `CI` / `GITHUB_ACTIONS` to choose `once` locally and `none` in CI.
- Claude Agent SDK subprocess cassettes use `BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE`, defaulting to `once` locally and `none` in CI.

### Local Development

- **Record mode:** `once` -- records a new cassette if one doesn't exist, replays if it does.
- **API keys:** You need real API keys set in your environment to record new cassettes.

```bash
# Run tests (replays cassettes, records missing ones with real keys)
nox -s "test_openai(latest)"

# Record a specific test's cassette from scratch
export OPENAI_API_KEY="sk-..."
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

Claude Agent SDK subprocess cassette examples:

```bash
# Record or replay the real Claude subprocess transport
nox -s "test_claude_agent_sdk(latest)"

# Force re-recording of the subprocess cassette files
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"

# Re-record a focused Claude Agent SDK test
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all \
  nox -s "test_claude_agent_sdk(latest)" -- -k "test_calculator_with_multiple_operations"
```

### CI (GitHub Actions)

- **Record mode:** `none` -- only replays existing cassettes; fails if a cassette is missing.
- **API keys:** Not required. The `conftest.py` fixtures set dummy fallback values:
  - `OPENAI_API_KEY` -> `sk-test-dummy-api-key-for-vcr-tests`
  - `ANTHROPIC_API_KEY` -> `sk-ant-test-dummy-api-key-for-vcr-tests`
  - `GOOGLE_API_KEY` -> `your_google_api_key_here`
- **Claude Agent SDK tests:** Replay committed subprocess cassettes from `py/src/braintrust/wrappers/claude_agent_sdk/cassettes/`.
- **No secrets needed:** CI workflows do not pass `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` as secrets. This means forks and external contributors can run CI without configuring any API key secrets.

## Recording Modes

| Mode | Behavior |
|---|---|
| `once` (local default) | Record if cassette is missing, replay otherwise |
| `new_episodes` | Record new interactions, replay existing ones |
| `all` | Always record, overwriting existing cassettes |
| `none` (CI default) | Replay only, fail if cassette is missing |

Override the mode with `--vcr-record=<mode>`:

```bash
nox -s "test_openai(latest)" -- --vcr-record=all
```

Or disable VCR entirely with `--disable-vcr` (requires real API keys):

```bash
nox -s "test_openai(latest)" -- --disable-vcr
```

For Claude Agent SDK subprocess cassettes, override with `BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE`:

```bash
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all nox -s "test_claude_agent_sdk(latest)"
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=none nox -s "test_claude_agent_sdk(latest)"
```

## Sensitive Data Filtering

Cassettes automatically filter out sensitive headers so API keys are never stored:

- `authorization`
- `x-api-key`, `api-key`
- `openai-api-key`, `openai-organization`
- `x-goog-api-key`
- `x-bt-auth-token`

Claude Agent SDK subprocess cassettes do not store HTTP headers. They store the JSON control protocol exchanged between the SDK and the `claude` subprocess. Re-record them only when the SDK/CLI behavior change is intentional.

## Adding Tests That Need VCR

1. Add `@pytest.mark.vcr` to your test function.
2. Run the test locally with a real API key to record the cassette.
3. Commit the cassette file along with your test.
4. CI will replay the cassette automatically.

```python
@pytest.mark.vcr
def test_my_new_feature(memory_logger):
    client = wrap_openai(openai.OpenAI())
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    spans = memory_logger.pop()
    assert len(spans) == 1
```

## Claude Agent SDK Subprocess Cassettes

Use the subprocess cassette transport for tests that exercise the real `claude_agent_sdk` without mocks.

Guidelines:

1. Keep using the real SDK and bundled CLI.
2. Route the test through `make_cassette_transport(...)`.
3. Record locally with a real Anthropic-capable environment.
4. Commit the generated JSON cassette files under `py/src/braintrust/wrappers/claude_agent_sdk/cassettes/`.

Example workflow:

```bash
cd py
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all \
  nox -s "test_claude_agent_sdk(latest)" -- -k "test_query_async_iterable"
```

## For External Contributors

**When do I need an API key?**

You need a real API key only when the HTTP interactions in a cassette need to change. This happens when you:

- Write a new VCR test (new cassette needed)
- Change a test's API call (different model, prompt, parameters, etc.)
- Delete a cassette and need to re-record it
- Re-record Claude Agent SDK subprocess cassette files

You do **not** need an API key when you:

- Add assertions to existing test responses
- Refactor test logic without changing the API call
- Refactor Claude Agent SDK assertions without changing the subprocess conversation
- Work on non-VCR tests (core SDK, CLI, OTel, etc.)

**Workflow for re-recording cassettes:**

```bash
# 1. Set the real API key for the provider you're testing
export OPENAI_API_KEY="sk-..."

# 2. Re-record the cassette
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_my_changed_test"

# 3. Commit the updated cassette alongside your code changes
git add py/src/braintrust/wrappers/cassettes/test_my_changed_test.yaml
```

For Claude Agent SDK subprocess cassettes:

```bash
# 1. Ensure your Anthropic credentials work with the bundled Claude CLI
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Re-record the subprocess cassette
cd py
BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE=all \
  nox -s "test_claude_agent_sdk(latest)" -- -k "test_calculator_with_multiple_operations"

# 3. Commit the JSON cassette
git add py/src/braintrust/wrappers/claude_agent_sdk/cassettes/
```

**CI will work without secrets.** Forks do not need to configure provider API secrets for replayed cassette runs. Just make sure any new or modified YAML/JSON cassette files are committed.
