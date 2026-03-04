# VCR Testing

This repo uses [VCR.py](https://github.com/kevin1024/vcrpy) to record and replay HTTP interactions in tests. This lets the test suite run without real API keys or network access in most cases.

## How It Works

Tests decorated with `@pytest.mark.vcr` record HTTP requests/responses into YAML "cassette" files on first run. Subsequent runs replay from the cassette instead of making real API calls.

### Cassette Locations

| Test suite | Cassette directory |
|---|---|
| Python SDK (wrappers) | `py/src/braintrust/wrappers/cassettes/` |
| Langchain integration | `integrations/langchain-py/src/tests/cassettes/` |

## Local Development vs CI

The key difference between local and CI is the VCR **record mode**, controlled by the `CI` / `GITHUB_ACTIONS` environment variables.

### Local Development

- **Record mode:** `once` -- records a new cassette if one doesn't exist, replays if it does.
- **API keys:** You need real API keys set in your environment to record new cassettes.
- **`test_latest_wrappers_novcr` session:** Runs normally, making real API calls (no VCR).

```bash
# Run tests (replays cassettes, records missing ones with real keys)
nox -s "test_openai(latest)"

# Record a specific test's cassette from scratch
export OPENAI_API_KEY="sk-..."
nox -s "test_openai(latest)" -- --vcr-record=all -k "test_openai_chat_metrics"
```

### CI (GitHub Actions)

- **Record mode:** `none` -- only replays existing cassettes; fails if a cassette is missing.
- **API keys:** Not required. The `conftest.py` fixtures set dummy fallback values:
  - `OPENAI_API_KEY` -> `sk-test-dummy-api-key-for-vcr-tests`
  - `ANTHROPIC_API_KEY` -> `sk-ant-test-dummy-api-key-for-vcr-tests`
  - `GOOGLE_API_KEY` -> `your_google_api_key_here`
- **`test_latest_wrappers_novcr` session:** Automatically skipped in CI since it disables VCR and would need real keys.
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

## Sensitive Data Filtering

Cassettes automatically filter out sensitive headers so API keys are never stored:

- `authorization`
- `x-api-key`, `api-key`
- `openai-api-key`, `openai-organization`
- `x-goog-api-key`
- `x-bt-auth-token`

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

## For External Contributors

**When do I need an API key?**

You need a real API key only when the HTTP interactions in a cassette need to change. This happens when you:

- Write a new VCR test (new cassette needed)
- Change a test's API call (different model, prompt, parameters, etc.)
- Delete a cassette and need to re-record it

You do **not** need an API key when you:

- Add assertions to existing test responses
- Refactor test logic without changing the API call
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

**CI will work without secrets.** Forks do not need to configure any API key secrets — CI replays from committed cassettes using dummy keys. Just make sure any new or modified cassettes are committed.
