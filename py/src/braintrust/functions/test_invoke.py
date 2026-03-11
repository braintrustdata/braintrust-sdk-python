"""Tests for the invoke module, particularly init_function."""

import json

import pytest
from braintrust.bt_json import bt_dumps
from braintrust.functions.invoke import init_function
from braintrust.logger import _internal_get_global_state, _internal_reset_global_state


class TestInitFunction:
    """Tests for init_function."""

    def setup_method(self):
        """Reset state before each test."""
        _internal_reset_global_state()

    def teardown_method(self):
        """Clean up after each test."""
        _internal_reset_global_state()

    def test_init_function_disables_span_cache(self):
        """Test that init_function disables the span cache."""
        state = _internal_get_global_state()

        # Cache should be disabled by default (it's only enabled during evals)
        assert state.span_cache.disabled is True

        # Enable the cache (simulating what happens during eval)
        state.span_cache.start()
        assert state.span_cache.disabled is False

        # Call init_function
        f = init_function("test-project", "test-function")

        # Cache should now be disabled (init_function explicitly disables it)
        assert state.span_cache.disabled is True
        assert f.__name__ == "init_function-test-project-test-function-latest"

    def test_init_function_with_version(self):
        """Test that init_function creates a function with the correct name including version."""
        f = init_function("my-project", "my-scorer", version="v1")
        assert f.__name__ == "init_function-my-project-my-scorer-v1"

    def test_init_function_without_version_uses_latest(self):
        """Test that init_function uses 'latest' in name when version not specified."""
        f = init_function("my-project", "my-scorer")
        assert f.__name__ == "init_function-my-project-my-scorer-latest"

    def test_init_function_permanently_disables_cache(self):
        """Test that init_function permanently disables the cache (can't be re-enabled)."""
        state = _internal_get_global_state()

        # Enable the cache
        state.span_cache.start()
        assert state.span_cache.disabled is False

        # Call init_function
        init_function("test-project", "test-function")
        assert state.span_cache.disabled is True

        # Try to start again - should still be disabled because of explicit disable
        state.span_cache.start()
        assert state.span_cache.disabled is True


class TestInvokeSerializationRegression:
    """Regression tests for JSON serialization in invoke (GitHub issue #38)."""

    def test_llm_provider_messages_are_serializable(self):
        provider_messages = []

        try:
            from openai.types.chat import ChatCompletionMessage

            provider_messages.append(ChatCompletionMessage(role="assistant", content="The answer is X."))
        except ImportError:
            print("OpenAI not imported")

        try:
            from anthropic.types import Message, TextBlock, Usage

            provider_messages.append(
                Message(
                    id="msg_123",
                    type="message",
                    role="assistant",
                    content=[TextBlock(type="text", text="The answer is X.")],
                    model="claude-3-5-sonnet-20241022",
                    stop_reason="end_turn",
                    stop_sequence=None,
                    usage=Usage(input_tokens=10, output_tokens=20),
                )
            )
        except ImportError:
            print("Anthropic not imported")

        try:
            from google.genai.types import Content, Part

            provider_messages.append(Content(role="model", parts=[Part(text="The answer is X.")]))
        except ImportError:
            print("Google GenAI not imported")

        if not provider_messages:
            pytest.skip("no supported LLM provider packages available")

        for msg in provider_messages:
            result = bt_dumps(msg)
            assert isinstance(result, str)
            # Verify the output is valid JSON and serialization didn't silently fail
            json.loads(result)
