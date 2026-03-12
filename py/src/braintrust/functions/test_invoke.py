"""Tests for the invoke module, particularly init_function."""

import json
from unittest.mock import MagicMock, patch

import pytest
from braintrust.functions.invoke import init_function, invoke
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


def _invoke_with_messages(messages):
    """Call invoke() with mocked proxy_conn; return the parsed request body."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {}
    mock_conn = MagicMock()
    mock_conn.post.return_value = mock_resp

    with (
        patch("braintrust.functions.invoke.login"),
        patch("braintrust.functions.invoke.get_span_parent_object") as mock_parent,
        patch("braintrust.functions.invoke.proxy_conn", return_value=mock_conn),
    ):
        mock_parent.return_value.export.return_value = "span-export"
        invoke(project_name="test-project", slug="test-fn", messages=messages)

    kwargs = mock_conn.post.call_args.kwargs
    assert "data" in kwargs, "invoke must use data= (bt_dumps) not json= (json.dumps) (see issue 38)"
    assert "json" not in kwargs
    return json.loads(kwargs["data"])


def test_invoke_serializes_openai_messages():
    openai_chat = pytest.importorskip("openai.types.chat")
    msg = openai_chat.ChatCompletionMessage(role="assistant", content="The answer is X.")
    parsed = _invoke_with_messages([msg])
    assert isinstance(parsed, dict) and parsed


def test_invoke_serializes_anthropic_messages():
    anthropic_types = pytest.importorskip("anthropic.types")
    msg = anthropic_types.Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[anthropic_types.TextBlock(type="text", text="The answer is X.")],
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
    )
    parsed = _invoke_with_messages([msg])
    assert isinstance(parsed, dict) and parsed


def test_invoke_serializes_google_messages():
    google_types = pytest.importorskip("google.genai.types")
    msg = google_types.Content(role="model", parts=[google_types.Part(text="The answer is X.")])
    parsed = _invoke_with_messages([msg])
    assert isinstance(parsed, dict) and parsed
