"""
Tests for DSPy integration with Braintrust.
"""

import inspect
import os
from types import SimpleNamespace

import dspy
import pytest
from braintrust import logger
from braintrust.integrations.dspy import BraintrustDSpyCallback
from braintrust.integrations.test_utils import run_in_subprocess, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "test-dspy-app"
MODEL = "openai/gpt-4o-mini"
# DSPy >= 3.4 defaults to engine="auto", which routes OpenAI calls through the
# vendored lm15 raw-socket transport that VCR cannot intercept. Pin LiteLLM so
# the request is recorded.
LM_KWARGS = {"engine": "litellm"} if "engine" in inspect.signature(dspy.LM.__init__).parameters else {}


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.vcr
def test_dspy_callback(memory_logger):
    """Test DSPy callback logs spans correctly."""
    assert not memory_logger.pop()

    # Configure DSPy with Braintrust callback
    lm = dspy.LM(MODEL, cache=False, **LM_KWARGS)
    dspy.configure(lm=lm, callbacks=[BraintrustDSpyCallback()])

    # Use ChainOfThought for a more interesting test
    cot = dspy.ChainOfThought("question -> answer")
    result = cot(question="What is 2+2?")

    assert result.answer  # Verify we got a response

    # Check logged spans
    spans = memory_logger.pop()
    assert len(spans) >= 4  # Should have module, adapter format, LM, and adapter parse spans

    spans_by_name = {span["span_attributes"]["name"]: span for span in spans}

    module_span = spans_by_name["dspy.module.ChainOfThought"]
    assert module_span["span_attributes"]["type"] == "task"
    assert module_span["metadata"]["module_class"].endswith("ChainOfThought")

    lm_span = spans_by_name["dspy.lm"]
    assert lm_span["context"]["span_origin"]["instrumentation"]["name"] == "dspy-auto"
    assert lm_span["span_attributes"].get("type") != "llm"
    assert "metadata" in lm_span
    assert "model" in lm_span["metadata"]
    assert MODEL in lm_span["metadata"]["model"]
    assert lm_span["metadata"]["provider"] == "openai"
    assert "input" in lm_span
    assert "output" in lm_span
    # The latest cassette contains a real OpenAI usage payload. DSPy 2.6's
    # recorded response predates usage being included.
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest":
        assert lm_span["metrics"]["prompt_tokens"] == 170
        assert lm_span["metrics"]["completion_tokens"] == 53
        assert lm_span["metrics"]["tokens"] == 223

    format_span = spans_by_name["dspy.adapter.format"]
    assert format_span["span_attributes"]["type"] == "task"
    assert format_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert "signature" in format_span["input"]
    assert "demos" in format_span["input"]
    assert "inputs" in format_span["input"]
    assert isinstance(format_span["output"], list)

    parse_span = spans_by_name["dspy.adapter.parse"]
    assert parse_span["span_attributes"]["type"] == "task"
    assert parse_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert "signature" in parse_span["input"]
    assert "completion" in parse_span["input"]
    assert isinstance(parse_span["output"], dict)

    # Verify spans are nested under the broader DSPy execution
    span_ids = {span["span_id"] for span in spans}
    assert lm_span.get("span_parents")
    assert format_span.get("span_parents")
    assert parse_span.get("span_parents")
    assert format_span["span_parents"][0] in span_ids
    assert lm_span["span_parents"][0] in span_ids
    assert parse_span["span_parents"][0] in span_ids


def test_dspy_callback_ignores_usage_from_previous_lm_call(memory_logger):
    """A failed call must not inherit usage from an older history entry."""
    instance = SimpleNamespace(
        model=MODEL,
        history=[
            {
                "prompt": "previous request",
                "outputs": ["previous response"],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            }
        ],
    )
    callback = BraintrustDSpyCallback()
    callback.on_lm_start("failed-call", instance, {"prompt": "new request"})
    callback.on_lm_end("failed-call", None, RuntimeError("provider failed"))

    span = next(span for span in memory_logger.pop() if span["span_attributes"]["name"] == "dspy.lm")
    assert not {"prompt_tokens", "completion_tokens", "tokens"} & span.get("metrics", {}).keys()


def test_dspy_callback_correlates_concurrent_lm_usage(memory_logger):
    """Concurrent calls on one LM must read their own new history entries."""
    instance = SimpleNamespace(model=MODEL, history=[])
    callback = BraintrustDSpyCallback()
    callback.on_lm_start("call-a", instance, {"prompt": "same request"})
    callback.on_lm_start("call-b", instance, {"prompt": "same request"})

    outputs_a = ["same response"]
    outputs_b = ["same response"]
    entry_a = {
        "prompt": "same request",
        "outputs": outputs_a,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    entry_b = {
        "prompt": "same request",
        "outputs": outputs_b,
        "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
    }
    # Both provider calls can append history before either end callback runs.
    instance.history.extend([entry_a, entry_b])
    callback.on_lm_end("call-a", outputs_a)
    callback.on_lm_end("call-b", outputs_b)

    spans = [span for span in memory_logger.pop() if span["span_attributes"]["name"] == "dspy.lm"]
    assert [span["metrics"]["tokens"] for span in spans] == [12, 35]


def test_dspy_adapter_callbacks(memory_logger):
    """Adapter format/parse callbacks should log spans without an LM call."""
    assert not memory_logger.pop()

    dspy.configure(callbacks=[BraintrustDSpyCallback()])

    signature = dspy.make_signature("question -> answer")
    adapter = dspy.ChatAdapter()
    formatted = adapter.format(
        signature,
        demos=[{"question": "1+1", "answer": "2"}],
        inputs={"question": "2+2"},
    )
    parsed = adapter.parse(signature, "[[ ## answer ## ]]\n4")

    assert formatted
    assert parsed == {"answer": "4"}

    spans = memory_logger.pop()
    assert len(spans) == 2

    spans_by_name = {span["span_attributes"]["name"]: span for span in spans}
    format_span = spans_by_name["dspy.adapter.format"]
    parse_span = spans_by_name["dspy.adapter.parse"]

    assert format_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert format_span["input"]["inputs"] == {"question": "2+2"}
    assert format_span["output"] == formatted

    assert parse_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert parse_span["input"]["completion"] == "[[ ## answer ## ]]\n4"
    assert parse_span["output"] == parsed


class TestPatchDSPy:
    """Tests for patch_dspy()."""

    def test_patch_dspy_wraps_configure(self):
        """After patch_dspy(), dspy.configure() should auto-add BraintrustDSpyCallback."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy, BraintrustDSpyCallback
            assert patch_dspy(), "patch_dspy() should return True"
            assert patch_dspy(), "second patch_dspy() should be a no-op that still returns True"

            import dspy

            # Configure without explicitly adding callback
            dspy.configure(lm=None)

            # Check that exactly one BraintrustDSpyCallback was auto-added (no double-wrap)
            from dspy.dsp.utils.settings import settings
            callbacks = settings.callbacks
            bt_callbacks = [cb for cb in callbacks if isinstance(cb, BraintrustDSpyCallback)]
            assert len(bt_callbacks) == 1, f"Expected one BraintrustDSpyCallback in {callbacks}"
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_patch_dspy_preserves_existing_callbacks(self):
        """patch_dspy() should preserve user-provided callbacks."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy, BraintrustDSpyCallback
            patch_dspy()

            import dspy
            from dspy.utils.callback import BaseCallback

            class MyCallback(BaseCallback):
                pass

            my_callback = MyCallback()
            dspy.configure(lm=None, callbacks=[my_callback])

            from dspy.dsp.utils.settings import settings
            callbacks = settings.callbacks

            # Should have both callbacks
            has_my_callback = any(cb is my_callback for cb in callbacks)
            has_bt_callback = any(isinstance(cb, BraintrustDSpyCallback) for cb in callbacks)

            assert has_my_callback, "User callback should be preserved"
            assert has_bt_callback, "BraintrustDSpyCallback should be added"
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_patch_dspy_does_not_duplicate_callback(self):
        """patch_dspy() should not add duplicate BraintrustDSpyCallback."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy, BraintrustDSpyCallback
            patch_dspy()

            import dspy

            # User explicitly adds BraintrustDSpyCallback
            bt_callback = BraintrustDSpyCallback()
            dspy.configure(lm=None, callbacks=[bt_callback])

            from dspy.dsp.utils.settings import settings
            callbacks = settings.callbacks

            # Should only have one BraintrustDSpyCallback
            bt_callbacks = [cb for cb in callbacks if isinstance(cb, BraintrustDSpyCallback)]
            assert len(bt_callbacks) == 1, f"Expected 1 BraintrustDSpyCallback, got {len(bt_callbacks)}"
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_legacy_wrapper_import_still_works(self):
        """The old braintrust.wrappers.dspy import path should still work."""
        result = run_in_subprocess("""
            from braintrust.wrappers.dspy import BraintrustDSpyCallback, patch_dspy
            assert BraintrustDSpyCallback is not None
            assert callable(patch_dspy)
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout


class TestAutoInstrumentDSPy:
    """Tests for auto_instrument() with DSPy."""

    def test_auto_instrument_dspy(self):
        """Test auto_instrument patches DSPy, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_dspy.py")
