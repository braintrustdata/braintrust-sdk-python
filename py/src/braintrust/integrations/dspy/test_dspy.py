"""
Tests for DSPy integration with Braintrust.
"""

import dspy
import pytest
from braintrust import logger
from braintrust.integrations.dspy import BraintrustDSpyCallback
from braintrust.integrations.test_utils import run_in_subprocess, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "test-dspy-app"
MODEL = "openai/gpt-4o-mini"


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
    lm = dspy.LM(MODEL)
    dspy.configure(lm=lm, callbacks=[BraintrustDSpyCallback()])

    # Use ChainOfThought for a more interesting test
    cot = dspy.ChainOfThought("question -> answer")
    result = cot(question="What is 2+2?")

    assert result.answer  # Verify we got a response

    # Check logged spans
    spans = memory_logger.pop()
    assert len(spans) >= 4  # Should have module, adapter format, LM, and adapter parse spans

    spans_by_name = {span["span_attributes"]["name"]: span for span in spans}

    lm_span = spans_by_name["dspy.lm"]
    assert "metadata" in lm_span
    assert "model" in lm_span["metadata"]
    assert MODEL in lm_span["metadata"]["model"]
    assert "input" in lm_span
    assert "output" in lm_span

    format_span = spans_by_name["dspy.adapter.format"]
    parse_span = spans_by_name["dspy.adapter.parse"]

    assert format_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert "signature" in format_span["input"]
    assert "demos" in format_span["input"]
    assert "inputs" in format_span["input"]
    assert isinstance(format_span["output"], list)

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

    def test_patch_dspy_patches_configure(self):
        """patch_dspy() should patch dspy.configure via the integration patcher."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy
            result = patch_dspy()
            assert result, "patch_dspy() should return True"
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_patch_dspy_wraps_configure(self):
        """After patch_dspy(), dspy.configure() should auto-add BraintrustDSpyCallback."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy, BraintrustDSpyCallback
            patch_dspy()

            import dspy

            # Configure without explicitly adding callback
            dspy.configure(lm=None)

            # Check that BraintrustDSpyCallback was auto-added
            from dspy.dsp.utils.settings import settings
            callbacks = settings.callbacks
            has_bt_callback = any(isinstance(cb, BraintrustDSpyCallback) for cb in callbacks)
            assert has_bt_callback, f"Expected BraintrustDSpyCallback in {callbacks}"
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

    def test_patch_dspy_idempotent(self):
        """Multiple patch_dspy() calls should be safe."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy
            import dspy

            patch_dspy()
            patch_dspy()  # Second call - should be no-op, not double-wrap

            # Verify configure still works
            lm = dspy.LM("openai/gpt-4o-mini")
            dspy.configure(lm=lm)
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
