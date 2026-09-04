"""Test auto_instrument for DSPy.

Note: This test focuses on patching behavior only. Span verification for DSPy
is done in test_dspy.py::test_dspy_callback which uses pytest-vcr (supports httpx).
The standalone VCR in test_utils doesn't capture httpx used by litellm/dspy.
"""

import dspy
from braintrust.integrations.dspy import BraintrustDSpyCallback
from braintrust.integrations.test_utils import run_auto_smoke


def _is_patched() -> bool:
    return bool(getattr(dspy.configure, "__braintrust_patched_dspy_configure__", False))


run_auto_smoke("dspy", is_patched=_is_patched)

# Verify callback is added when configure() is called.
dspy.configure(lm=None)
from dspy.dsp.utils.settings import settings


has_bt_callback = any(isinstance(cb, BraintrustDSpyCallback) for cb in settings.callbacks)
assert has_bt_callback, "Expected BraintrustDSpyCallback in callbacks after configure()"

print("SUCCESS")
