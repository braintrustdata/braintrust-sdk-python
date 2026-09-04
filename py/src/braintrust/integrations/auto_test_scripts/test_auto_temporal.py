"""Test auto_instrument for Temporal."""

from braintrust.integrations.temporal import BraintrustPlugin, setup_temporal
from braintrust.integrations.test_utils import run_auto_smoke


run_auto_smoke("temporal")

assert setup_temporal() == True
assert BraintrustPlugin is not None

print("SUCCESS")
