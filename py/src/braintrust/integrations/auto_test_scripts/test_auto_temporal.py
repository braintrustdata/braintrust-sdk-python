"""Test auto_instrument for Temporal."""

from braintrust.auto import auto_instrument
from braintrust.integrations.temporal import BraintrustPlugin, setup_temporal


results = auto_instrument()
assert results.get("temporal") == True

results2 = auto_instrument()
assert results2.get("temporal") == True

assert setup_temporal() == True
assert BraintrustPlugin is not None

print("SUCCESS")
