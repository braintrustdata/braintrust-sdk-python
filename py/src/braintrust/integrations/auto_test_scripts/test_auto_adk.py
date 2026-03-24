"""Test auto_instrument for Google ADK."""

from braintrust.auto import auto_instrument


# 1. Instrument
results = auto_instrument()
assert results.get("adk") == True, "auto_instrument should return True for adk"

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("adk") == True, "auto_instrument should still return True on second call"

# 3. Verify classes are patched using patcher markers

print("SUCCESS")
