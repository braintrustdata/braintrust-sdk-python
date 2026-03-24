"""Test auto_instrument for LangChain."""

from braintrust.auto import auto_instrument
from braintrust.integrations.langchain import BraintrustCallbackHandler

# 1. Instrument
results = auto_instrument()
assert results.get("langchain") == True, "auto_instrument should return True for langchain"

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("langchain") == True, "auto_instrument should still return True on second call"

# 3. Verify that a global handler was registered with LangChain
from langchain_core.callbacks import CallbackManager

manager = CallbackManager.configure()
handler = next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None)
assert handler is not None, "BraintrustCallbackHandler should be registered globally after auto_instrument()"

print("SUCCESS")
