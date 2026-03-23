"""Test auto_instrument patch selection for Anthropic."""

import anthropic
from braintrust.auto import auto_instrument
from braintrust.integrations import IntegrationPatchConfig


results = auto_instrument(
    anthropic=IntegrationPatchConfig(
        enabled_patchers={"anthropic.init.sync"},
    )
)
assert results.get("anthropic") == True

patched_sync = anthropic.Anthropic(api_key="test-key")
unpatched_async = anthropic.AsyncAnthropic(api_key="test-key")
assert type(patched_sync.messages).__module__ == "braintrust.integrations.anthropic.tracing"
assert type(unpatched_async.messages).__module__.startswith("anthropic.")

print("SUCCESS")
