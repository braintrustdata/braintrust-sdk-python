"""Test auto_instrument for Google ADK."""

import importlib
from importlib.metadata import version as pkg_version

from braintrust.auto import auto_instrument
from braintrust.integrations.adk.patchers import (
    AgentRunAsyncPatcher,
    _RunnerRunAsyncSubPatcher,
    _RunnerRunSyncSubPatcher,
    _ThreadBridgePlatformSubPatcher,
    _ThreadBridgeRunnersSubPatcher,
)
from google.adk import runners as adk_runners
from google.adk.agents import BaseAgent
from google.adk.runners import Runner


platform_thread = importlib.import_module("google.adk.platform.thread")
assert importlib.import_module("google.adk").__name__ == "google.adk"
assert pkg_version("google-adk")


def is_patched(target, patcher):
    return bool(getattr(target, patcher.patch_marker_attr(), False))


# 1. Verify ADK surfaces are not patched initially.
assert not is_patched(BaseAgent.run_async, AgentRunAsyncPatcher)
assert not is_patched(Runner.run, _RunnerRunSyncSubPatcher)
assert not is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert not is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 2. Instrument.
results = auto_instrument()
assert results.get("adk") == True, "auto_instrument should return True for adk"

# 3. Verify the imported google.adk surfaces are patched.
assert is_patched(BaseAgent.run_async, AgentRunAsyncPatcher)
assert is_patched(Runner.run, _RunnerRunSyncSubPatcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 4. Idempotent.
results2 = auto_instrument()
assert results2.get("adk") == True, "auto_instrument should still return True on second call"
assert is_patched(BaseAgent.run_async, AgentRunAsyncPatcher)
assert is_patched(Runner.run, _RunnerRunSyncSubPatcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)

print("SUCCESS")
