"""Test auto_instrument for Google ADK."""

import importlib
from importlib.metadata import version as pkg_version

from braintrust.auto import auto_instrument
from braintrust.integrations.adk.patchers import (
    AgentRunAsyncPatcher,
    AgentRunPatcher,
    _RunnerRunAsyncSubPatcher,
    _ThreadBridgePlatformSubPatcher,
    _ThreadBridgeRunnersSubPatcher,
)
from google.adk import runners as adk_runners
from google.adk.agents import BaseAgent
from google.adk.runners import Runner


platform_thread = importlib.import_module("google.adk.platform.thread")
base_node = importlib.import_module("google.adk.workflow._base_node") if hasattr(BaseAgent, "run") else None
assert importlib.import_module("google.adk").__name__ == "google.adk"
assert pkg_version("google-adk")


def is_patched(target, patcher):
    return bool(getattr(target, patcher.patch_marker_attr(), False))


# 1. Verify ADK surfaces are not patched initially.
agent_run_target = base_node.BaseNode.run if base_node is not None else BaseAgent.run_async
agent_run_patcher = AgentRunPatcher if base_node is not None else AgentRunAsyncPatcher
assert not is_patched(agent_run_target, agent_run_patcher)
assert not is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert not is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 2. Instrument.
results = auto_instrument()
assert results.get("adk") == True, "auto_instrument should return True for adk"

# 3. Verify the imported google.adk surfaces are patched.
assert is_patched(agent_run_target, agent_run_patcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(Runner.run, _RunnerRunAsyncSubPatcher)
assert is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 4. Idempotent.
results2 = auto_instrument()
assert results2.get("adk") == True, "auto_instrument should still return True on second call"
assert is_patched(agent_run_target, agent_run_patcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(Runner.run, _RunnerRunAsyncSubPatcher)

print("SUCCESS")
