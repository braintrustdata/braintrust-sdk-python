"""Test auto_instrument for Google ADK."""

import importlib
from importlib.metadata import version as pkg_version

from braintrust.integrations.adk.patchers import (
    AgentRunAsyncPatcher,
    AgentRunPatcher,
    _RunnerRunAsyncSubPatcher,
    _ThreadBridgePlatformSubPatcher,
    _ThreadBridgeRunnersSubPatcher,
)
from braintrust.integrations.test_utils import run_auto_smoke
from google.adk import runners as adk_runners
from google.adk.agents import BaseAgent
from google.adk.runners import Runner


platform_thread = importlib.import_module("google.adk.platform.thread")
base_node = importlib.import_module("google.adk.workflow._base_node") if hasattr(BaseAgent, "run") else None
assert importlib.import_module("google.adk").__name__ == "google.adk"
assert pkg_version("google-adk")


agent_run_target = base_node.BaseNode.run if base_node is not None else BaseAgent.run_async
agent_run_patcher = AgentRunPatcher if base_node is not None else AgentRunAsyncPatcher


def _marker(target, patcher) -> bool:
    return bool(getattr(target, patcher.patch_marker_attr(), False))


def _is_patched() -> bool:
    return (
        _marker(agent_run_target, agent_run_patcher)
        and _marker(Runner.run_async, _RunnerRunAsyncSubPatcher)
        and _marker(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
        and _marker(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)
    )


run_auto_smoke("adk", is_patched=_is_patched)

# Runner.run must stay unpatched even after auto_instrument — only run_async is instrumented.
assert not _marker(Runner.run, _RunnerRunAsyncSubPatcher)

print("SUCCESS")
