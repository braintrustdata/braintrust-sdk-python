"""Test auto_instrument for Google ADK."""

import asyncio
import importlib
import os
from importlib.metadata import version as pkg_version

from braintrust.auto import auto_instrument
from braintrust.conftest import get_vcr_config
from braintrust.integrations.adk.patchers import (
    AgentRunAsyncPatcher,
    _RunnerRunAsyncSubPatcher,
    _ThreadBridgePlatformSubPatcher,
    _ThreadBridgeRunnersSubPatcher,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, autoinstrument_test_context
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
assert not is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert not is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 2. Instrument.
results = auto_instrument()
assert results.get("adk") == True, "auto_instrument should return True for adk"

# 3. Verify the imported google.adk surfaces are patched.
assert is_patched(BaseAgent.run_async, AgentRunAsyncPatcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(Runner.run, _RunnerRunAsyncSubPatcher)
assert is_patched(platform_thread.create_thread, _ThreadBridgePlatformSubPatcher)
assert is_patched(adk_runners.create_thread, _ThreadBridgeRunnersSubPatcher)

# 4. Idempotent.
results2 = auto_instrument()
assert results2.get("adk") == True, "auto_instrument should still return True on second call"
assert is_patched(BaseAgent.run_async, AgentRunAsyncPatcher)
assert is_patched(Runner.run_async, _RunnerRunAsyncSubPatcher)
assert not is_patched(Runner.run, _RunnerRunAsyncSubPatcher)
assert results2.get("google_genai") == True


# 5. With ADK and Google GenAI both instrumented, a Gemini request emits one
# usage-bearing ``llm`` span.
async def run_agent():
    from google.adk import Agent
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    agent = Agent(
        name="metrics_agent",
        model=(
            "gemini-2.5-flash-lite"
            if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") in {"latest", "2.6.3"}
            else "gemini-2.0-flash"
        ),
        instruction="You are a helpful assistant.",
    )
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="metrics_app", user_id="test-user", session_id="test-session-metrics"
    )
    runner = Runner(agent=agent, app_name="metrics_app", session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text="Say hello in 3 words")])
    async for _ in runner.run_async(user_id="test-user", session_id="test-session-metrics", new_message=message):
        pass


os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")


def uppercase_method(request):
    # google-genai sends lowercase HTTP methods; cassettes record them uppercased.
    request.method = request.method.upper()
    return request


vcr_config = {**get_vcr_config(), "before_record_request": uppercase_method}
with autoinstrument_test_context(
    "test_adk_captures_metrics", integration="adk", vcr_config=vcr_config
) as memory_logger:
    asyncio.run(run_agent())
    spans = memory_logger.pop()

usage_spans = [span for span in spans if "prompt_tokens" in span.get("metrics", {})]
assert [span["span_attributes"]["name"] for span in usage_spans] == ["generate_content"], usage_spans
(llm_span,) = usage_spans
assert llm_span["span_attributes"]["type"] == "llm"
assert llm_span["context"]["span_origin"]["instrumentation"]["name"] == "google-genai-auto"
assert_metrics_are_valid(llm_span["metrics"])

(adk_model_span,) = [span for span in spans if span["span_attributes"]["name"].startswith("llm_call")]
assert adk_model_span["span_attributes"]["type"] == "task"
assert llm_span["span_parents"] == [adk_model_span["span_id"]]
assert [span for span in spans if span["span_attributes"]["type"] == "llm"] == [llm_span]

print("SUCCESS")
