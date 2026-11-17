"""Antigravity setup is idempotent across both provider import orders."""

import asyncio
import importlib
import os
import sys
import tempfile

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


provider = None
if os.environ.get("BRAINTRUST_ANTIGRAVITY_IMPORT_ORDER") == "before":
    provider = importlib.import_module("google.antigravity")
else:
    assert "google.antigravity" not in sys.modules


async def main():
    assert "antigravity" not in auto_instrument(antigravity=False)
    assert auto_instrument()["antigravity"] is True
    assert auto_instrument()["antigravity"] is True
    sdk = provider or importlib.import_module("google.antigravity")
    from braintrust.integrations.antigravity._test_transport import cassette_transport

    with tempfile.TemporaryDirectory() as workspace:
        with autoinstrument_test_context("chat", integration="antigravity", use_vcr=False) as memory_logger:
            with cassette_transport("chat"):
                config = sdk.LocalAgentConfig(model="gemini-3.8-flash", app_data_dir=workspace).lightweight()
                async with sdk.Agent(config) as agent:
                    response = await agent.chat("Reply with exactly: antigravity tracing complete")
                    assert "antigravity tracing complete" in (await response.text()).lower()
            spans = memory_logger.pop()
            assert len([s for s in spans if s.get("span_attributes", {}).get("name") == "antigravity.chat"]) == 1
            assert all(s["context"]["span_origin"]["instrumentation"]["name"] == "antigravity-auto" for s in spans)


asyncio.run(main())
print("SUCCESS")
