"""Subprocess coverage for Cursor SDK auto-instrumentation and import order."""

# pylint: disable=import-error

import os
import tempfile
from pathlib import Path

import cursor_sdk
from braintrust.auto import auto_instrument
from braintrust.integrations.cursor_sdk._test_vcr import cursor_vcr_config
from braintrust.integrations.test_utils import autoinstrument_test_context
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type


results = auto_instrument()
assert results.get("cursor_sdk") is True
assert auto_instrument().get("cursor_sdk") is True


with tempfile.TemporaryDirectory() as workspace:
    Path(workspace, "README.md").write_text("Cursor auto-instrumentation workspace.\n", encoding="utf-8")
    with autoinstrument_test_context(
        "test_auto_cursor_sdk",
        integration="cursor_sdk",
        vcr_config=cursor_vcr_config(),
    ) as memory_logger:
        with cursor_sdk.CursorClient.launch_bridge(workspace=workspace) as client:
            with client.agents.create(
                model="composer-2.5",
                api_key=os.environ.get("CURSOR_API_KEY", "crsr_test_key_for_cassette_playback"),
                local=cursor_sdk.LocalAgentOptions(cwd=workspace),
            ) as agent:
                result = agent.send("Reply with exactly: cursor tracing complete").wait()

        assert result.status == "finished"
        spans = memory_logger.pop()
        assert find_spans_by_type(spans, SpanTypeAttribute.TASK)
        assert find_spans_by_type(spans, SpanTypeAttribute.LLM)
        assert all(span["context"]["span_origin"]["instrumentation"]["name"] == "cursor-sdk-auto" for span in spans)

print("SUCCESS")
