"""Exercise auto setup, independent controls, capture settings and manual overrides."""

import inspect
import os
import sys

from braintrust.auto import auto_instrument


# Keep this subprocess focused on MCP; no optional model instrumentation is needed.
OPTIONS = {
    name: False for name, param in inspect.signature(auto_instrument).parameters.items() if param.default is True
}
ORDER = os.environ["BRAINTRUST_MCP_TEST_IMPORT_ORDER"]
assert auto_instrument(**OPTIONS) == {}
assert "mcp" not in sys.modules and "fastmcp" not in sys.modules


def setup(client=True, server=True, capture=True):
    result = auto_instrument(**{**OPTIONS, "mcp_client": client, "mcp_server": server, "mcp_capture_content": capture})
    assert result["mcp"] is True


if ORDER == "before_import":
    # Exercise the public defaults as well as the explicit controls below.
    assert auto_instrument(**{name: value for name, value in OPTIONS.items() if not name.startswith("mcp_")}) == {
        "mcp": True
    }

import anyio  # noqa: E402
from braintrust.integrations.mcp import _test_harness as harness  # noqa: E402
from braintrust.integrations.mcp import wrap_mcp_client, wrap_mcp_server  # noqa: E402
from braintrust.integrations.test_utils import autoinstrument_test_context  # noqa: E402
from braintrust.logger import start_span  # noqa: E402
from braintrust.test_helpers import find_span_by_name  # noqa: E402


async def main():
    with autoinstrument_test_context("test_auto_mcp", use_vcr=False) as memory_logger:
        # Server runners retain callbacks when they start; automatic server
        # setup belongs before serving. Client setup also covers active sessions.
        modes = {
            "before_import": [(True, True)],
            "after_import": [(True, False), (False, True)],
            "after_connect": [(True, False)],
        }[ORDER]
        for client_enabled, server_enabled in modes:
            server = harness.build_server()
            if ORDER != "after_connect":
                setup(client_enabled, server_enabled)
            async with harness.serve_http(harness.http_app(server)) as url:
                async with harness.connect_http(url) as client:
                    negotiation = [
                        s
                        for s in memory_logger.pop()
                        if s.get("metadata", {}).get("mcp.method.name") in ("initialize", "server/discover")
                    ]
                    if ORDER != "after_connect":
                        assert (
                            any(s["metadata"].get("mcp.span.kind") == "client" for s in negotiation) == client_enabled
                        )
                        assert (
                            any(s["metadata"].get("mcp.span.kind") != "client" for s in negotiation) == server_enabled
                        )
                    # The combined case checks live capture updates; the
                    # independent cases only need to verify content opt-out.
                    for capture in (True, False) if ORDER == "before_import" else (False,):
                        setup(client_enabled, server_enabled, capture)
                        setup(client_enabled, server_enabled, capture)
                        with start_span(name="agent") as agent:
                            await client.call_tool("add", {"a": 2, "b": 3})
                        spans = memory_logger.pop()
                        mcp = [s for s in spans if s["span_attributes"]["name"] == "tools/call add"]
                        outgoing = [s for s in mcp if s["metadata"].get("mcp.span.kind") == "client"]
                        incoming = [s for s in mcp if s not in outgoing]
                        assert len(outgoing) == int(client_enabled), spans
                        assert len(incoming) == int(server_enabled), spans
                        for span in mcp:
                            if capture:
                                assert span["input"] == {"a": 2, "b": 3}
                                assert span["output"]["content"][0]["text"] == "5"
                            else:
                                assert span.get("input") is None and span.get("output") is None
                        if outgoing:
                            assert outgoing[0]["span_parents"] == [agent.span_id]
                        if outgoing and incoming:
                            assert incoming[0]["span_parents"] == [outgoing[0]["span_id"]]
                            assert incoming[0]["root_span_id"] == outgoing[0]["root_span_id"] == agent.root_span_id
                        if incoming:
                            assert find_span_by_name(spans, "nested-work")["span_parents"] == [incoming[0]["span_id"]]

        # Global capture changes leave explicit per-object settings in control.
        if ORDER != "before_import":
            return
        for manual_capture in (True, False):
            setup(capture=not manual_capture)
            server = wrap_mcp_server(harness.build_server(), capture_content=manual_capture)
            async with harness.serve_http(harness.http_app(server)) as url:
                async with harness.connect_http(url) as client:
                    wrap_mcp_client(client.raw, capture_content=manual_capture)
                    setup(capture=not manual_capture)
                    memory_logger.pop()
                    await client.call_tool("add", {"a": 2, "b": 3})
                    spans = memory_logger.pop()
                    tools = [s for s in spans if s["span_attributes"]["name"] == "tools/call add"]
                    assert len(tools) == 2
                    if manual_capture:
                        assert all(s["input"] == {"a": 2, "b": 3} for s in tools)
                    else:
                        assert all(s.get("input") is None and s.get("output") is None for s in tools)


anyio.run(main)
print("SUCCESS")
