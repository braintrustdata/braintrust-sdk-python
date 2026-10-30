"""Coexistence with real framework MCP adapters and upstream OTel hooks."""

# These optional packages and the MCP 1.x module are installed by test_mcp_interop,
# rather than the shared lint environment (which uses MCP 2.x).
# pylint: disable=import-error,no-name-in-module

import pytest
from braintrust import logger
from braintrust.integrations.mcp import _test_harness as harness
from braintrust.integrations.mcp import wrap_mcp_client, wrap_mcp_server
from braintrust.test_helpers import find_span_by_name, init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-mcp-interop")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.asyncio
async def test_framework_mcp_tool(memory_logger):
    from braintrust.integrations.langchain import BraintrustCallbackHandler
    from langchain_mcp_adapters.tools import load_mcp_tools

    server = wrap_mcp_server(harness.build_server())
    async with harness.connect_in_memory(server) as client:
        session = wrap_mcp_client(client.raw)
        tools = await load_mcp_tools(session)
        add = next(tool for tool in tools if tool.name == "add")
        await add.ainvoke({"a": 2, "b": 3}, config={"callbacks": [BraintrustCallbackHandler()]})
    spans = memory_logger.pop()
    framework = [s for s in spans if s["context"]["span_origin"]["instrumentation"]["name"] == "langchain-auto"]
    assert len(framework) == 1
    outgoing = [s for s in spans if s.get("metadata", {}).get("mcp.span.kind") == "client"]
    client_tool = find_span_by_name(outgoing, "tools/call add")
    assert client_tool["span_parents"] == [framework[0]["span_id"]]
    server_tool = find_span_by_name([s for s in spans if s not in outgoing], "tools/call add")
    assert server_tool["span_parents"] == [client_tool["span_id"]]
    assert find_span_by_name(spans, "nested-work")["span_parents"] == [server_tool["span_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["braintrust_first", "otel_first"])
async def test_upstream_otel_owns_client_operations(memory_logger, order):
    from mcp.shared.session import BaseSession
    from opentelemetry.instrumentation.mcp import McpInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = McpInstrumentor()
    original_send = BaseSession.send_request
    try:
        async with harness.connect_in_memory(harness.build_server()) as client:
            if order == "braintrust_first":
                wrap_mcp_client(client.raw)
            instrumentor.instrument(tracer_provider=provider)
            if order == "otel_first":
                wrap_mcp_client(client.raw)
            await client.call_tool("add", {"a": 1, "b": 2})
        spans = memory_logger.pop()
        assert not [s for s in spans if s.get("metadata", {}).get("mcp.span.kind") == "client"]
        tools = [s for s in exporter.get_finished_spans() if s.name == "add.tool"]
        assert len(tools) == 1
    finally:
        instrumentor.uninstrument()
        # 0.62.4 leaves its BaseSession hook installed on uninstrument().
        BaseSession.send_request = original_send
        provider.shutdown()
