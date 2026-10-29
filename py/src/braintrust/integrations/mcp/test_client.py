"""Real MCP client operations, independently and connected to server traces."""

import asyncio
import json
import re

import anyio
import pytest
from braintrust import logger
from braintrust.integrations.mcp import _test_harness as harness
from braintrust.integrations.mcp import wrap_mcp_client, wrap_mcp_server
from braintrust.logger import current_span, start_span
from braintrust.test_helpers import find_span_by_name, init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-mcp-client")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def client_spans(spans):
    return [s for s in spans if s.get("metadata", {}).get("mcp.span.kind") == "client"]


def assert_client(span, method):
    assert span["metadata"]["mcp.method.name"] == method
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "mcp-auto"
    assert span["metrics"]["end"] >= span["metrics"]["start"]


@pytest.mark.asyncio
async def test_client_rpc_error(memory_logger):
    async with harness.connect_in_memory(harness.build_server()) as client:
        wrap_mcp_client(client.raw, capture_content=False)
        with pytest.raises(Exception):
            await client.get_prompt("missing", {})
    spans = memory_logger.pop()
    # Client wrapping must not enable instrumentation on the unwrapped server.
    assert len(spans) == 1
    span = find_span_by_name(spans, "prompts/get missing")
    assert span["metadata"]["mcp.span.kind"] == "client"
    assert_client(span, "prompts/get")
    assert span["error"]
    assert span["metadata"]["error.type"] == span["metadata"]["rpc.response.status_code"]
    assert span.get("input") is None and span.get("output") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        None,
        pytest.param(
            "auto",
            marks=pytest.mark.skipif(
                harness.FLAVOR != "mcp" or not harness.MCP_V2, reason="Modern official MCP 2.x transport"
            ),
        ),
    ],
)
async def test_concurrent_connected_requests(memory_logger, mode):
    server = wrap_mcp_server(harness.build_server())
    async with harness.connect_in_memory(server, mode=mode) as client:
        wrap_mcp_client(client.raw)
        # Wrapping the high-level client and its session must still create one span.
        if hasattr(client.raw, "session"):
            wrap_mcp_client(client.raw.session)
        with start_span(name="agent") as agent:
            await asyncio.gather(*(client.call_tool("add", {"a": i, "b": 1}) for i in range(8)))
    spans = memory_logger.pop()
    outgoing = [s for s in client_spans(spans) if s["metadata"]["mcp.method.name"] == "tools/call"]
    assert len(outgoing) == 8
    by_input = {s["input"]["a"]: s for s in outgoing}
    incoming = [s for s in spans if s["span_attributes"]["name"] == "tools/call add" and s not in outgoing]
    assert len(incoming) == 8
    for span in incoming:
        parent = by_input[span["input"]["a"]]
        assert parent["span_parents"] == [agent.span_id]
        assert span["span_parents"] == [parent["span_id"]]
        assert span["root_span_id"] == parent["root_span_id"] == agent.root_span_id
    by_id = {s["span_id"]: s for s in incoming}
    nested = [s for s in spans if s["span_attributes"]["name"] == "nested-work"]
    assert len(nested) == 8
    for span in nested:
        (parent_id,) = span["span_parents"]
        assert span["input"] == by_id[parent_id]["input"]


@pytest.mark.asyncio
async def test_connected_http(memory_logger):
    server = wrap_mcp_server(harness.build_server(), capture_content=False)
    async with harness.serve_http(harness.http_app(server)) as url:
        async with harness.connect_http(url) as client:
            wrap_mcp_client(client.raw, capture_content=False)
            with start_span(name="agent") as agent:
                result = await client.call_tool("add", {"a": 2, "b": 3})
                assert result.content[0].text == "5"
    spans = memory_logger.pop()
    outgoing = find_span_by_name(client_spans(spans), "tools/call add")
    incoming = find_span_by_name([s for s in spans if s not in client_spans(spans)], "tools/call add")
    nested = find_span_by_name(spans, "nested-work")
    assert outgoing["span_parents"] == [agent.span_id]
    assert incoming["span_parents"] == [outgoing["span_id"]]
    assert nested["span_parents"] == [incoming["span_id"]]
    assert incoming["root_span_id"] == outgoing["root_span_id"] == agent.root_span_id
    for span in (outgoing, incoming):
        assert span.get("input") is None and span.get("output") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("server_traced", [False, True])
async def test_connected_stdio(memory_logger, tmp_path, server_traced):
    spans_file = tmp_path / "server-spans.json"
    env = {
        harness.SPANS_FILE_ENV: str(spans_file),
        harness.WRAP_ENV: "wrap" if server_traced else "none",
    }
    async with harness.connect_stdio(env) as client:
        wrap_mcp_client(client.raw)
        with start_span(name="agent") as agent:
            await client.call_tool("add", {"a": 2, "b": 3})
    with anyio.fail_after(10):
        while not spans_file.exists():
            await anyio.sleep(0.05)
    remote = json.loads(spans_file.read_text())
    outgoing = find_span_by_name(client_spans(memory_logger.pop()), "tools/call add")
    assert outgoing["span_parents"] == [agent.span_id]
    assert outgoing["root_span_id"] == agent.root_span_id
    assert outgoing["output"]["content"][0]["text"] == "5"
    if server_traced:
        incoming = find_span_by_name(remote, "tools/call add")
        assert incoming["span_parents"] == [outgoing["span_id"]]
        assert incoming["root_span_id"] == outgoing["root_span_id"]
        assert find_span_by_name(remote, "nested-work")["span_parents"] == [incoming["span_id"]]
    else:
        assert not [s for s in remote if "mcp.method.name" in s.get("metadata", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_content", [True, False])
async def test_client_http_operations(memory_logger, capture_content):
    request_meta = []
    server = harness.build_server(request_meta=request_meta)
    async with harness.serve_http(harness.http_app(server)) as url:
        async with harness.connect_http(url, trace_client=True) as client:
            assert wrap_mcp_client(client.raw, capture_content=capture_content) is client.raw
            wrap_mcp_client(client.raw, capture_content=capture_content)
            with start_span(name="agent") as agent:
                await client.call_tool("add", {"a": 4, "b": 5}, meta={"Baggage": "tenant=demo", "custom": "untouched"})
                await client.call_tool("fail")
                await client.get_prompt("greet", {"name": "Ada"})
                await client.read_resource("resource://greeting")
                image = await client.call_tool("image")
                assert image.content[0].data == harness.PNG_BASE64
                assert current_span() is agent
    (meta,) = request_meta
    assert meta["custom"] == "untouched"
    assert "tenant=demo" in meta["baggage"]
    assert "braintrust.parent=" in meta["baggage"]
    assert "Baggage" not in meta
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", meta["traceparent"])
    spans = client_spans(memory_logger.pop())
    for span in spans:
        assert_client(span, span["metadata"]["mcp.method.name"])
    tools = [s for s in spans if s["metadata"]["mcp.method.name"] == "tools/call"]
    assert len(tools) == 3
    for span in tools:
        assert span["span_attributes"]["type"] == "tool"
        assert span["span_parents"] == [agent.span_id]
        assert "tokens" not in span["metrics"]
    discovery = find_span_by_name(spans, "tools/list")
    assert discovery.get("input") is None and discovery.get("output") is None
    failed = find_span_by_name(spans, "tools/call fail")
    assert failed["metadata"]["error.type"] == "tool_error"
    initialized = [s for s in spans if s["metadata"]["mcp.method.name"] == "initialize"]
    if initialized:
        assert initialized[0]["metadata"]["mcp.server.name"] == "braintrust-mcp-test"
    else:
        assert find_span_by_name(spans, "server/discover")
    from braintrust import Attachment

    if capture_content:
        tool = find_span_by_name(spans, "tools/call add")
        assert tool["input"] == {"a": 4, "b": 5}
        assert tool["output"]["content"][0]["text"] == "9"
        assert failed["output"]["isError"] is True
        assert failed["error"] == failed["output"]["content"][0]["text"]
        prompt = find_span_by_name(spans, "prompts/get greet")
        assert prompt["input"] == {"name": "Ada"}
        assert prompt["output"]["messages"][0]["content"]["text"] == "Say hello to Ada."
        resource = find_span_by_name(spans, "resources/read")
        assert resource["input"] == {"uri": "resource://greeting"}
        assert resource["output"]["contents"][0]["text"] == "Hello from a resource"
        assert isinstance(find_span_by_name(spans, "tools/call image")["output"]["content"][0]["data"], Attachment)
    else:
        assert failed["error"] == "Tool returned an error result"
        for span in spans:
            assert span.get("input") is None and span.get("output") is None


@pytest.mark.asyncio
async def test_cancelled_client_operation(memory_logger):
    harness.cancel_tool_started = anyio.Event()
    async with harness.connect_in_memory(harness.build_server()) as client:
        wrap_mcp_client(client.raw)
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                tg.start_soon(client.call_tool, "wait_for_cancel")
                await harness.cancel_tool_started.wait()
                tg.cancel_scope.cancel()
        await client.call_tool("add", {"a": 1, "b": 1})
    spans = client_spans(memory_logger.pop())
    cancelled = find_span_by_name(spans, "tools/call wait_for_cancel")
    assert_client(cancelled, "tools/call")
    assert cancelled["metadata"]["mcp.cancelled"] is True
    assert not cancelled.get("error") and "error.type" not in cancelled["metadata"]
    assert not cancelled.get("output")
    assert "mcp.cancelled" not in find_span_by_name(spans, "tools/call add")["metadata"]


@pytest.mark.skipif(harness.FLAVOR != "fastmcp", reason="Only FastMCP supports reconnecting the same client")
@pytest.mark.asyncio
async def test_fastmcp_reconnect(memory_logger):
    from fastmcp import Client

    client = wrap_mcp_client(Client(harness.build_server()))
    for _ in range(2):
        async with client:
            wrap_mcp_client(client)
            await client.call_tool("add", {"a": 1, "b": 2})
    tools = [s for s in client_spans(memory_logger.pop()) if s["metadata"]["mcp.method.name"] == "tools/call"]
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_request_copy_and_baggage(memory_logger):
    from mcp import types

    async with harness.connect_in_memory(wrap_mcp_server(harness.build_server())) as client:
        session = client.raw.session if hasattr(client.raw, "session") else client.raw
        wrap_mcp_client(session)
        data = {
            "method": "tools/call",
            "params": {
                "name": "add",
                "arguments": {"a": 1, "b": 2},
                "_meta": {"Baggage": "tenant=demo", "traceparent": "invalid", "custom": "untouched"},
            },
        }
        request = types.CallToolRequest.model_validate(data)
        if not harness.MCP_V2:
            request = types.ClientRequest(request)  # pylint: disable=not-callable
        before = request.model_dump(by_alias=True, mode="json", exclude_none=True)
        with start_span(name="agent"):
            await session.send_request(request, types.CallToolResult)
        assert request.model_dump(by_alias=True, mode="json", exclude_none=True) == before
    spans = memory_logger.pop()
    outgoing = find_span_by_name(client_spans(spans), "tools/call add")
    incoming = find_span_by_name([s for s in spans if s not in client_spans(spans)], "tools/call add")
    assert incoming["span_parents"] == [outgoing["span_id"]]
    assert "_meta" not in outgoing["input"]


@pytest.mark.asyncio
async def test_client_timeout(memory_logger):
    from datetime import timedelta

    async with harness.connect_in_memory(harness.build_server()) as client:
        session = client.raw.session if hasattr(client.raw, "session") else client.raw
        wrap_mcp_client(session)
        timeout = 0.05 if harness.MCP_V2 else timedelta(seconds=0.05)
        with pytest.raises(Exception):
            await session.call_tool("wait_for_cancel", read_timeout_seconds=timeout)
        await client.call_tool("add", {"a": 1, "b": 1})
    spans = client_spans(memory_logger.pop())
    timed_out = find_span_by_name(spans, "tools/call wait_for_cancel")
    assert_client(timed_out, "tools/call")
    assert timed_out["error"] and timed_out["metadata"]["error.type"]
    assert not timed_out.get("output")
    assert not find_span_by_name(spans, "tools/call add").get("error")


@pytest.mark.asyncio
async def test_client_disconnect(memory_logger, tmp_path):
    env = {
        harness.SPANS_FILE_ENV: str(tmp_path / "unused-spans.json"),
    }
    with anyio.fail_after(10):
        with pytest.raises(Exception):
            async with harness.connect_stdio(env) as client:
                wrap_mcp_client(client.raw)
                await client.call_tool("disconnect")
    span = find_span_by_name(client_spans(memory_logger.pop()), "tools/call disconnect")
    assert_client(span, "tools/call")
    assert span["error"] and span["metadata"]["error.type"]
    assert not span.get("output")
