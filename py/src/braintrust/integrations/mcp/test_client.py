"""Real MCP client operations, independently and connected to server traces."""

import asyncio
import json
import re

import anyio
import pytest
from braintrust import logger
from braintrust.conftest import get_vcr_config
from braintrust.integrations.mcp import _test_harness as harness
from braintrust.integrations.mcp import wrap_mcp_client, wrap_mcp_server
from braintrust.logger import current_span, start_span
from braintrust.test_helpers import find_span_by_name, init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-mcp-client")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(scope="module")
def vcr_config():
    return {**get_vcr_config(), "ignore_localhost": True}


def client_spans(spans):
    return [s for s in spans if s.get("metadata", {}).get("mcp.span.kind") == "client"]


def assert_client(span, method):
    assert span["metadata"]["mcp.method.name"] == method
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "mcp-auto"
    assert span["metrics"]["end"] >= span["metrics"]["start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("server_traced", [False, True])
async def test_client_operations(memory_logger, server_traced):
    server = harness.build_server()
    if server_traced:
        wrap_mcp_server(server)
    async with harness.connect_in_memory(server) as client:
        assert wrap_mcp_client(client.raw) is client.raw
        wrap_mcp_client(client.raw)
        with start_span(name="agent") as agent:
            await client.call_tool("add", {"a": 2, "b": 3})
            await client.get_prompt("greet", {"name": "Ada"})
            await client.read_resource("resource://greeting")
            await client.list_tools()
            assert current_span() is agent

    spans = memory_logger.pop()
    outgoing = client_spans(spans)
    tool = find_span_by_name(outgoing, "tools/call add")
    assert tool["span_parents"] == [agent.span_id]
    assert tool["span_attributes"]["type"] == "tool"
    assert tool["input"] == {"a": 2, "b": 3}
    assert tool["output"]["content"][0]["text"] == "5"
    assert "tokens" not in tool["metrics"]
    prompt = find_span_by_name(outgoing, "prompts/get greet")
    assert prompt["input"] == {"name": "Ada"}
    assert prompt["output"]["messages"][0]["content"]["text"] == "Say hello to Ada."
    assert find_span_by_name(outgoing, "resources/read")["input"] == {"uri": "resource://greeting"}
    discovery = find_span_by_name(outgoing, "tools/list")
    assert not discovery.get("input") and not discovery.get("output")
    for span in outgoing:
        assert_client(span, span["metadata"]["mcp.method.name"])
    if server_traced:
        incoming = [s for s in spans if s not in outgoing]
        remote = find_span_by_name(incoming, "tools/call add")
        assert remote["span_parents"] == [tool["span_id"]]
        nested = find_span_by_name(spans, "nested-work")
        assert nested["span_parents"] == [remote["span_id"]]
        assert remote["root_span_id"] == tool["root_span_id"] == agent.root_span_id
    else:
        assert len([s for s in spans if "mcp.method.name" in s.get("metadata", {})]) == len(outgoing)


@pytest.mark.asyncio
async def test_client_errors_and_capture_opt_out(memory_logger):
    async with harness.connect_in_memory(harness.build_server()) as client:
        wrap_mcp_client(client.raw)
        await client.call_tool("fail")
        wrap_mcp_client(client.raw, capture_content=False)
        await client.call_tool("fail")
        await client.call_tool("add", {"a": 1, "b": 2})
        with pytest.raises(Exception):
            await client.get_prompt("missing", {})
    spans = client_spans(memory_logger.pop())
    failures = [s for s in spans if s["span_attributes"]["name"] == "tools/call fail"]
    assert len(failures) == 2
    assert failures[0]["output"]["isError"] is True
    assert failures[0]["metadata"]["error.type"] == "tool_error"
    assert failures[0]["error"] == failures[0]["output"]["content"][0]["text"]
    assert failures[1]["error"] == "Tool returned an error result"
    for span in spans[1:]:
        assert span.get("input") is None and span.get("output") is None
    missing = find_span_by_name(spans, "prompts/get missing")
    assert missing["error"] and missing["metadata"]["error.type"]


@pytest.mark.asyncio
async def test_concurrent_connected_requests(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    async with harness.connect_in_memory(server) as client:
        wrap_mcp_client(client.raw)
        # Wrapping the high-level client and its session must still create one span.
        if hasattr(client.raw, "session"):
            wrap_mcp_client(client.raw.session)
        with start_span(name="agent"):
            await asyncio.gather(*(client.call_tool("add", {"a": i, "b": 1}) for i in range(8)))
    spans = memory_logger.pop()
    outgoing = [s for s in client_spans(spans) if s["metadata"]["mcp.method.name"] == "tools/call"]
    assert len(outgoing) == 8
    by_input = {s["input"]["a"]: s for s in outgoing}
    incoming = [s for s in spans if s["span_attributes"]["name"] == "tools/call add" and s not in outgoing]
    assert len(incoming) == 8
    for span in incoming:
        parent = by_input[span["input"]["a"]]
        assert span["span_parents"] == [parent["span_id"]]
        assert span["root_span_id"] == parent["root_span_id"]


@pytest.mark.vcr(cassette_name=f"{harness.FLAVOR}_test_http_tool_with_model_call")
@pytest.mark.asyncio
async def test_connected_http_model_call(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    async with harness.serve_http(harness.http_app(server)) as url:
        async with harness.connect_http(url) as client:
            wrap_mcp_client(client.raw)
            with start_span(name="agent") as agent:
                await client.call_tool("ask_llm", {"question": "Reply with the single word: pong"})
    spans = memory_logger.pop()
    outgoing = find_span_by_name(client_spans(spans), "tools/call ask_llm")
    incoming = find_span_by_name([s for s in spans if s not in client_spans(spans)], "tools/call ask_llm")
    llm = find_span_by_name(spans, "anthropic.messages.create")
    assert outgoing["span_parents"] == [agent.span_id]
    assert incoming["span_parents"] == [outgoing["span_id"]]
    assert llm["span_parents"] == [incoming["span_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("server_traced", [False, True])
async def test_connected_stdio_model_call(memory_logger, tmp_path, request, vcr_cassette_dir, server_traced):
    from pathlib import Path

    from braintrust.integrations.mcp._test_transport import TRANSCRIPT_ENV

    spans_file = tmp_path / "server-spans.json"
    transcript = tmp_path / "transport.json"
    env = {
        harness.SPANS_FILE_ENV: str(spans_file),
        harness.CASSETTE_ENV: f"{harness.FLAVOR}_test_stdio_subprocess",
        harness.RECORD_MODE_ENV: "none",
        harness.WRAP_ENV: "wrap" if server_traced else "none",
        TRANSCRIPT_ENV: str(transcript),
    }
    async with harness.connect_stdio(env) as client:
        wrap_mcp_client(client.raw)
        with start_span(name="agent") as agent:
            await client.call_tool("ask_llm", {"question": "Reply with the single word: pong"})
    with anyio.fail_after(10):
        while not spans_file.exists():
            await anyio.sleep(0.05)
    remote = json.loads(spans_file.read_text())
    outgoing = find_span_by_name(client_spans(memory_logger.pop()), "tools/call ask_llm")
    assert outgoing["span_parents"] == [agent.span_id]
    if server_traced:
        incoming = find_span_by_name(remote, "tools/call ask_llm")
        assert incoming["span_parents"] == [outgoing["span_id"]]
        assert incoming["root_span_id"] == outgoing["root_span_id"]
        assert find_span_by_name(remote, "anthropic.messages.create")["span_parents"] == [incoming["span_id"]]
    else:
        assert not [s for s in remote if "mcp.method.name" in s.get("metadata", {})]
    observed = json.loads(transcript.read_text())
    recording = Path(vcr_cassette_dir) / f"{harness.FLAVOR}_client_stdio.json"
    mode = request.config.getoption("--vcr-record", default=None)
    if mode == "all" or (mode == "once" and not recording.exists()):
        recording.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    else:
        assert observed == json.loads(recording.read_text())


def _normalize_http_request(request):
    # Normalize only variable trace identity in VCR's recording/matching copy.
    # The real transport still sends the client's own span context.
    if request.method == "POST" and request.body:
        data = json.loads(request.body)
        meta = data.get("params", {}).get("_meta", {})
        if meta.get("custom") == "untouched":
            assert "tenant=demo" in meta["baggage"]
            assert "braintrust.parent=" in meta["baggage"]
            assert "Baggage" not in meta
        if "traceparent" in meta:
            assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", meta["traceparent"])
            meta["traceparent"] = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
        request.body = json.dumps(data, sort_keys=True).encode()
    return request


@pytest.mark.vcr(
    cassette_name=f"{harness.FLAVOR}_client_http",
    ignore_localhost=False,
    match_on=["method", "scheme", "host", "path", "query", "body"],
    before_record_request=_normalize_http_request,
)
@pytest.mark.asyncio
async def test_client_http_recording(memory_logger):
    # VCR records the real MCP HTTP exchange, including initialize and teardown.
    # Replay exercises the real client against those protocol responses.
    async with harness.serve_http(harness.http_app(harness.build_server())) as url:
        async with harness.connect_http(url, trace_client=True) as client:
            await client.call_tool("add", {"a": 4, "b": 5}, meta={"Baggage": "tenant=demo", "custom": "untouched"})
            await client.call_tool("fail")
            await client.get_prompt("greet", {"name": "Ada"})
            await client.read_resource("resource://greeting")
            await client.call_tool("image")
    spans = client_spans(memory_logger.pop())
    assert find_span_by_name(spans, "tools/call add")["output"]["content"][0]["text"] == "9"
    assert find_span_by_name(spans, "tools/call fail")["metadata"]["error.type"] == "tool_error"
    assert find_span_by_name(spans, "prompts/get greet")["output"]["messages"]
    assert find_span_by_name(spans, "resources/read")["output"]["contents"]
    initialized = [s for s in spans if s["metadata"]["mcp.method.name"] == "initialize"]
    if initialized:
        assert initialized[0]["metadata"]["mcp.server.name"] == "braintrust-mcp-test"
    else:
        assert find_span_by_name(spans, "server/discover")
    from braintrust import Attachment

    assert isinstance(find_span_by_name(spans, "tools/call image")["output"]["content"][0]["data"], Attachment)


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


@pytest.mark.asyncio
async def test_preconnect_wrapping_and_reconnect(memory_logger):
    if harness.FLAVOR == "mcp" and not harness.MCP_V2:
        pytest.skip("MCP 1.x exposes sessions, covered by HTTP preconnect wrapping")
    if harness.FLAVOR == "fastmcp":
        from fastmcp import Client
    else:
        from mcp import Client
    client = wrap_mcp_client(Client(harness.build_server()))
    for _ in range(2):
        async with client:
            wrap_mcp_client(client)
            await client.call_tool("add", {"a": 1, "b": 2})
        if harness.FLAVOR == "mcp":
            # Official 2.x clients are single-use; FastMCP permits reconnects.
            client = wrap_mcp_client(Client(harness.build_server()))
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
        harness.CASSETTE_ENV: f"{harness.FLAVOR}_test_stdio_subprocess",
        harness.RECORD_MODE_ENV: "none",
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


@pytest.mark.skipif(harness.FLAVOR != "mcp" or not harness.MCP_V2, reason="Modern official MCP 2.x transport")
@pytest.mark.asyncio
async def test_modern_connected_client(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    async with harness.connect_in_memory(server, mode="auto") as client:
        wrap_mcp_client(client.raw)
        await client.call_tool("add", {"a": 1, "b": 2})
    spans = memory_logger.pop()
    outgoing = find_span_by_name(client_spans(spans), "tools/call add")
    incoming = find_span_by_name([s for s in spans if s not in client_spans(spans)], "tools/call add")
    assert incoming["span_parents"] == [outgoing["span_id"]]


@pytest.mark.asyncio
async def test_capture_opt_out_keeps_connected_parentage(memory_logger):
    server = wrap_mcp_server(harness.build_server(), capture_content=False)
    async with harness.connect_in_memory(server) as client:
        wrap_mcp_client(client.raw, capture_content=False)
        await client.call_tool("add", {"a": 1, "b": 2})
    spans = memory_logger.pop()
    outgoing = find_span_by_name(client_spans(spans), "tools/call add")
    incoming = find_span_by_name([s for s in spans if s not in client_spans(spans)], "tools/call add")
    assert incoming["span_parents"] == [outgoing["span_id"]]
    for span in (outgoing, incoming):
        assert not span.get("input") and not span.get("output")
