"""MCP server instrumentation tests.

The same tests run against the bundled server abstraction (``test_mcp``
sessions: ``FastMCP`` on MCP 1.x, ``MCPServer`` on MCP 2.x) and standalone
``fastmcp`` (``test_fastmcp`` sessions). MCP traffic is always real: clients
talk to servers in-process, over streamable HTTP on localhost, or over stdio
to a server subprocess. Model calls made inside tools replay from cassettes.
"""

import asyncio
import json

import anyio
import pytest
from braintrust import Attachment, logger
from braintrust.conftest import get_vcr_config
from braintrust.integrations.mcp import MCPIntegration, wrap_mcp_server
from braintrust.integrations.mcp import _test_harness as harness
from braintrust.integrations.mcp.tracing import set_global_server_config
from braintrust.logger import start_span
from braintrust.test_helpers import find_span_by_name, init_test_logger


PROJECT_NAME = "test-mcp"
FLAVOR = harness.FLAVOR
INSTRUMENTATION = "mcp-auto"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(autouse=True)
def reset_global_server_config():
    yield
    set_global_server_config(None)


@pytest.fixture(scope="module")
def vcr_config():
    # MCP traffic to the localhost test server is real; only model calls replay.
    return {**get_vcr_config(), "ignore_localhost": True}


def _mcp_spans(spans):
    return [span for span in spans if "mcp.method.name" in (span.get("metadata") or {})]


def _assert_origin(span):
    assert span["context"]["span_origin"]["instrumentation"]["name"] == INSTRUMENTATION


@pytest.mark.asyncio
async def test_request_spans(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server) as client:
        await client.list_tools()
        await client.call_tool("add", {"a": 2, "b": 3})
        await client.call_tool("fail")

    spans = memory_logger.pop()
    span = find_span_by_name(spans, "tools/call add")
    assert span["span_attributes"]["type"] == "tool"
    assert span["input"] == {"a": 2, "b": 3}
    assert span["output"]["content"] == [{"type": "text", "text": "5"}]
    assert span["output"]["isError"] is False
    metadata = span["metadata"]
    assert metadata["mcp.method.name"] == "tools/call"
    assert metadata["gen_ai.tool.name"] == "add"
    assert metadata["gen_ai.operation.name"] == "execute_tool"
    assert isinstance(metadata["jsonrpc.request.id"], str)
    assert metadata["mcp.protocol.version"]
    assert metadata["mcp.client.name"]
    assert "error.type" not in metadata
    _assert_origin(span)

    nested = find_span_by_name(spans, "nested-work")
    assert nested["span_parents"] == [span["span_id"]]
    assert nested["root_span_id"] == span["root_span_id"]
    # User spans do not inherit the integration's provenance.
    assert nested["context"]["span_origin"]["instrumentation"]["name"] != INSTRUMENTATION

    # A returned tool error fails the span without a JSON-RPC status code.
    failed = find_span_by_name(spans, "tools/call fail")
    assert failed["metadata"]["error.type"] == "tool_error"
    assert "rpc.response.status_code" not in failed["metadata"]
    assert failed["output"]["isError"] is True
    assert failed["error"] == failed["output"]["content"][0]["text"]

    # Initialization and discovery requests get method-named function spans
    # without payloads.
    discovery = [
        s
        for s in _mcp_spans(spans)
        if s["metadata"]["mcp.method.name"] in ("initialize", "server/discover", "tools/list")
    ]
    assert {s["metadata"]["mcp.method.name"] for s in discovery} & {"initialize", "server/discover"}
    assert "tools/list" in {s["span_attributes"]["name"] for s in discovery}
    for span in discovery:
        assert span["span_attributes"]["type"] == "function"
        assert span.get("input") is None
        assert span.get("output") is None


def _missing_resource_status_code():
    if harness.MCP_V2:
        return "-32602"
    if FLAVOR == "fastmcp" and harness.FASTMCP_MAJOR >= 3:
        return "-32002"
    # MCP 1.x FastMCP and fastmcp 2 report unknown resources as handler failures.
    return "0"


@pytest.mark.asyncio
async def test_prompt_and_resource_spans(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server) as client:
        await client.get_prompt("greet", {"name": "Ada"})
        await client.read_resource("resource://greeting")
        with pytest.raises(Exception):
            await client.read_resource("resource://missing")

    spans = memory_logger.pop()
    prompt = find_span_by_name(spans, "prompts/get greet")
    assert prompt["span_attributes"]["type"] == "function"
    assert prompt["input"] == {"name": "Ada"}
    assert prompt["output"]["messages"][0]["role"] == "user"
    assert prompt["output"]["messages"][0]["content"] == {"type": "text", "text": "Say hello to Ada."}
    assert prompt["metadata"]["gen_ai.prompt.name"] == "greet"
    assert "gen_ai.operation.name" not in prompt["metadata"]

    reads = {s["input"]["uri"]: s for s in spans if s["span_attributes"]["name"] == "resources/read"}
    resource = reads["resource://greeting"]
    assert resource["span_attributes"]["type"] == "function"
    assert resource["metadata"]["mcp.resource.uri"] == "resource://greeting"
    assert resource["output"]["contents"][0]["text"] == "Hello from a resource"
    # Resource URIs stay out of span names.
    assert all("resource://" not in s["span_attributes"]["name"] for s in spans)

    missing = reads["resource://missing"]
    code = _missing_resource_status_code()
    assert missing["metadata"]["rpc.response.status_code"] == code
    if code == "0":
        assert missing["metadata"]["error.type"] == "0"
        assert missing["error"]
    else:
        # Caller errors do not mark the server span as failed.
        assert "error.type" not in missing["metadata"]
        assert not missing.get("error")
    assert missing.get("output") is None


@pytest.mark.asyncio
async def test_inline_media_becomes_attachments(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server) as client:
        result = await client.call_tool("image")
        await client.read_resource("resource://pixel.png")

    # The application still sees the original base64 data.
    assert result.content[0].data == harness.PNG_BASE64

    spans = memory_logger.pop()
    image_block = find_span_by_name(spans, "tools/call image")["output"]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["mimeType"] == "image/png"
    assert isinstance(image_block["data"], Attachment)
    assert image_block["data"].reference["content_type"] == "image/png"

    resource = find_span_by_name(spans, "resources/read")["output"]["contents"][0]
    assert resource["uri"] == "resource://pixel.png"
    assert isinstance(resource["blob"], Attachment)
    assert resource["blob"].reference["filename"] == "pixel.png"


@pytest.mark.asyncio
async def test_capture_content_disabled(memory_logger):
    server = wrap_mcp_server(harness.build_server(), capture_content=False)

    async with harness.connect_in_memory(server) as client:
        await client.call_tool("add", {"a": 1, "b": 1})
        await client.call_tool("fail")
        await client.read_resource("resource://greeting")

    spans = memory_logger.pop()
    for span in _mcp_spans(spans):
        assert span.get("input") is None, span["span_attributes"]["name"]
        assert span.get("output") is None, span["span_attributes"]["name"]
    failed = find_span_by_name(spans, "tools/call fail")
    assert failed["error"] == "Tool returned an error result"
    assert failed["metadata"]["error.type"] == "tool_error"
    # Identity, timing, and allowlisted metadata stay available.
    assert find_span_by_name(spans, "resources/read")["metadata"]["mcp.resource.uri"] == "resource://greeting"


@pytest.mark.asyncio
async def test_outgoing_notification_is_child_of_request(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server) as client:
        await client.call_tool("notify")

    spans = memory_logger.pop()
    tool = find_span_by_name(spans, "tools/call notify")
    notification = find_span_by_name(spans, "notifications/message")
    assert notification["span_parents"] == [tool["span_id"]]
    assert notification["span_attributes"]["type"] == "function"
    assert notification["metadata"]["mcp.method.name"] == "notifications/message"
    assert notification["output"]["level"] == "info"
    assert "jsonrpc.request.id" not in notification["metadata"]
    _assert_origin(notification)


@pytest.mark.asyncio
async def test_incoming_trace_context_sets_parent(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    with start_span(name="remote-caller") as remote:
        carrier = remote.inject({})

    async with harness.connect_in_memory(server) as client:
        await client.call_tool("add", {"a": 1, "b": 2}, meta=dict(carrier))
        # Invalid context, under an ambient caller span.
        with start_span(name="test-caller") as caller:
            await client.call_tool("add", {"a": 1, "b": 2}, meta={"traceparent": "00-zz-bad-01"})

    spans = memory_logger.pop()
    linked, fresh = [s for s in spans if s["span_attributes"]["name"] == "tools/call add"]
    assert linked["root_span_id"] == remote.root_span_id
    assert linked["span_parents"] == [remote.span_id]
    # Propagation fields are never copied into the span.
    assert not any(key in linked["metadata"] for key in ("traceparent", "baggage", "tracestate", "_meta"))
    # Invalid context starts a new trace and does not break the request; the
    # ambient span is never the parent.
    assert fresh["root_span_id"] not in (remote.root_span_id, caller.root_span_id)
    assert not fresh.get("span_parents")


@pytest.mark.asyncio
async def test_concurrent_requests_keep_separate_contexts(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server) as client:
        await asyncio.gather(*(client.call_tool("add", {"a": i, "b": i}) for i in range(5)))

    spans = memory_logger.pop()
    tools = {s["span_id"]: s for s in spans if s["span_attributes"]["name"] == "tools/call add"}
    assert len(tools) == 5
    assert len({s["metadata"]["jsonrpc.request.id"] for s in tools.values()}) == 5
    for nested in (s for s in spans if s["span_attributes"]["name"] == "nested-work"):
        (parent_id,) = nested["span_parents"]
        assert tools[parent_id]["input"] == nested["input"]


@pytest.mark.asyncio
async def test_repeated_instrumentation_does_not_duplicate_spans(memory_logger):
    server = harness.build_server()
    wrap_mcp_server(server)
    wrap_mcp_server(server)
    wrap_mcp_server(harness.lowlevel_server(server))
    assert MCPIntegration.setup()
    assert MCPIntegration.setup()

    async with harness.connect_in_memory(server) as client:
        await client.call_tool("add", {"a": 1, "b": 2})
        await client.call_tool("add", {"a": 1, "b": 2})

    spans = memory_logger.pop()
    tool_spans = [s for s in spans if s["span_attributes"]["name"] == "tools/call add"]
    # Separate requests have distinct spans, and no request is traced twice.
    assert len(tool_spans) == 2
    assert len({s["span_id"] for s in tool_spans}) == 2


@pytest.mark.asyncio
async def test_only_wrapped_servers_are_traced_until_setup(memory_logger):
    wrap_mcp_server(harness.build_server("other"))
    server = harness.build_server()

    async with harness.connect_in_memory(server) as client:
        await client.call_tool("add", {"a": 1, "b": 2})
    spans = memory_logger.pop()
    assert not _mcp_spans(spans)
    assert [s["span_attributes"]["name"] for s in spans] == ["nested-work"]

    assert MCPIntegration.setup()
    async with harness.connect_in_memory(server) as client:
        await client.call_tool("add", {"a": 1, "b": 2})
    _assert_origin(find_span_by_name(memory_logger.pop(), "tools/call add"))


@pytest.mark.asyncio
async def test_cancelled_request(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    harness.cancel_tool_started = anyio.Event()

    async with harness.connect_in_memory(server) as client:
        if not harness.MCP_V2:
            # MCP 1.x clients do not send notifications/cancelled on their own.
            from mcp import types

            session = client.raw if FLAVOR == "mcp" else client.raw.session

            async def cancel_when_started():
                await harness.cancel_tool_started.wait()
                await session.send_notification(
                    types.ClientNotification(  # pylint: disable=not-callable  # MCP 1.x root model
                        types.CancelledNotification(
                            params=types.CancelledNotificationParams(requestId=int(harness.cancel_tool_request_id))
                        )
                    )
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(cancel_when_started)
                with pytest.raises(Exception):
                    await client.call_tool("wait_for_cancel")
        else:
            # The client sends notifications/cancelled when its call is cancelled.
            with anyio.move_on_after(10):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(client.call_tool, "wait_for_cancel")
                    await harness.cancel_tool_started.wait()
                    tg.cancel_scope.cancel()
        await client.call_tool("add", {"a": 1, "b": 1})

    spans = memory_logger.pop()
    cancelled = find_span_by_name(spans, "tools/call wait_for_cancel")
    assert cancelled["metadata"]["mcp.cancelled"] is True
    assert "error.type" not in cancelled["metadata"]
    assert not cancelled.get("error")
    assert cancelled.get("output") is None

    notification = find_span_by_name(spans, "notifications/cancelled")
    assert notification["span_id"] != cancelled["span_id"]
    assert "mcp.cancelled" not in notification["metadata"]
    assert "reason" not in (notification.get("input") or {})
    # Requests after the cancellation are unaffected.
    assert "mcp.cancelled" not in find_span_by_name(spans, "tools/call add")["metadata"]


@pytest.mark.asyncio
async def test_disconnect_ends_pending_request_with_error(memory_logger):
    server = wrap_mcp_server(harness.build_server())
    harness.cancel_tool_started = anyio.Event()

    async def call(client):
        with pytest.raises(BaseException):
            await client.call_tool("wait_for_cancel")

    with anyio.move_on_after(10):
        async with anyio.create_task_group() as tg:
            async with harness.connect_in_memory(server) as client:
                tg.start_soon(call, client)
                await harness.cancel_tool_started.wait()
            # The connection closed under the pending request.
            tg.cancel_scope.cancel()

    spans = memory_logger.pop()
    span = find_span_by_name(spans, "tools/call wait_for_cancel")
    assert span["error"]
    assert span["metadata"]["error.type"]
    assert "mcp.cancelled" not in span["metadata"]


@pytest.mark.vcr(cassette_name=f"{FLAVOR}_test_http_tool_with_model_call")
@pytest.mark.asyncio
async def test_http_tool_with_model_call(memory_logger):
    server = wrap_mcp_server(harness.build_server())

    async with harness.serve_http(harness.http_app(server)) as url:
        async with harness.connect_http(url) as client:
            await client.call_tool("ask_llm", {"question": "Reply with the single word: pong"})

    spans = memory_logger.pop()
    tool = find_span_by_name(spans, "tools/call ask_llm")
    llm = find_span_by_name(spans, "anthropic.messages.create")
    assert llm["span_parents"] == [tool["span_id"]]
    assert llm["span_attributes"]["type"] == "llm"
    assert llm["metrics"]["tokens"] > 0
    # MCP spans never invent model fields.
    assert "model" not in tool["metadata"]
    assert "tokens" not in tool["metrics"]
    assert tool["output"]["content"][0]["type"] == "text"

    metadata = tool["metadata"]
    assert metadata["network.protocol.name"] == "http"
    assert metadata["network.protocol.version"] == "1.1"
    assert metadata["network.transport"] == "tcp"
    assert metadata["client.address"] == "127.0.0.1"
    assert isinstance(metadata["client.port"], int)
    if "mcp.session.id" in metadata:
        assert isinstance(metadata["mcp.session.id"], str)
    _assert_origin(tool)


@pytest.mark.parametrize("wrap", ["wrap", "setup"])
@pytest.mark.asyncio
async def test_stdio_subprocess(tmp_path, request, wrap):
    spans_file = tmp_path / "spans.json"
    record_mode = request.config.getoption("--vcr-record", default=None)
    if request.config.getoption("--disable-vcr", default=False):
        record_mode = "disabled"
    env = {
        harness.SPANS_FILE_ENV: str(spans_file),
        harness.CASSETTE_ENV: f"{FLAVOR}_test_stdio_subprocess",
        harness.WRAP_ENV: wrap,
    }
    if record_mode:
        env[harness.RECORD_MODE_ENV] = record_mode

    async with harness.connect_stdio(env) as client:
        await client.call_tool("add", {"a": 2, "b": 2})
        await client.call_tool("ask_llm", {"question": "Reply with the single word: pong"})
        await client.call_tool("notify")

    with anyio.fail_after(10):
        while not spans_file.exists():
            await anyio.sleep(0.05)
    spans = json.loads(spans_file.read_text())

    add = find_span_by_name(spans, "tools/call add")
    assert add["input"] == {"a": 2, "b": 2}
    assert find_span_by_name(spans, "nested-work")["span_parents"] == [add["span_id"]]
    ask = find_span_by_name(spans, "tools/call ask_llm")
    assert find_span_by_name(spans, "anthropic.messages.create")["span_parents"] == [ask["span_id"]]
    assert find_span_by_name(spans, "notifications/message")["span_parents"] == [
        find_span_by_name(spans, "tools/call notify")["span_id"]
    ]
    for span in _mcp_spans(spans):
        assert "network.protocol.name" not in span["metadata"]
        _assert_origin(span)
    initialized = [s for s in spans if s["span_attributes"]["name"] == "notifications/initialized"]
    if initialized:
        assert "jsonrpc.request.id" not in initialized[0]["metadata"]


@pytest.mark.skipif(FLAVOR != "mcp" or not harness.MCP_V2, reason="MCP 2.x in-process modern protocol path")
@pytest.mark.asyncio
async def test_modern_in_process_requests(memory_logger):
    # Modern in-process clients dispatch directly, without a JSON-RPC
    # dispatcher, so operations end when the runner returns.
    server = wrap_mcp_server(harness.build_server())

    async with harness.connect_in_memory(server, mode="auto") as client:
        await client.call_tool("add", {"a": 1, "b": 2})
        await client.call_tool("fail")

    spans = memory_logger.pop()
    add = find_span_by_name(spans, "tools/call add")
    assert find_span_by_name(spans, "nested-work")["span_parents"] == [add["span_id"]]
    assert add["metadata"]["mcp.protocol.version"] >= "2026"
    assert add["metadata"]["mcp.server.name"] == "braintrust-mcp-test"
    assert find_span_by_name(spans, "tools/call fail")["metadata"]["error.type"] == "tool_error"
