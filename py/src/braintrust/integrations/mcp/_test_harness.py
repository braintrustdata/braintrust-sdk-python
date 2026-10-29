"""Shared MCP server/client harness for the MCP integration tests.

Builds the same demo server on the bundled server abstraction (``FastMCP`` on
MCP 1.x, ``MCPServer`` on MCP 2.x) or on standalone ``fastmcp``, and connects
real MCP clients to it in-process, over streamable HTTP (uvicorn on
localhost), or over stdio (a real server subprocess running this module).

Run as ``python _test_harness.py stdio`` to serve the demo server over stdio;
the subprocess records its spans with the memory logger and writes them to
``BRAINTRUST_MCP_TEST_SPANS_FILE`` on exit.
"""

import contextlib
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

import anyio
import mcp
from braintrust.integrations.mcp import MCPIntegration
from braintrust.integrations.mcp import _resolve_lowlevel_server as lowlevel_server
from braintrust.integrations.mcp.patchers import _MCP_V2
from braintrust.integrations.versioning import version_satisfies


FLAVOR = os.environ.get("BRAINTRUST_MCP_TEST_FLAVOR") or (
    "fastmcp" if importlib.util.find_spec("fastmcp") is not None else "mcp"
)
MCP_V2 = version_satisfies(MCPIntegration.detect_version(mcp), _MCP_V2)
FASTMCP_MAJOR = int(importlib.import_module("fastmcp").__version__.split(".")[0]) if FLAVOR == "fastmcp" else 0
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
SPANS_FILE_ENV = "BRAINTRUST_MCP_TEST_SPANS_FILE"
WRAP_ENV = "BRAINTRUST_MCP_TEST_WRAP"

# Set by the ``wait_for_cancel`` tool so tests can cancel a running request.
cancel_tool_started: anyio.Event | None = None
cancel_tool_request_id: Any = None


def _server_classes() -> tuple[Any, Any]:
    if FLAVOR == "fastmcp":
        from fastmcp import Context, FastMCP

        return FastMCP, Context
    if MCP_V2:
        from mcp.server.mcpserver import Context, MCPServer

        return MCPServer, Context
    from mcp.server.fastmcp import Context, FastMCP  # pylint: disable=no-name-in-module  # MCP 1.x

    return FastMCP, Context


def build_server(name: str = "braintrust-mcp-test") -> Any:
    from braintrust.logger import start_span

    server_class, context_class = _server_classes()
    server = server_class(name)

    @server.tool()
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        with start_span(name="nested-work") as span:
            span.log(input={"a": a, "b": b})
        return a + b

    @server.tool()
    async def fail() -> str:
        """Always fails."""
        raise ValueError("tool exploded")

    @server.tool()
    async def image() -> Any:
        """Return an inline image."""
        return [mcp.types.ImageContent.model_validate({"type": "image", "data": PNG_BASE64, "mimeType": "image/png"})]

    async def notify(ctx: context_class) -> str:  # type: ignore[valid-type]
        """Send a log notification while running."""
        await ctx.info("working on it")
        return "done"

    server.tool()(notify)

    async def notify_after_disabling_capture(ctx: context_class) -> str:  # type: ignore[valid-type]
        """Turn content capture off mid-request, then send a notification."""
        from braintrust.integrations.mcp import wrap_mcp_server

        wrap_mcp_server(server, capture_content=False)
        await ctx.info("sensitive progress detail")
        return "done"

    server.tool()(notify_after_disabling_capture)

    async def wait_for_cancel(ctx: context_class) -> str:  # type: ignore[valid-type]
        """Block until cancelled."""
        global cancel_tool_request_id
        cancel_tool_request_id = ctx.request_id
        if cancel_tool_started is not None:
            cancel_tool_started.set()
        await anyio.sleep(30)
        return "not cancelled"

    server.tool()(wait_for_cancel)

    @server.prompt()
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Say hello to {name}."

    @server.prompt()
    def broken() -> str:
        """Always fails to render."""
        raise RuntimeError("prompt exploded")

    @server.resource("resource://greeting")
    def greeting() -> str:
        return "Hello from a resource"

    @server.resource("resource://pixel.png", mime_type="image/png")
    def pixel() -> bytes:
        import base64

        return base64.b64decode(PNG_BASE64)

    return server


def http_app(server: Any) -> Any:
    if FLAVOR == "fastmcp":
        return server.http_app()
    return server.streamable_http_app()


class Client:
    """Uniform calls over the real MCP or fastmcp client for each flavor."""

    def __init__(self, client: Any):
        self.raw = client

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, meta: Any = None) -> Any:
        kwargs: dict[str, Any] = {}
        if meta is not None:
            kwargs["meta"] = meta
        if FLAVOR == "fastmcp":
            kwargs["raise_on_error"] = False
        return await self.raw.call_tool(name, arguments or {}, **kwargs)

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.raw.get_prompt(name, arguments)

    async def read_resource(self, uri: str) -> Any:
        if FLAVOR == "mcp" and not MCP_V2:
            from pydantic import AnyUrl

            return await self.raw.read_resource(AnyUrl(uri))
        return await self.raw.read_resource(uri)

    async def list_tools(self) -> Any:
        return await self.raw.list_tools()


@contextlib.asynccontextmanager
async def connect_in_memory(
    server: Any, mode: str | None = None, raise_exceptions: bool = False
) -> AsyncIterator[Client]:
    if FLAVOR == "fastmcp":
        from fastmcp import Client as FastMCPClient

        async with FastMCPClient(server) as client:
            yield Client(client)
    elif MCP_V2:
        from mcp import Client as MCPClient

        async with MCPClient(server, mode=mode or "legacy", log_level="info") as client:
            yield Client(client)
    else:
        from mcp.shared.memory import (  # pylint: disable=no-name-in-module  # MCP 1.x
            create_connected_server_and_client_session,
        )

        async with create_connected_server_and_client_session(
            lowlevel_server(server), raise_exceptions=raise_exceptions
        ) as session:
            yield Client(session)


@contextlib.asynccontextmanager
async def connect_http(url: str) -> AsyncIterator[Client]:
    if FLAVOR == "fastmcp":
        from fastmcp import Client as FastMCPClient

        async with FastMCPClient(url) as client:
            yield Client(client)
    elif MCP_V2:
        from mcp import Client as MCPClient

        async with MCPClient(url, log_level="info") as client:
            yield Client(client)
    else:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client  # pylint: disable=no-name-in-module  # MCP 1.x

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield Client(session)


@contextlib.asynccontextmanager
async def connect_stdio(env: dict[str, str]) -> AsyncIterator[Client]:
    args = [os.path.abspath(__file__), "stdio"]
    full_env = {**os.environ, **env}
    if FLAVOR == "fastmcp":
        from fastmcp import Client as FastMCPClient
        from fastmcp.client.transports import StdioTransport

        transport = StdioTransport(command=sys.executable, args=args, env=full_env, keep_alive=False)
        async with FastMCPClient(transport) as client:
            yield Client(client)
        return
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=args, env=full_env)
    if MCP_V2:
        from mcp import Client as MCPClient

        async with MCPClient(stdio_client(params), log_level="info") as client:
            yield Client(client)
    else:
        from mcp import ClientSession

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield Client(session)


@contextlib.asynccontextmanager
async def serve_http(app: Any) -> AsyncIterator[str]:
    """Serve *app* on an ephemeral localhost port in the current event loop.

    Same loop and thread as the test, so the server's spans reach the test's
    memory logger.
    """
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve)
        with anyio.fail_after(10):
            while not server.started:
                await anyio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True


async def _serve_stdio() -> None:
    from braintrust import logger
    from braintrust.integrations.mcp import wrap_mcp_server
    from braintrust.test_helpers import init_test_logger

    init_test_logger("test-mcp-stdio")
    with logger._internal_with_memory_background_logger() as memory_logger:
        server = build_server()
        if os.environ.get(WRAP_ENV) == "setup":
            MCPIntegration.setup()
        else:
            wrap_mcp_server(server)
        run = server.run_stdio_async
        if "show_banner" in inspect.signature(run).parameters:
            await run(show_banner=False)
        else:
            await run()
        spans = memory_logger.pop()
    with open(os.environ[SPANS_FILE_ENV], "w") as f:
        json.dump(spans, f, default=repr)


if __name__ == "__main__" and sys.argv[1:] == ["stdio"]:
    anyio.run(_serve_stdio)
