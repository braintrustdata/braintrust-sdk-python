"""Braintrust instrumentation for Model Context Protocol clients and servers."""

import logging
import sys
from typing import Any, TypeVar

from braintrust.integrations.versioning import version_satisfies

from ._client import CLIENT_CONFIG_ATTR, ClientConfig, configure_session, set_session_config
from .integration import MCPIntegration
from .patchers import ClientEnterPatcher, ServerRunPatcher
from .tracing import SERVER_CONFIG_ATTR, MCPServerConfig


logger = logging.getLogger(__name__)

ServerT = TypeVar("ServerT")


def _resolve_lowlevel_server(server: Any) -> Any | None:
    try:
        from mcp.server.lowlevel.server import Server
    except ImportError:
        return None
    if isinstance(server, Server):
        return server
    # Bundled FastMCP (1.x), MCPServer (2.x), and standalone fastmcp.FastMCP
    # all delegate protocol handling to a low-level server.
    for attr in ("_lowlevel_server", "_mcp_server"):
        candidate = getattr(server, attr, None)
        if isinstance(candidate, Server):
            return candidate
    return None


def wrap_mcp_server(server: ServerT, capture_content: bool = True) -> ServerT:
    """Trace incoming MCP requests to *server*, from receipt through response sending.

    Accepts a low-level ``mcp.server.lowlevel.Server``, the bundled
    ``FastMCP`` (MCP 1.x) or ``MCPServer`` (MCP 2.x), or a standalone
    ``fastmcp.FastMCP`` server, and returns the same object. Calling it again
    updates the settings without adding duplicate spans.

    :param server: The MCP server to trace.
    :param capture_content: Log tool and prompt arguments, results, resource
        contents, and notification parameters (default). When False, spans keep
        names, timing, allowlisted metadata, and error classification only.
    """
    lowlevel = _resolve_lowlevel_server(server)
    if lowlevel is None:
        logger.warning("Unsupported MCP server %r. Not wrapping.", type(server).__name__)
        return server
    if not MCPIntegration.install_patchers():
        logger.warning("Unsupported mcp version. Not wrapping.")
        return server
    setattr(lowlevel, SERVER_CONFIG_ATTR, MCPServerConfig(capture_content=capture_content))
    import mcp

    if version_satisfies(MCPIntegration.detect_version(mcp), ServerRunPatcher.version_spec):
        ServerRunPatcher.wrap_target(lowlevel)
    return server


def wrap_mcp_client(client_or_session: ServerT, capture_content: bool = True) -> ServerT:
    """Trace outgoing MCP round trips and propagate W3C context in ``params._meta``.

    Accepts an official ``mcp.ClientSession`` (1.x/2.x), ``mcp.Client`` (2.x),
    or standalone ``fastmcp.Client`` (2.x–4.x). Returns the same object; wrap
    before connecting to include initialization, or wrap an active session.
    Repeated calls update content settings without adding operation spans.

    Content capture defaults to True. False omits arguments and results;
    method, timing, identifiers, and error classification remain available.
    Each client and server can be instrumented independently.

    One-way notifications and detached task APIs have no client operation span.
    If ``opentelemetry-instrumentation-mcp`` is enabled, its instrumentor owns
    client spans, propagation, and capture settings; no duplicate span is added.
    """
    try:
        import mcp
        from mcp.client.session import ClientSession
    except ImportError:
        return client_or_session
    supported = [ClientSession]
    if hasattr(mcp, "Client"):
        supported.append(mcp.Client)
    fastmcp = sys.modules.get("fastmcp")
    if fastmcp is not None and hasattr(fastmcp, "Client"):
        supported.append(fastmcp.Client)
    if not isinstance(client_or_session, tuple(supported)):
        logger.warning("Unsupported MCP client %r. Not wrapping.", type(client_or_session).__name__)
        return client_or_session
    if not MCPIntegration.install_patchers():
        logger.warning("Unsupported mcp version. Not wrapping.")
        return client_or_session
    config = ClientConfig(capture_content=capture_content)
    if isinstance(client_or_session, ClientSession):
        set_session_config(client_or_session, config)
    else:
        setattr(client_or_session, CLIENT_CONFIG_ATTR, config)
        configure_session(client_or_session, config)
        # Python resolves special methods on the class, not the instance.
        ClientEnterPatcher.wrap_target(type(client_or_session))
    return client_or_session


__all__ = ["MCPIntegration", "wrap_mcp_client", "wrap_mcp_server"]
