"""Braintrust instrumentation for Model Context Protocol (MCP) servers."""

import logging
from typing import Any, TypeVar

from braintrust.integrations.versioning import version_satisfies

from .integration import MCPIntegration
from .patchers import ServerRunPatcher
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


__all__ = ["MCPIntegration", "wrap_mcp_server"]
