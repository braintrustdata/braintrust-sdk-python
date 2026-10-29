from typing import Any

from braintrust.integrations.base import BaseIntegration

from ._client import ClientConfig, set_global_client_config
from .patchers import CLIENT_PATCHERS, SERVER_PATCHERS
from .tracing import MCPServerConfig, set_global_server_config


class MCPIntegration(BaseIntegration):
    """Braintrust instrumentation hooks for MCP on mcp>=1.28.1,<3.

    Covers low-level servers and the server abstractions built on them: the
    bundled ``FastMCP`` (1.x) / ``MCPServer`` (2.x) and standalone ``fastmcp``.
    ``setup()`` enables server tracing by default. Pass ``client=True`` to
    enable clients too, and ``server=False`` as well for clients only. Manual
    wrappers override the process defaults for individual clients and servers.
    """

    name = "mcp"
    import_names = ("mcp",)
    distribution_names = ("mcp",)
    min_version = "1.28.1"
    patchers = (*SERVER_PATCHERS, *CLIENT_PATCHERS)

    @classmethod
    def setup(
        cls, *, target: Any | None = None, client: bool = False, server: bool = True, capture_content: bool = True
    ) -> bool:
        """Configure process-wide MCP tracing, independently for each side.

        Repeated calls update defaults without adding hooks or duplicate spans.
        :param client: Enable outgoing client round trips.
        :param server: Enable incoming server requests and notifications.
        :param capture_content: Log arguments, results, and notification
            parameters. When False, spans keep names, timing, allowlisted
            metadata, and error classification only.
        """
        if not cls.install_patchers(target=target):
            return False
        set_global_client_config(ClientConfig(capture_content=capture_content) if client else None)
        set_global_server_config(MCPServerConfig(capture_content=capture_content) if server else None)
        return True

    @classmethod
    def install_patchers(cls, *, target: Any | None = None) -> bool:
        """Install the class-level hooks without enabling tracing by itself."""
        return super().setup(target=target)
