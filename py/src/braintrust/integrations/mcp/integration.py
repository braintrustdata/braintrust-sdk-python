from typing import Any

from braintrust.integrations.base import BaseIntegration

from .patchers import SERVER_PATCHERS
from .tracing import MCPServerConfig, set_global_server_config


class MCPIntegration(BaseIntegration):
    """Braintrust instrumentation for MCP servers on mcp>=1.28.1,<3.

    Covers low-level servers and the server abstractions built on them: the
    bundled ``FastMCP`` (1.x) / ``MCPServer`` (2.x) and standalone ``fastmcp``.
    """

    name = "mcp"
    import_names = ("mcp",)
    distribution_names = ("mcp",)
    min_version = "1.28.1"
    patchers = SERVER_PATCHERS

    @classmethod
    def setup(cls, *, target: Any | None = None, capture_content: bool = True) -> bool:
        """Trace every MCP server in the process.

        :param capture_content: Log arguments, results, and notification
            parameters. When False, spans keep names, timing, allowlisted
            metadata, and error classification only.
        """
        if not cls.install_patchers(target=target):
            return False
        set_global_server_config(MCPServerConfig(capture_content=capture_content))
        return True

    @classmethod
    def install_patchers(cls, *, target: Any | None = None) -> bool:
        """Install the class-level hooks without tracing any server by itself."""
        return super().setup(target=target)
