from braintrust.integrations.base import FunctionWrapperPatcher

from . import _server_v1, _server_v2


_MCP_V1 = "<2.0.0a0"
_MCP_V2 = ">=2.0.0a0,<3"


# -- MCP 1.x ------------------------------------------------------------------


class ServerSessionInitPatcher(FunctionWrapperPatcher):
    """Observe traced sessions' streams from receipt through response sending."""

    name = "mcp.server.v1.session_init"
    version_spec = _MCP_V1
    target_module = "mcp.server.session"
    target_path = "ServerSession.__init__"
    wrapper = _server_v1.server_session_init_wrapper


class ServerHandleMessagePatcher(FunctionWrapperPatcher):
    """Make the operation span current while request and notification handlers run."""

    name = "mcp.server.v1.handle_message"
    version_spec = _MCP_V1
    target_module = "mcp.server.lowlevel.server"
    target_path = "Server._handle_message"
    wrapper = _server_v1.server_handle_message_wrapper


class ServerRunPatcher(FunctionWrapperPatcher):
    """Per-server ``run`` hook used by ``wrap_mcp_server`` (never applied globally)."""

    name = "mcp.server.v1.run"
    version_spec = _MCP_V1
    target_path = "run"
    wrapper = _server_v1.server_run_wrapper


# -- MCP 2.x ------------------------------------------------------------------


class ServerRunnerRequestPatcher(FunctionWrapperPatcher):
    name = "mcp.server.v2.runner_on_request"
    version_spec = _MCP_V2
    target_module = "mcp.server.runner"
    target_path = "ServerRunner._on_request"
    wrapper = _server_v2.runner_on_request_wrapper


class ServerRunnerNotifyPatcher(FunctionWrapperPatcher):
    name = "mcp.server.v2.runner_on_notify"
    version_spec = _MCP_V2
    target_module = "mcp.server.runner"
    target_path = "ServerRunner._on_notify"
    wrapper = _server_v2.runner_on_notify_wrapper


class DispatcherHandleRequestPatcher(FunctionWrapperPatcher):
    """Keep dispatcher-owned operations open until the response write settles."""

    name = "mcp.server.v2.dispatcher_handle_request"
    version_spec = _MCP_V2
    target_module = "mcp.shared.jsonrpc_dispatcher"
    target_path = "JSONRPCDispatcher._handle_request"
    wrapper = _server_v2.dispatcher_handle_request_wrapper


class DispatcherWriteResultPatcher(FunctionWrapperPatcher):
    name = "mcp.server.v2.dispatcher_write_result"
    version_spec = _MCP_V2
    target_module = "mcp.shared.jsonrpc_dispatcher"
    target_path = "JSONRPCDispatcher._write_result"
    wrapper = _server_v2.dispatcher_write_result_wrapper


class DispatcherWriteErrorPatcher(FunctionWrapperPatcher):
    name = "mcp.server.v2.dispatcher_write_error"
    version_spec = _MCP_V2
    target_module = "mcp.shared.jsonrpc_dispatcher"
    target_path = "JSONRPCDispatcher._write_error"
    wrapper = _server_v2.dispatcher_write_error_wrapper


class DispatcherNotifyPatcher(FunctionWrapperPatcher):
    """Trace notifications sent by dispatchers that serve a traced server."""

    name = "mcp.server.v2.dispatcher_notify"
    version_spec = _MCP_V2
    target_module = "mcp.shared.jsonrpc_dispatcher"
    target_path = "JSONRPCDispatcher.notify"
    wrapper = _server_v2.dispatcher_notify_wrapper


SERVER_PATCHERS = (
    ServerSessionInitPatcher,
    ServerHandleMessagePatcher,
    ServerRunnerRequestPatcher,
    ServerRunnerNotifyPatcher,
    DispatcherHandleRequestPatcher,
    DispatcherWriteResultPatcher,
    DispatcherWriteErrorPatcher,
    DispatcherNotifyPatcher,
)
