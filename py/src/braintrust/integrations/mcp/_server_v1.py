"""Server request lifecycle hooks for MCP 1.x.

MCP 1.x validates requests inside ``BaseSession._receive_loop`` and writes
validation errors straight to the session's write stream, so handler-level
hooks cannot see receipt, validation failures, or response sending. Instead,
each traced ``ServerSession`` gets thin observers around its read stream,
write stream, and incoming-message queue:

- the read observer starts an operation when the transport delivers a parsed
  request or notification, before validation;
- the incoming-queue observer links a validated notification to its operation;
- ``Server._handle_message`` makes the operation current while handlers run;
- the write observer records the result or error and ends the operation once
  the transport accepted the response.

The observers only read messages; they never change what MCP sends or receives.
"""

import contextvars
import functools
import weakref
from typing import Any

from .tracing import (
    SERVER_CONFIG_ATTR,
    ConnectionClosedError,
    MCPServerConfig,
    ServerOperation,
    activate,
    call_arg,
    contain,
    end_outgoing_notification,
    global_server_config,
    http_session_id,
    http_transport_metadata,
    related_operation,
    run_operation,
    start_outgoing_notification,
    start_server_operation,
)


# Set while a wrapped ``Server.run`` constructs its session.
_pending_server_config: contextvars.ContextVar[MCPServerConfig | None] = contextvars.ContextVar(
    "braintrust_mcp_pending_server_config", default=None
)

_session_traces: "weakref.WeakKeyDictionary[Any, _SessionTrace]" = weakref.WeakKeyDictionary()


@functools.cache
def _mcp() -> Any:
    """MCP 1.x types, imported on first use so this module loads without mcp."""
    from mcp import types
    from mcp.shared.session import RequestResponder  # pylint: disable=import-error,no-name-in-module  # MCP 1.x

    return types, RequestResponder


class _SessionTrace:
    """Pending operations for one server session (one MCP connection)."""

    def __init__(self, session: Any, config: MCPServerConfig):
        self._session_ref = weakref.ref(session)
        self.config = config
        self.pending: dict[Any, ServerOperation] = {}
        self.forwarded_notifications: dict[int, ServerOperation] = {}
        self.inbound_notification: ServerOperation | None = None
        self.protocol_version: str | None = None

    def _session_attr(self, name: str) -> Any:
        session = self._session_ref()
        return getattr(session, name, None) if session is not None else None

    def settle_inbound(self) -> None:
        """End a notification the receive loop handled without forwarding it."""
        operation, self.inbound_notification = self.inbound_notification, None
        if operation is not None:
            operation.finish()

    def on_receive(self, item: Any) -> None:
        types, _ = _mcp()
        message = getattr(getattr(item, "message", None), "root", None)
        is_request = isinstance(message, types.JSONRPCRequest)
        if not is_request and not isinstance(message, types.JSONRPCNotification):
            # Responses to server-initiated requests are not operations.
            return
        request_context = getattr(item.metadata, "request_context", None)
        params = message.params
        operation = start_server_operation(
            method=message.method,
            params=params,
            request_id=message.id if is_request else None,
            config=self.config,
            session_id=http_session_id(request_context),
            protocol_version=self.protocol_version,
            client_info=getattr(self._session_attr("_client_params"), "clientInfo", None),
            transport_metadata=http_transport_metadata(request_context),
            jsonrpc_version=message.jsonrpc,
        )
        if operation is None:
            return
        if is_request:
            self.pending[message.id] = operation
            return
        if message.method == "notifications/cancelled" and isinstance(params, dict):
            # The session cancels only requests that are still in flight.
            target_id = params.get("requestId")
            target = self.pending.get(target_id)
            if target is not None and target_id in (self._session_attr("_in_flight") or {}):
                target.mark_cancelled()
        self.inbound_notification = operation

    def on_forward(self, message: Any) -> None:
        types, _ = _mcp()
        if self.inbound_notification is not None and isinstance(message, types.ClientNotification):
            self.forwarded_notifications[id(message)] = self.inbound_notification
            self.inbound_notification = None

    def on_send(self, item: Any) -> tuple[ServerOperation | None, Any]:
        types, _ = _mcp()
        message = getattr(getattr(item, "message", None), "root", None)
        if isinstance(message, types.JSONRPCResponse):
            operation = self.pending.pop(message.id, None)
            if operation is not None:
                version = message.result.get("protocolVersion") if operation.method == "initialize" else None
                if isinstance(version, str):
                    self.protocol_version = version
                operation.record_result(message.result)
            return operation, None
        if isinstance(message, types.JSONRPCError):
            operation = self.pending.pop(message.id, None)
            if operation is not None:
                operation.record_error_response(message.error.code, message.error.message)
            return operation, None
        if isinstance(message, types.JSONRPCNotification):
            related_id = getattr(item.metadata, "related_request_id", None)
            span = start_outgoing_notification(
                method=message.method,
                params=message.params,
                config=self.config,
                related=related_operation(related_id) or self.pending.get(related_id),
            )
            return None, span
        return None, None

    def close(self) -> None:
        operations = [*self.pending.values(), *self.forwarded_notifications.values()]
        if self.inbound_notification is not None:
            operations.append(self.inbound_notification)
        self.pending.clear()
        self.forwarded_notifications.clear()
        self.inbound_notification = None
        for operation in operations:
            operation.finish(ConnectionClosedError())


class _StreamObserver:
    """Delegates to an anyio memory stream; subclasses observe its messages."""

    def __init__(self, stream: Any, trace: _SessionTrace):
        self._stream = stream
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> "_StreamObserver":
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._stream.__aexit__(*exc_info)


class _ObservedReadStream(_StreamObserver):
    async def __aexit__(self, *exc_info: Any) -> Any:
        try:
            return await self._stream.__aexit__(*exc_info)
        finally:
            contain(self._trace.close)

    def __aiter__(self) -> "_ObservedReadStream":
        return self

    async def __anext__(self) -> Any:
        contain(self._trace.settle_inbound)
        try:
            item = await self._stream.__anext__()
        except BaseException:
            contain(self._trace.close)
            raise
        contain(self._trace.on_receive, item)
        return item

    async def receive(self) -> Any:
        return await self.__anext__()


class _ObservedWriteStream(_StreamObserver):
    async def send(self, item: Any) -> None:
        operation, notification_span = contain(self._trace.on_send, item) or (None, None)

        def settle(error: BaseException | None = None) -> None:
            if operation is not None:
                operation.finish(error)
            if notification_span is not None:
                end_outgoing_notification(notification_span, error)

        try:
            await self._stream.send(item)
        except BaseException as error:
            settle(error)
            raise
        settle()


class _ObservedIncomingWriter(_StreamObserver):
    async def send(self, item: Any) -> None:
        contain(self._trace.on_forward, item)
        await self._stream.send(item)


def _install_session_observers(session: Any, config: MCPServerConfig) -> None:
    if session in _session_traces:
        return
    streams = ("_read_stream", "_write_stream", "_incoming_message_stream_writer")
    if not all(hasattr(session, name) for name in streams):
        return
    trace = _SessionTrace(session, config)
    session._read_stream = _ObservedReadStream(session._read_stream, trace)
    session._write_stream = _ObservedWriteStream(session._write_stream, trace)
    session._incoming_message_stream_writer = _ObservedIncomingWriter(session._incoming_message_stream_writer, trace)
    _session_traces[session] = trace


def server_session_init_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    result = wrapped(*args, **kwargs)
    config = _pending_server_config.get() or global_server_config()
    if config is not None:
        # A session consumes the config of the run that created it, so a
        # server started from inside a handler is not traced implicitly.
        _pending_server_config.set(None)
        contain(_install_session_observers, instance, config)
    return result


async def server_handle_message_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    message = call_arg(args, kwargs, 0, "message")
    session = call_arg(args, kwargs, 1, "session")
    trace = _session_traces.get(session) if session is not None else None
    if trace is None:
        return await wrapped(*args, **kwargs)
    if isinstance(message, _mcp()[1]):
        operation = trace.pending.get(message.request_id)
        if operation is None:
            return await wrapped(*args, **kwargs)
        # The write observer ends the operation once the response is sent.
        with activate(operation):
            return await wrapped(*args, **kwargs)
    operation = trace.forwarded_notifications.pop(id(message), None)
    if operation is None:
        return await wrapped(*args, **kwargs)
    return await run_operation(operation, wrapped, args, kwargs)


async def server_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """Per-server ``run`` wrapper installed by ``wrap_mcp_server`` on MCP 1.x.

    Per instance rather than on ``Server.run``, because fastmcp 2/3 override
    ``run`` without calling it.
    """
    config = getattr(instance, SERVER_CONFIG_ATTR, None)
    token = _pending_server_config.set(config if isinstance(config, MCPServerConfig) else None)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _pending_server_config.reset(token)
