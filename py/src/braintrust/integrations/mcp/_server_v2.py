"""Server request lifecycle hooks for MCP 2.x.

Every MCP 2.x server path (stdio, streamable HTTP in both protocol eras, and
in-process clients) dispatches through ``ServerRunner._on_request`` and
``ServerRunner._on_notify``, which validate, dispatch, run the handler, and
shape the wire result. Operations start there.

When a ``JSONRPCDispatcher`` owns the request, it writes the response after the
runner returns. Its ``_handle_request`` hook keeps the operation open through
the response write, observes the written error code, and detects requests the
peer cancelled (which the dispatcher settles without a response). Without a
dispatcher (single-exchange HTTP, in-process direct dispatch) the operation
ends when the runner returns.
"""

import contextvars
import weakref
from dataclasses import dataclass
from typing import Any

import anyio

from .tracing import (
    MCPServerConfig,
    ServerOperation,
    activate,
    call_arg,
    contain,
    end_outgoing_notification,
    http_session_id,
    http_transport_metadata,
    related_operation,
    run_operation,
    server_config,
    start_outgoing_notification,
    start_server_operation,
)


@dataclass
class _DispatchSlot:
    """Links a dispatcher-owned request to the operation the runner starts."""

    dispatcher: Any
    operation: ServerOperation | None = None
    written: bool = False


_dispatch_slot: contextvars.ContextVar[_DispatchSlot | None] = contextvars.ContextVar(
    "braintrust_mcp_dispatch_slot", default=None
)

# Dispatchers that serve a traced server, for outgoing notifications.
_server_dispatchers: "weakref.WeakKeyDictionary[Any, MCPServerConfig]" = weakref.WeakKeyDictionary()


def _cancel_requested(dctx: Any) -> bool:
    event = getattr(dctx, "cancel_requested", None)
    try:
        return bool(event is not None and event.is_set())
    except Exception:
        return False


def _start(runner: Any, args: Any, kwargs: Any, config: MCPServerConfig, notification: bool) -> ServerOperation | None:
    dctx = call_arg(args, kwargs, 0, "dctx")
    connection = getattr(runner, "connection", None)
    request_context = getattr(getattr(dctx, "message_metadata", None), "request_context", None)
    return start_server_operation(
        method=call_arg(args, kwargs, 1, "method"),
        params=call_arg(args, kwargs, 2, "params"),
        request_id=None if notification else getattr(dctx, "request_id", None),
        config=config,
        session_id=getattr(connection, "session_id", None) or http_session_id(request_context),
        protocol_version=getattr(connection, "protocol_version", None),
        client_info=getattr(getattr(connection, "client_params", None), "client_info", None),
        transport_metadata=http_transport_metadata(request_context),
    )


def _record_handler_error(operation: ServerOperation, dctx: Any, error: BaseException) -> None:
    if isinstance(error, anyio.get_cancelled_exc_class()) and _cancel_requested(dctx):
        operation.mark_cancelled()
        return
    error_data = None
    if isinstance(error, Exception):
        from mcp.shared.jsonrpc_dispatcher import handler_exception_to_error_data

        error_data = contain(handler_exception_to_error_data, error)
    if error_data is not None:
        operation.record_error_response(error_data.code, error_data.message)
    else:
        operation.record_exception(error)


async def runner_on_request_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    slot = _dispatch_slot.get()
    if slot is not None and slot.operation is not None:
        # Another layer already traces this request.
        return await wrapped(*args, **kwargs)
    config = server_config(getattr(instance, "server", None))
    operation = contain(_start, instance, args, kwargs, config, False) if config is not None else None
    if operation is None:
        return await wrapped(*args, **kwargs)
    dctx = call_arg(args, kwargs, 0, "dctx")
    dispatcher_owned = slot is not None
    if dispatcher_owned:
        slot.operation = operation
        if _server_dispatchers.get(slot.dispatcher) is not config:
            _server_dispatchers[slot.dispatcher] = config
    try:
        with activate(operation):
            result = await wrapped(*args, **kwargs)
    except BaseException as error:
        _record_handler_error(operation, dctx, error)
        if not dispatcher_owned:
            operation.finish()
        raise
    operation.record_result(result)
    if not dispatcher_owned:
        if _cancel_requested(dctx):
            operation.mark_cancelled()
        operation.finish()
    return result


async def runner_on_notify_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    config = server_config(getattr(instance, "server", None))
    operation = contain(_start, instance, args, kwargs, config, True) if config is not None else None
    if operation is None:
        return await wrapped(*args, **kwargs)
    return await run_operation(operation, wrapped, args, kwargs)


async def dispatcher_handle_request_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    slot = _DispatchSlot(dispatcher=instance)
    token = _dispatch_slot.set(slot)
    error = None
    try:
        return await wrapped(*args, **kwargs)
    except BaseException as caught:
        error = caught
        raise
    finally:
        _dispatch_slot.reset(token)
        operation = slot.operation
        if operation is not None:
            # A request the peer cancelled settles without a response.
            if not slot.written and _cancel_requested(call_arg(args, kwargs, 1, "dctx")):
                operation.mark_cancelled()
            operation.finish(error)


def _slot_for(request_id: Any) -> _DispatchSlot | None:
    slot = _dispatch_slot.get()
    if slot is None or slot.operation is None or slot.operation.request_id != request_id:
        return None
    return slot


async def dispatcher_write_result_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    slot = _slot_for(call_arg(args, kwargs, 0, "request_id"))
    if slot is not None:
        slot.written = True
    return await wrapped(*args, **kwargs)


async def dispatcher_write_error_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    slot = _slot_for(call_arg(args, kwargs, 0, "request_id"))
    if slot is not None:
        slot.written = True
        error = call_arg(args, kwargs, 1, "error")
        slot.operation.record_error_response(getattr(error, "code", None), getattr(error, "message", None))
    return await wrapped(*args, **kwargs)


async def dispatcher_notify_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    config = _server_dispatchers.get(instance)
    span = None
    if config is not None:
        span = contain(
            lambda: start_outgoing_notification(
                method=call_arg(args, kwargs, 0, "method"),
                params=call_arg(args, kwargs, 1, "params"),
                config=config,
                related=related_operation(kwargs.get("_related_request_id")),
            )
        )
    if span is None:
        return await wrapped(*args, **kwargs)
    try:
        result = await wrapped(*args, **kwargs)
    except BaseException as error:
        end_outgoing_notification(span, error)
        raise
    end_outgoing_notification(span)
    return result
