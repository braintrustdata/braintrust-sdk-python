"""Client hooks shared by MCP 1.x/2.x and standalone FastMCP.

ClientSession.send_request owns typed operation spans. MCP 2.x raw dispatcher
hooks also cover negotiation/custom requests that bypass that method, and defer
to it when already active. High-level client hooks copy settings to each new
session, including reconnects and initialization.

Client span contract:
* One span per eligible request round trip, through response parsing. Session
  lifetimes, one-way notifications, and detached task APIs have no client span.
* Names/types and content match the server contract: tools/call <name> is a
  tool; prompts/get <name>, resources/read, and other methods are functions.
  Discovery payloads are omitted. Content capture defaults to enabled.
* metadata.mcp.span.kind = client distinguishes the round trip from execution;
  protocol correlation/identity fields use the server's allowlist. Provenance
  is mcp-auto. Client spans do not own model/token usage.
* Parent is the active caller. Its outgoing span injects W3C traceparent,
  tracestate and Braintrust baggage in params._meta, preserving other entries.
* JSON-RPC errors (including caller errors) fail the client span. Returned tool
  errors preserve the result and never change application exception behavior.
  Runtime cancellation ends the span with mcp.cancelled and no error.
* A framework tool span may parent this distinct network round trip. An enabled
  opentelemetry-instrumentation-mcp owns client operations and propagation,
  including its own capture settings; Braintrust adds no duplicate client span.
"""

import contextvars
import sys
from collections.abc import Mapping
from typing import Any, get_args

import anyio
from braintrust import logger as bt_logger
from braintrust.logger import inject_trace_context

from .tracing import MCPServerConfig as ClientConfig
from .tracing import ServerOperation, call_arg, contain, start_client_operation


CLIENT_CONFIG_ATTR = "__braintrust_mcp_client_config__"


# Pydantic include keys are Python field names, which differ across MCP majors.
# Content-free spans need only outcome, protocol identity, and result metadata.
_RESULT_METADATA_FIELDS = frozenset(
    {"meta", "isError", "is_error", "protocolVersion", "protocol_version", "serverInfo", "server_info"}
)


_enter_config: contextvars.ContextVar[ClientConfig | None] = contextvars.ContextVar(
    "braintrust_mcp_client_enter_config", default=None
)
_outgoing_operation: contextvars.ContextVar[tuple[ServerOperation, Any] | None] = contextvars.ContextVar(
    "braintrust_mcp_client_operation", default=None
)


def upstream_owns_operations() -> bool:
    """Defer to an explicitly enabled OTel MCP instrumentor, in either setup order.

    Its client hooks own operation spans and propagation. Do not import optional
    OTel packages or change their patch chains/configuration on the user's behalf.
    """
    module = sys.modules.get("opentelemetry.instrumentation.mcp")
    instrumentor = getattr(module, "McpInstrumentor", None)
    return bool(instrumentor is not None and instrumentor().is_instrumented_by_opentelemetry)


def configure_session(client: Any, config: ClientConfig) -> None:
    try:
        session = client.session
    except (AttributeError, RuntimeError):
        # A client that has not connected has no session yet.
        return
    set_session_config(session, config)


def set_session_config(session: Any, config: ClientConfig) -> None:
    setattr(session, CLIENT_CONFIG_ATTR, config)
    dispatcher = getattr(session, "_dispatcher", None)
    if dispatcher is not None:
        setattr(dispatcher, CLIENT_CONFIG_ATTR, config)


async def client_enter_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    config = getattr(instance, CLIENT_CONFIG_ATTR, None)
    if not isinstance(config, ClientConfig):
        return await wrapped(*args, **kwargs)
    token = _enter_config.set(config)
    try:
        result = await wrapped(*args, **kwargs)
        contain(configure_session, instance, config)
        return result
    finally:
        _enter_config.reset(token)


def session_init_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    result = wrapped(*args, **kwargs)
    config = _enter_config.get()
    if config is not None:
        contain(set_session_config, instance, config)
    return result


def _prepare(session: Any, request: Any, config: ClientConfig) -> tuple[ServerOperation, Any] | None:
    data = request.model_dump(by_alias=True, mode="json", exclude_none=True)
    params = data.get("params") or {}
    operation = start_client_operation(
        method=data["method"],
        params=params,
        request_id=getattr(session, "_request_id", None),  # MCP 1.x allocates synchronously on entry.
        config=config,
        protocol_version=getattr(session, "_negotiated_version", None),
    )
    if operation is None:
        return None
    try:
        # Work on telemetry/transport copies, never the caller's protocol object.
        meta = dict(params.get("_meta") or {})
        inject_trace_context(meta, span=operation.span)
        from mcp import types

        root = getattr(request, "root", request)
        protocol_params = root.params
        if protocol_params is None:
            annotation = type(root).model_fields["params"].annotation
            params_type = next(
                (candidate for candidate in get_args(annotation) if hasattr(candidate, "model_fields")), annotation
            )
            protocol_params = params_type()
        meta_type = getattr(types, "RequestParamsMeta", None) or types.RequestParams.Meta
        # 1.x Meta is a Pydantic model; 2.x RequestParamsMeta is a TypedDict.
        protocol_params = protocol_params.model_copy(update={"meta": meta_type(**meta)})
        copied_root = root.model_copy(update={"params": protocol_params})
        copied = request.model_copy(update={"root": copied_root}) if root is not request else copied_root
    except Exception:
        # An unfamiliar extension shape must still execute unchanged.
        copied = request
    return operation, copied


def _record_exception(operation: ServerOperation, error: BaseException) -> None:
    if isinstance(error, anyio.get_cancelled_exc_class()):
        operation.mark_cancelled()
        return
    from mcp.shared import exceptions

    error_class = getattr(exceptions, "MCPError", None) or getattr(exceptions, "McpError")
    if isinstance(error, error_class):
        operation.record_error_response(error.error.code, error.error.message)
    else:
        operation.record_exception(error)


async def send_request_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    config = getattr(instance, CLIENT_CONFIG_ATTR, None)
    if not isinstance(config, ClientConfig):
        return await wrapped(*args, **kwargs)
    if upstream_owns_operations():
        # MCP 1.x inherits this method. Resolve the live base method so an OTel
        # hook installed after our subclass hook remains visible as well.
        from mcp.shared.session import BaseSession  # pylint: disable=import-error,no-name-in-module  # MCP 1.x

        return await BaseSession.send_request(instance, *args, **kwargs)
    prepared = contain(_prepare, instance, call_arg(args, kwargs, 0, "request"), config)
    if prepared is None:
        return await wrapped(*args, **kwargs)
    operation, request = prepared
    if args:
        args = (request, *args[1:])
    else:
        kwargs = {**kwargs, "request": request}
    return await _run_request(operation, getattr(instance, "_dispatcher", None), wrapped, args, kwargs)


async def _run_request(operation: ServerOperation, dispatcher: Any, wrapped: Any, args: Any, kwargs: Any) -> Any:
    token = _outgoing_operation.set((operation, dispatcher))
    manager = bt_logger._state.context_manager
    span_token = manager.set_current_span(operation.span)
    try:
        result = await wrapped(*args, **kwargs)
        include = None if operation.captures_output else _RESULT_METADATA_FIELDS
        payload = (
            result
            if isinstance(result, Mapping)
            else contain(lambda: result.model_dump(by_alias=True, mode="json", exclude_none=True, include=include))
        )
        contain(operation.record_result, payload)
        return result
    except BaseException as error:
        contain(_record_exception, operation, error)
        raise
    finally:
        manager.unset_current_span(span_token)
        _outgoing_operation.reset(token)
        contain(operation.finish)


def _prepare_raw(method: str, params: Any, config: ClientConfig) -> tuple[ServerOperation, Any] | None:
    params = params or {}
    meta = dict(params.get("_meta") or {})
    operation = start_client_operation(method=method, params=params, request_id=None, config=config)
    if operation is None:
        return None
    inject_trace_context(meta, span=operation.span)
    return operation, {**params, "_meta": meta}


async def raw_request_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    # 2.x negotiation and direct dispatcher calls can bypass send_request. The
    # typed wrapper remains the owner when it is already tracing this dispatcher.
    config = getattr(instance, CLIENT_CONFIG_ATTR, None)
    active = _outgoing_operation.get()
    if not isinstance(config, ClientConfig) or (active is not None and active[1] is instance):
        return await wrapped(*args, **kwargs)
    prepared = contain(_prepare_raw, call_arg(args, kwargs, 0, "method"), call_arg(args, kwargs, 1, "params"), config)
    if prepared is None:
        return await wrapped(*args, **kwargs)
    operation, params = prepared
    if len(args) > 1:
        args = (args[0], params, *args[2:])
    else:
        kwargs = {**kwargs, "params": params}
    return await _run_request(operation, instance, wrapped, args, kwargs)


def allocate_id_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    request_id = wrapped(*args, **kwargs)
    active = _outgoing_operation.get()
    if active is not None and active[1] is instance:
        operation = active[0]
        operation.request_id = request_id
        operation.span.log(metadata={"jsonrpc.request.id": str(request_id)})
    return request_id
