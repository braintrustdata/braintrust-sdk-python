"""Version-independent span shaping for MCP client and server operations.

The version-specific lifecycle hooks (``_server_v1`` / ``_server_v2``) observe
when a request arrives, which result or error is sent, and when sending
finishes. They feed plain JSON-RPC values (method, params dict, result dict,
error code) into the helpers here, so span names, payloads, metadata, and
error classification are identical across MCP 1.x and 2.x.
"""

import contextlib
import contextvars
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from braintrust import logger as _bt_logger
from braintrust.context import BraintrustContextManager
from braintrust.integrations.utils import _resolved_attachment_from_base64
from braintrust.logger import NOOP_SPAN, Span, extract_trace_context, start_span
from braintrust.span_types import SpanTypeAttribute


logger = logging.getLogger(__name__)

_INSTRUMENTATION = "mcp-auto"
_INTERNAL = {"instrumentation": _INSTRUMENTATION}

# OTel MCP server conventions classify these JSON-RPC errors as caller errors.
_CALLER_ERROR_CODES = frozenset({-32700, -32600, -32601, -32602, -32002})
_CONTENT_METHODS = frozenset({"tools/call", "prompts/get", "resources/read"})
# Result keys whose items are protocol content (may carry ``_meta`` or inline media).
_CONTENT_KEYS = ("content", "contents", "messages")
_RESOURCE_URI_METHODS = frozenset(
    {"resources/read", "resources/subscribe", "resources/unsubscribe", "notifications/resources/updated"}
)
_TRACE_CONTEXT_KEYS = ("traceparent", "tracestate", "baggage")
_PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
_CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
_SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"
_RELATED_TASK_META_KEY = "io.modelcontextprotocol/related-task"
_TOOL_ERROR_FALLBACK = "Tool returned an error result"
_CONNECTION_CLOSED_MESSAGE = "MCP connection closed before the response completed"
_MAX_CONTENT_DEPTH = 128
_DROP = object()


@dataclass(frozen=True)
class MCPServerConfig:
    """Per-server tracing settings."""

    capture_content: bool = True


# Per-server config set by ``wrap_mcp_server`` on the low-level server.
SERVER_CONFIG_ATTR = "__braintrust_mcp_server_config__"

_global_server_config: MCPServerConfig | None = None


def set_global_server_config(config: MCPServerConfig | None) -> None:
    global _global_server_config
    _global_server_config = config


def global_server_config() -> MCPServerConfig | None:
    return _global_server_config


def server_config(server: Any) -> MCPServerConfig | None:
    """The config ``wrap_mcp_server`` set on *server*, else the global one."""
    config = getattr(server, SERVER_CONFIG_ATTR, None)
    return config if isinstance(config, MCPServerConfig) else _global_server_config


class ConnectionClosedError(ConnectionError):
    """Recorded when a connection ends while an operation is still pending."""

    def __init__(self) -> None:
        super().__init__(_CONNECTION_CLOSED_MESSAGE)


def contain(callback: Any, *args: Any) -> Any:
    """Run instrumentation code without letting its failures reach MCP."""
    try:
        return callback(*args)
    except Exception:
        logger.debug("MCP instrumentation failed", exc_info=True)
        return None


def call_arg(args: Any, kwargs: Any, index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


@dataclass
class ServerOperation:
    """Telemetry state for one incoming request or notification.

    The outcome is collected here and logged once, when the operation ends
    after the response was sent. Only the span is shared with the
    application, through the current-span context.
    """

    span: Span
    method: str
    request_id: str | int | None
    capture_content: bool
    ended: bool = False
    cancelled: bool = False
    result: Mapping[str, Any] | None = None
    status_code: str | None = None
    failure: Any = None
    error_type: str | None = None
    client: bool = False

    @property
    def captures_output(self) -> bool:
        return self.capture_content and self.method in _CONTENT_METHODS

    def record_result(self, result: Any) -> None:
        """Keep the JSON-RPC result; it is shaped when the operation ends."""
        if not self.ended and isinstance(result, Mapping):
            self.result = result

    def record_error_response(self, code: Any, message: Any) -> None:
        """Classify a JSON-RPC error response."""
        if self.ended:
            return
        self.status_code = str(code)
        if self.cancelled:
            return
        # The response classifies the outcome, replacing a handler exception
        # the library already converted into this error.
        self.failure = None
        self.error_type = None
        if not self.client and isinstance(code, int) and code in _CALLER_ERROR_CODES:
            return
        # Error messages can echo application data.
        detail = message if self.capture_content and isinstance(message, str) and message else None
        self.failure = f"MCP error {code}: {detail}" if detail else f"MCP error {code}"
        self.error_type = str(code)

    def record_exception(self, error: BaseException) -> None:
        """Record a failure that may not be followed by an error response."""
        if not self.ended and self.failure is None:
            self.failure = error
            self.error_type = type(error).__name__

    def mark_cancelled(self) -> None:
        if not self.ended:
            self.cancelled = True

    def finish(self, error: BaseException | None = None) -> None:
        """Log the outcome and end the span once; later calls are ignored."""
        if self.ended:
            return
        self.ended = True
        event: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        result = self.result
        if result is not None:
            metadata.update(protocol_meta_metadata(result.get("_meta"), "response"))
            if self.method in ("initialize", "server/discover"):
                if isinstance(result.get("protocolVersion"), str):
                    metadata["mcp.protocol.version"] = result["protocolVersion"]
                metadata.update(identity_metadata(result.get("serverInfo"), "server"))
            if self.captures_output:
                event["output"] = sanitize_result(result)
            if self.method == "tools/call" and result.get("isError") is True and self.failure is None:
                self.failure = tool_error_message(result, self.capture_content)
                self.error_type = "tool_error"
        if self.status_code is not None:
            metadata["rpc.response.status_code"] = self.status_code
        if error is not None and self.failure is None:
            self.failure = error
            self.error_type = type(error).__name__
        if self.cancelled:
            metadata["mcp.cancelled"] = True
        elif self.failure is not None:
            # Without content capture, exceptions keep only their type.
            redact = not self.capture_content and isinstance(self.failure, BaseException)
            event["error"] = type(self.failure).__name__ if redact else self.failure
            metadata["error.type"] = self.error_type or "_OTHER"
        if metadata:
            event["metadata"] = metadata
        if event:
            self.span.log(**event)
        self.span.end()


_active_operation: contextvars.ContextVar[ServerOperation | None] = contextvars.ContextVar(
    "braintrust_mcp_server_operation", default=None
)


@contextlib.contextmanager
def activate(operation: ServerOperation) -> Iterator[None]:
    """Make *operation* the current MCP operation and Braintrust span."""
    operation_token = _active_operation.set(operation)
    span_token = _bt_logger._state.context_manager.set_current_span(operation.span)
    try:
        yield
    finally:
        _bt_logger._state.context_manager.unset_current_span(span_token)
        _active_operation.reset(operation_token)


async def run_operation(operation: ServerOperation, wrapped: Any, args: Any, kwargs: Any) -> Any:
    """Run *wrapped* as *operation* and end the operation when it returns."""
    try:
        with activate(operation):
            result = await wrapped(*args, **kwargs)
    except BaseException as error:
        operation.finish(error)
        raise
    operation.finish()
    return result


def related_operation(request_id: Any) -> ServerOperation | None:
    """The running request an outgoing notification belongs to.

    Notifications are sent from the request's handler, where it is the active
    operation. Libraries may re-expose the ID as a string (``ctx.request_id``).
    """
    operation = _active_operation.get()
    if operation is None or operation.ended or request_id is None:
        return None
    if operation.request_id == request_id or str(operation.request_id) == str(request_id):
        return operation
    return None


def _meta_of(params: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = params.get("_meta")
    return meta if isinstance(meta, Mapping) else {}


def _is_task_message(method: str, params: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    """Task submission and polling have a separate lifecycle and are not traced."""
    return (
        method.startswith("tasks/") or params.get("task") is not None or meta.get(_RELATED_TASK_META_KEY) is not None
    )


def _start_operation(
    *,
    method: str,
    params: Any,
    request_id: Any,
    config: MCPServerConfig,
    session_id: str | None = None,
    protocol_version: str | None = None,
    client_info: Any = None,
    transport_metadata: Mapping[str, Any] | None = None,
    jsonrpc_version: Any = "2.0",
    client: bool = False,
) -> ServerOperation | None:
    """Shape one request or server notification, with side-specific parentage."""
    if not isinstance(method, str) or (request_id is not None and not _is_request_id(request_id)):
        return None
    params = params if isinstance(params, Mapping) else {}
    meta = _meta_of(params)
    if _is_task_message(method, params, meta):
        return None
    is_notification = request_id is None and not client
    is_tool_call = method == "tools/call" and not is_notification
    target = params.get("name") if method in ("tools/call", "prompts/get") and not is_notification else None
    target = target if isinstance(target, str) and target else None

    metadata: dict[str, Any] = {"mcp.method.name": method}
    if client:
        metadata["mcp.span.kind"] = "client"
    if request_id is not None:
        metadata["jsonrpc.request.id"] = str(request_id)
    if session_id:
        metadata["mcp.session.id"] = session_id
    if is_tool_call:
        metadata["gen_ai.operation.name"] = "execute_tool"
        if target:
            metadata["gen_ai.tool.name"] = target
    if method == "prompts/get" and target:
        metadata["gen_ai.prompt.name"] = target
    if method in _RESOURCE_URI_METHODS and isinstance(params.get("uri"), str):
        metadata["mcp.resource.uri"] = params["uri"]
    if protocol_version:
        metadata["mcp.protocol.version"] = protocol_version
    if isinstance(jsonrpc_version, str) and jsonrpc_version != "2.0":
        metadata["jsonrpc.protocol.version"] = jsonrpc_version
    if transport_metadata:
        metadata.update(transport_metadata)
    metadata.update(identity_metadata(client_info, "client"))
    if method == "initialize":
        metadata.update(identity_metadata(params.get("clientInfo"), "client"))
        if isinstance(params.get("protocolVersion"), str):
            metadata["mcp.protocol.version"] = params["protocolVersion"]
    metadata.update(protocol_meta_metadata(meta, "request"))

    event: dict[str, Any] = {"metadata": metadata}
    if config.capture_content:
        payload = _operation_input(method, params, is_notification)
        if payload is not None:
            event["input"] = payload

    span_kwargs = dict(
        name=f"{method} {target}" if target else method,
        type=SpanTypeAttribute.TOOL if is_tool_call else SpanTypeAttribute.FUNCTION,
        **event,
    )
    span = (
        start_span(set_current=False, internal=_INTERNAL, **span_kwargs)
        if client
        else start_detached_span(parent=extract_trace_context(_trace_carrier(meta)), **span_kwargs)
    )
    return ServerOperation(
        span=span, method=method, request_id=request_id, capture_content=config.capture_content, client=client
    )


def start_server_operation(**kwargs: Any) -> ServerOperation | None:
    """Start an incoming operation using only the propagated parent."""
    return _start_operation(**kwargs)


def start_client_operation(**kwargs: Any) -> ServerOperation | None:
    """Start an outgoing request under the caller's current span."""
    return _start_operation(client=True, **kwargs)


def start_outgoing_notification(
    *, method: Any, params: Any, config: MCPServerConfig, related: ServerOperation | None
) -> Span | None:
    """Start the span for a notification the server sends."""
    if not isinstance(method, str):
        return None
    params = params if isinstance(params, Mapping) else {}
    meta = _meta_of(params)
    if _is_task_message(method, params, meta):
        return None
    metadata: dict[str, Any] = {"mcp.method.name": method}
    if method in _RESOURCE_URI_METHODS and isinstance(params.get("uri"), str):
        metadata["mcp.resource.uri"] = params["uri"]
    event: dict[str, Any] = {"metadata": metadata}
    if config.capture_content:
        payload = notification_content(method, params)
        if payload:
            event["output"] = payload
    if related is not None and not related.ended:
        return related.span.start_span(name=method, type=SpanTypeAttribute.FUNCTION, internal=_INTERNAL, **event)
    return start_detached_span(
        name=method,
        type=SpanTypeAttribute.FUNCTION,
        parent=extract_trace_context(_trace_carrier(meta)),
        **event,
    )


def end_outgoing_notification(span: Span, error: BaseException | None = None) -> None:
    if error is not None:
        span.log(error=error, metadata={"error.type": type(error).__name__})
    span.end()


def start_detached_span(*, parent: Any, **kwargs: Any) -> Span:
    """Start a span whose parent is only the propagated MCP context.

    One transport request can carry several MCP operations, so the ambient
    context (an HTTP server span, or the caller's span for in-process servers)
    must not become the parent. Without propagated context this starts a new
    trace under the active logger or experiment.
    """
    state = _bt_logger._state
    context_manager = state.context_manager
    otel_context = None
    if isinstance(context_manager, BraintrustContextManager):
        span_token = context_manager.set_current_span(None)
    else:
        from opentelemetry import context as otel_context

        span_token = otel_context.attach(otel_context.Context())
    parent_token = state.current_parent.set(None)
    try:
        return start_span(parent=parent, set_current=False, internal=_INTERNAL, **kwargs)
    except Exception:
        logger.debug("Failed to start MCP span", exc_info=True)
        return NOOP_SPAN
    finally:
        state.current_parent.reset(parent_token)
        if otel_context is not None:
            otel_context.detach(span_token)
        else:
            context_manager.unset_current_span(span_token)


def _trace_carrier(meta: Mapping[str, Any]) -> dict[str, str]:
    return {key: meta[key] for key in _TRACE_CONTEXT_KEYS if isinstance(meta.get(key), str)}


def _is_request_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _operation_input(method: str, params: Mapping[str, Any], is_notification: bool) -> Any:
    if is_notification:
        return notification_content(method, params) or None
    if method in ("tools/call", "prompts/get"):
        # Arguments are application data; log them as received.
        return params.get("arguments")
    if method == "resources/read" and "uri" in params:
        return {"uri": params["uri"]}
    return None


def notification_content(method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Notification parameters without ``_meta`` or a free-form cancellation reason."""
    omit = ("_meta", "reason") if method == "notifications/cancelled" else ("_meta",)
    if not any(key in params for key in omit):
        return params
    return {key: value for key, value in params.items() if key not in omit}


def identity_metadata(info: Any, side: str) -> dict[str, str]:
    """Translate self-reported client or server identity."""
    if info is None:
        return {}
    result: dict[str, str] = {}
    for key in ("name", "title", "version"):
        value = info.get(key) if isinstance(info, Mapping) else getattr(info, key, None)
        if isinstance(value, str):
            result[f"mcp.{side}.{key}"] = value
    return result


def protocol_meta_metadata(meta: Any, direction: str) -> dict[str, str]:
    """Translate the allowlisted ``_meta`` keys; everything else is omitted."""
    if not isinstance(meta, Mapping):
        return {}
    metadata: dict[str, str] = {}
    if direction == "request":
        if isinstance(meta.get(_PROTOCOL_VERSION_META_KEY), str):
            metadata["mcp.protocol.version"] = meta[_PROTOCOL_VERSION_META_KEY]
        metadata.update(identity_metadata(meta.get(_CLIENT_INFO_META_KEY), "client"))
    else:
        metadata.update(identity_metadata(meta.get(_SERVER_INFO_META_KEY), "server"))
    return metadata


def http_transport_metadata(request: Any) -> dict[str, Any]:
    """Read network attributes from an ASGI request attached to a message."""
    scope = getattr(request, "scope", None)
    if not isinstance(scope, Mapping) or scope.get("type") != "http":
        return {}
    metadata: dict[str, Any] = {"network.protocol.name": "http"}
    http_version = scope.get("http_version")
    if isinstance(http_version, str):
        metadata["network.protocol.version"] = http_version
        metadata["network.transport"] = "quic" if http_version.startswith("3") else "tcp"
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and len(client) == 2 and isinstance(client[0], str):
        metadata["client.address"] = client[0]
        if isinstance(client[1], int):
            metadata["client.port"] = client[1]
    return metadata


def http_session_id(request: Any) -> str | None:
    headers = getattr(request, "headers", None)
    try:
        value = headers.get("mcp-session-id") if headers is not None else None
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def tool_error_message(result: Mapping[str, Any], capture_content: bool) -> str:
    if capture_content and isinstance(result.get("content"), list):
        texts = [
            block["text"]
            for block in result["content"]
            if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)
    return _TOOL_ERROR_FALLBACK


def sanitize_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prepare a tool, prompt, or resource result for logging without serializing it.

    Only the protocol content lists are walked; application data such as
    ``structuredContent`` is passed through untouched. Braintrust serializes the
    rest when the span is sent.
    """
    changed: dict[str, Any] = {}
    for key in _CONTENT_KEYS:
        if key in result:
            value = sanitize_content(result[key])
            if value is not result[key]:
                changed[key] = value
    if not changed and "_meta" not in result:
        return result
    return {key: changed.get(key, value) for key, value in result.items() if key != "_meta"}


def sanitize_content(value: Any, _depth: int = 0) -> Any:
    """Drop ``_meta`` and replace inline base64 media with attachments in place.

    Copy-on-write: containers that need no change are returned as-is, and the
    original value is never mutated.
    """
    if _depth >= _MAX_CONTENT_DEPTH:
        return value
    if isinstance(value, list):
        items = None
        for index, item in enumerate(value):
            new_item = sanitize_content(item, _depth + 1)
            if new_item is not item and items is None:
                items = value[:index]
            if items is not None:
                items.append(new_item)
        return value if items is None else items
    if not isinstance(value, Mapping):
        return value

    media_key = None
    if value.get("type") in ("image", "audio") and isinstance(value.get("data"), str):
        media_key = "data"
    elif isinstance(value.get("uri"), str) and isinstance(value.get("blob"), str):
        media_key = "blob"

    copy: dict[str, Any] | None = None
    for index, (key, item) in enumerate(value.items()):
        if key == "_meta":
            new_item: Any = _DROP
        elif key == media_key:
            new_item = _to_attachment(item, value.get("mimeType"), value.get("uri"))
        else:
            new_item = sanitize_content(item, _depth + 1)
        if new_item is not item and copy is None:
            copy = dict(list(value.items())[:index])
        if copy is not None and new_item is not _DROP:
            copy[key] = new_item
    return value if copy is None else copy


def _to_attachment(data: str, mime_type: Any, uri: Any) -> Any:
    content_type = mime_type if isinstance(mime_type, str) and mime_type else "application/octet-stream"
    filename = None
    if isinstance(uri, str):
        parts = urlsplit(uri)
        filename = f"{parts.netloc}{parts.path}".rsplit("/", 1)[-1] or None
    resolved = _resolved_attachment_from_base64(
        data, content_type, filename=filename, prefix=content_type.split("/")[0]
    )
    # Values that cannot be decoded are logged unchanged.
    return resolved.attachment if resolved is not None else data
