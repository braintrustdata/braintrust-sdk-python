"""Detached OTel export snapshots for the shared span customizer hook."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

from braintrust.span_customizer import _log_customizer_failure, _run_span_export_customizers
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, StatusCode, TraceFlags, TraceState
from opentelemetry.trace.span import _is_valid_pair


try:
    from opentelemetry.exporter.otlp.proto.common._internal import _encode_key_value
except ImportError:
    # OpenTelemetry 1.16 predates the shared HTTP/gRPC encoder package.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter.encoder import _encode_key_value


def _encodable(key, value):
    try:
        _encode_key_value(key, value)
    except Exception:
        return False
    return True


# Older encoders can't send some values their SDK keeps (e.g. None inside arrays)
# and skip those attributes on export. Drop them before hooks run so only values a
# hook introduces fail closed. Newer encoders accept everything the SDK keeps.
_DROP_UNENCODABLE = not _encodable("probe", [None])


def _snapshot_attributes(attributes):
    attributes = _copy_value(attributes or {})
    if _DROP_UNENCODABLE:
        return {key: value for key, value in attributes.items() if _encodable(key, value)}
    return attributes


class _ExportSpan(SimpleNamespace):
    def get_span_context(self):
        return self.context


def _metadata_key(value):
    """Hash nested metadata without conflating OTLP scalar types."""
    if isinstance(value, dict):
        return (dict, frozenset((key, _metadata_key(item)) for key, item in value.items()))
    if isinstance(value, list):
        return (list, tuple(_metadata_key(item) for item in value))
    if isinstance(value, float):
        return (float, value.hex())
    return (type(value), value)


# OTLP groups resources and scopes by these keys. Keep detached attributes
# losslessly, without SDK constructors that can sanitize or log their contents.
@dataclass(frozen=True)
class _ExportResource:
    attributes: dict = field(compare=False)
    schema_url: str | None
    _attributes_key: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_attributes_key", _metadata_key(self.attributes))


@dataclass(frozen=True)
class _ExportScope:
    name: str
    version: str | None
    schema_url: str | None
    attributes: dict = field(compare=False)
    _attributes_key: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_attributes_key", _metadata_key(self.attributes))


def _copy_value(value):
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float, bytes):
        return value
    # SDK attributes may retain primitive subclasses (for example IntEnum).
    # Use the base implementations to detach them without invoking overrides.
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, bytes):
        # bytes.__bytes__ is unavailable on Python 3.10.
        return bytes.__getitem__(value, slice(None))
    raise TypeError("Invalid OTel snapshot value")


def _context_data(context):
    if context is None:
        return None
    return {
        "trace_id": context.trace_id,
        "span_id": context.span_id,
        "is_remote": context.is_remote,
        "trace_flags": int(context.trace_flags),
        "trace_state": dict(context.trace_state.items()),
    }


def _snapshot(span):
    scope = span.instrumentation_scope
    return {
        "name": span.name,
        "context": _context_data(span.context),
        "parent": _context_data(span.parent),
        "start_time": span.start_time,
        "end_time": span.end_time,
        "kind": span.kind.value,
        "attributes": _snapshot_attributes(span.attributes),
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": _snapshot_attributes(event.attributes),
                "dropped_attributes": getattr(event, "dropped_attributes", getattr(event.attributes, "dropped", 0)),
            }
            for event in span.events
        ],
        "links": [
            {
                "context": _context_data(link.context),
                "attributes": _snapshot_attributes(link.attributes),
                "dropped_attributes": getattr(link, "dropped_attributes", getattr(link.attributes, "dropped", 0)),
            }
            for link in span.links
        ],
        "status": {"status_code": span.status.status_code.value, "description": span.status.description},
        "resource": _resource_data(span.resource),
        "instrumentation_scope": _scope_data(scope),
        "dropped_attributes": span.dropped_attributes,
        "dropped_events": span.dropped_events,
        "dropped_links": span.dropped_links,
    }


def _resource_data(resource):
    return {"attributes": _snapshot_attributes(resource.attributes), "schema_url": resource.schema_url}


def _scope_data(scope):
    if not scope:
        return None
    return {
        "name": scope.name,
        "version": scope.version,
        "schema_url": scope.schema_url,
        "attributes": _snapshot_attributes(getattr(scope, "attributes", None)),
    }


def _string(value, optional=False):
    if value is None and optional:
        return value
    if type(value) is not str:
        raise TypeError("Expected an OTel string")
    value.encode("utf-8")
    return value


def _integer(value, bits, signed=False):
    if type(value) is not int:
        raise TypeError("Expected an OTel integer")
    low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1)) if signed else (0, 2**bits)
    if not low <= value < high:
        raise ValueError("OTel integer out of range")
    return value


def _attributes(data):
    if not isinstance(data, dict):
        raise TypeError("OTel attributes must be a dict")
    for key, value in data.items():
        if type(key) is not str:
            raise TypeError("OTel attribute keys must be strings")
        # The installed encoder defines which values are valid. Its attribute
        # encoding logs and skips failures, so encode directly to fail closed.
        _encode_key_value(key, value)
    return data


def _check_context(data):
    if not isinstance(data, dict):
        raise TypeError("OTel links require a span context")
    state = data.get("trace_state", {})
    if not isinstance(state, dict) or len(state) > 32:
        raise TypeError("Invalid OTel trace state")
    for key, value in state.items():
        # TraceState logs invalid pairs, so reject them before construction.
        if type(key) is not str or type(value) is not str or not _is_valid_pair(key, value):
            raise ValueError("Invalid OTel trace state")
    _integer(data["trace_id"], 128)
    _integer(data["span_id"], 64)
    _integer(data.get("trace_flags", 0), 8)
    if type(data.get("is_remote", False)) is not bool:
        raise TypeError("Invalid OTel span context")


def _check_scope(data):
    if data is None:
        return
    _string(data.get("name", ""))
    _string(data.get("version"), optional=True)
    _string(data.get("schema_url", ""), optional=True)
    if _attributes(data.get("attributes", {})) and not hasattr(InstrumentationScope, "attributes"):
        raise ValueError("Scope attributes require a newer OpenTelemetry SDK")


def _check_export_data(data):
    """Validate a hook result against what the OTLP encoder accepts, without encoding the span."""
    _string(data.get("name", ""))
    SpanKind(data.get("kind", 0))
    _attributes(data.get("attributes", {}))
    for event in data.get("events", []):
        _string(event["name"])
        _integer(event["timestamp"], 64)
        _attributes(event.get("attributes", {}))
        _integer(event.get("dropped_attributes", 0), 32)
    for link in data.get("links", []):
        _check_context(link["context"])
        _attributes(link.get("attributes", {}))
        _integer(link.get("dropped_attributes", 0), 32)
    status = data.get("status", {})
    StatusCode(status.get("status_code", 0))
    _string(status.get("description"), optional=True)
    resource = data.get("resource", {})
    _attributes(resource.get("attributes", {}))
    _string(resource.get("schema_url", ""), optional=True)
    _check_scope(data.get("instrumentation_scope"))
    for key in ("dropped_attributes", "dropped_events", "dropped_links"):
        _integer(data.get(key, 0), 32)


def _context(data):
    if data is None:
        return None
    return SpanContext(
        trace_id=data["trace_id"],
        span_id=data["span_id"],
        is_remote=data.get("is_remote", False),
        trace_flags=TraceFlags(data.get("trace_flags", 0)),
        trace_state=TraceState(list(data.get("trace_state", {}).items())),
    )


def _scope(data):
    if data is None:
        return None
    return _ExportScope(
        name=data.get("name", ""),
        version=data.get("version"),
        schema_url=data.get("schema_url", ""),
        attributes=data.get("attributes", {}),
    )


def _export_span(data):
    status = data.get("status", {})
    return _ExportSpan(
        name=data.get("name", ""),
        context=_context(data["context"]),
        parent=_context(data["parent"]),
        start_time=data["start_time"],
        end_time=data["end_time"],
        kind=SpanKind(data.get("kind", 0)),
        attributes=data.get("attributes", {}),
        events=[
            SimpleNamespace(
                name=event["name"],
                timestamp=event["timestamp"],
                attributes=event.get("attributes", {}),
                dropped_attributes=event.get("dropped_attributes", 0),
            )
            for event in data.get("events", [])
        ],
        links=[
            SimpleNamespace(
                context=_context(link["context"]),
                attributes=link.get("attributes", {}),
                dropped_attributes=link.get("dropped_attributes", 0),
            )
            for link in data.get("links", [])
        ],
        status=SimpleNamespace(
            status_code=StatusCode(status.get("status_code", 0)), description=status.get("description")
        ),
        resource=_ExportResource(
            attributes=data.get("resource", {}).get("attributes", {}),
            schema_url=data.get("resource", {}).get("schema_url", ""),
        ),
        instrumentation_scope=_scope(data.get("instrumentation_scope")),
        dropped_attributes=data.get("dropped_attributes", 0),
        dropped_events=data.get("dropped_events", 0),
        dropped_links=data.get("dropped_links", 0),
    )


def customize_otel_span(span, customizers):
    """Customize exactly once at the Braintrust exporter, never other consumers."""
    protected = {
        "context": _context_data(span.context),
        "parent": _context_data(span.parent),
        "start_time": span.start_time,
        "end_time": span.end_time,
    }
    routing = {key: value for key, value in (span.attributes or {}).items() if key == "braintrust.parent"}

    def stripped_record():
        stripped = deepcopy(protected)
        for field in ("context", "parent"):
            if stripped[field] is not None:
                stripped[field]["trace_state"] = {}
        stripped["attributes"] = {**routing, "braintrust.context_json": '{"customizer_error":true}'}
        return stripped

    def stripped_span():
        _log_customizer_failure()
        return _export_span(stripped_record())

    def restore(result):
        result = _copy_value(result)
        result.update(deepcopy(protected))
        attributes = result.setdefault("attributes", {})
        if not isinstance(attributes, dict):
            raise TypeError("OTel attributes must be a dict")
        attributes.pop("braintrust.parent", None)
        attributes.update(routing)
        # Validate every hook result before the next hook runs.
        _check_export_data(result)
        return result

    try:
        data = _snapshot(span)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # Unexportable source data fails closed for this span, not the batch.
        return stripped_span()

    data = _run_span_export_customizers(data, customizers, restore, stripped_record)
    try:
        return _export_span(data)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return stripped_span()
