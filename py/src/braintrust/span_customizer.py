"""Synchronous transformations of native export records."""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from .db_fields import (
    ARRAY_DELETE_FIELD,
    ID_FIELD,
    IS_MERGE_FIELD,
    MERGE_PATHS_FIELD,
    OBJECT_DELETE_FIELD,
    OBJECT_ID_KEYS,
    PARENT_ID_FIELD,
    TRANSACTION_ID_FIELD,
)


__all__ = ["SpanCustomizer", "SpanExportData", "set_span_customizers"]

SpanExportData = dict[str, Any]
"""A native, possibly incremental span export record, not a live span."""


class SpanCustomizer:
    """Extensible hooks for instrumentation-created spans. Omitted hooks are no-ops."""

    def on_span_export(self, data: SpanExportData) -> SpanExportData:
        """Return the original or a replacement export record, synchronously.

        Hooks run in registration order after lazy values resolve, before merging,
        attachment processing, masking, and serialization. Identity, routing, and
        merge protocol fields are restored after each hook. Exceptions and invalid
        returns fail open: in-place edits survive and later hooks still run.
        """
        return data


_span_customizers: tuple[SpanCustomizer, ...] = ()
_PROTECTED_FIELDS = frozenset(
    (
        ID_FIELD,
        "span_id",
        "root_span_id",
        "span_parents",
        "org_id",
        *OBJECT_ID_KEYS,
        IS_MERGE_FIELD,
        MERGE_PATHS_FIELD,
        PARENT_ID_FIELD,
        OBJECT_DELETE_FIELD,
        ARRAY_DELETE_FIELD,
        TRANSACTION_ID_FIELD,
    )
)

# Tags and record-level errors are intentionally outside the masking contract.
_MASKING_FIELDS = ("input", "output", "expected", "metadata", "context", "scores", "metrics")


class _MaskingCustomizer(SpanCustomizer):
    """Adapt field-level masking to a logger-local hook on all merged records."""

    def __init__(self, masking_function: Callable[[Any], Any]):
        self._masking_function = masking_function

    def on_span_export(self, data: SpanExportData) -> SpanExportData:
        for field in _MASKING_FIELDS:
            if field not in data:
                continue
            try:
                data[field] = self._masking_function(data[field])
            except Exception as error:
                # Fail closed per field, without leaking exception messages or stacks.
                message = f"ERROR: Failed to mask field '{field}' - {type(error).__name__}"
                if field in ("scores", "metrics"):
                    del data[field]
                    data["error"] = f"{data['error']}; {message}" if "error" in data else message
                else:
                    data[field] = {"error": message} if field == "metadata" else message
        return data


def set_span_customizers(customizers: Sequence[SpanCustomizer] | None) -> None:
    """Replace the process-wide ordered customizer list with a snapshot.

    Configure before making instrumented calls; pass None or an empty sequence to
    disable. The sequence is copied, but customizer instances are not. Each record
    uses the configuration active when it is logged, and retries reuse its
    transformed data. There is no environment-variable registration.

    Raises TypeError for classes, objects without a callable ``on_span_export``,
    and async hooks, since those would otherwise silently export unredacted data.
    """
    global _span_customizers
    snapshot = tuple(customizers) if customizers is not None else ()
    for customizer in snapshot:
        _validate_customizer(customizer)
    _span_customizers = snapshot


def _get_span_customizers() -> tuple[SpanCustomizer, ...]:
    return _span_customizers


def _validate_customizer(customizer: Any) -> None:
    # Reject misconfiguration eagerly: at export time these would silently fail
    # open and ship unredacted data.
    if isinstance(customizer, type):
        raise TypeError(f"Span customizers must be instances, not classes; did you mean {customizer.__name__}()?")
    hook = getattr(customizer, "on_span_export", None)
    if not callable(hook):
        raise TypeError(
            f"Span customizer {type(customizer).__name__} must define a callable on_span_export(data) method"
        )
    if inspect.iscoroutinefunction(hook):
        raise TypeError(f"{type(customizer).__name__}.on_span_export must be synchronous")


def _copy_protocol_value(value: Any) -> Any:
    # Copy only protocol containers. Payloads (especially Attachment objects) must
    # retain their existing serialization behavior and must not be deep-copied.
    if isinstance(value, list):
        return [_copy_protocol_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_protocol_value(item) for key, item in value.items()}
    return value


def _restore_protocol_fields(data: SpanExportData, protected: SpanExportData) -> SpanExportData:
    restored = {key: value for key, value in data.items() if key not in _PROTECTED_FIELDS}
    # Never expose the snapshot itself: a later hook may mutate nested arrays.
    restored.update({key: _copy_protocol_value(value) for key, value in protected.items()})
    return restored


def _customize_span_export(
    data: SpanExportData, customizers: Sequence[SpanCustomizer] | None = None
) -> SpanExportData:
    if customizers is None:
        customizers = _span_customizers
    if not customizers:
        return data

    protected = None
    for customizer in customizers:
        candidate = data
        try:
            hook = getattr(customizer, "on_span_export", None)
            if hook is None:
                continue
            if protected is None:
                protected = {key: _copy_protocol_value(data[key]) for key in _PROTECTED_FIELDS if key in data}
                # Protocol containers may alias span state; don't expose those to hooks.
                data = _restore_protocol_fields(data, protected)
                candidate = data
            result = hook(data)
            if inspect.isawaitable(result):
                # Do not execute asynchronous hooks or emit unawaited coroutine warnings.
                if inspect.iscoroutine(result):
                    result.close()
            elif type(result) is dict:
                candidate = result
        except Exception:
            # Deliberately fail open, retaining in-place changes. Never log the
            # exception, since its message may itself contain sensitive payloads.
            pass

        if protected is not None:
            data = _restore_protocol_fields(candidate, protected)
    return data
