"""Synchronous transformations of native export records."""

import inspect
import logging
import threading
from collections.abc import Callable, Iterable, Sequence
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
from .env import BraintrustEnv


__all__ = ["SpanCustomizer", "SpanExportData", "set_span_customizers"]

SpanExportData = dict[str, Any]
"""A native, possibly incremental span export record, not a live span."""


class SpanCustomizer:
    """Extensible hooks for all native SDK spans. Omitted hooks are no-ops."""

    def on_span_export(self, data: SpanExportData) -> SpanExportData:
        """Return the original or a replacement export record, synchronously.

        Hooks run in registration order after lazy values resolve, before merging,
        attachment processing, masking, and serialization. Identity, routing, and
        merge protocol fields are restored after each hook. Exceptions and invalid
        synchronous returns stop the hook chain and strip the outgoing record.
        Only original identity, routing/protocol fields, created,
        span_attributes.type, and metrics.start/end survive, plus
        context={"customizer_error": True}, replacing all original context. No
        attachments from failed records are uploaded. One safe error is logged per
        process, without payloads or exception details. Handle recoverable errors
        within the hook. Only KeyboardInterrupt and SystemExit propagate,
        including while snapshotting fields. Unreadable snapshot fields are
        omitted. Dict subclasses are accepted and copied into plain dicts.

        Records may contain SDK ``Attachment`` objects: integrations convert inline
        media at capture time. Remove or replace them to prevent their upload;
        hooks that serialize data must handle them.

        Manual roots and children, instrumented spans, logger spans, and experiment
        spans are included. Dataset rows and feedback records are not. Each hook
        receives a possibly incremental record, not a completed logical span;
        fields may be absent and hooks may run before the span ends. Stripping a
        record does not retract earlier exports or disable later records. Failed
        records retain their original timing without inventing absent timestamps;
        they bypass masking and are exported separately from other updates to the
        same row.
        """
        return data


_span_customizers: tuple[SpanCustomizer, ...] = ()
_failure_logged = False
_failure_log_lock = threading.Lock()
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
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                # Fail closed per field (including GeneratorExit and CancelledError),
                # without leaking exception messages or stacks.
                message = f"ERROR: Failed to mask field '{field}' - {type(error).__name__}"
                if field in ("scores", "metrics"):
                    del data[field]
                    data["error"] = f"{data['error']}; {message}" if "error" in data else message
                else:
                    data[field] = {"error": message} if field == "metadata" else message
        return data


def set_span_customizers(customizers: Sequence[SpanCustomizer] | None) -> None:
    """Replace the process-wide ordered customizer list with a snapshot.

    Configure before logging spans; pass None or an empty sequence to disable.
    The sequence is copied, but customizer instances are not. Each span record
    uses the configuration active when it is logged, and retries reuse its
    transformed or stripped data. There is no environment-variable registration.

    Raises TypeError for classes, objects without a callable ``on_span_export``,
    and async hooks. Invalid synchronous returns are rejected at export time.
    Non-empty customizers with BRAINTRUST_OTEL_COMPAT enabled log one error per
    registration attempt and leave the existing configuration unchanged.
    """
    global _span_customizers
    snapshot = tuple(customizers) if customizers is not None else ()
    if snapshot and BraintrustEnv.OTEL_COMPAT.get(False):
        logging.getLogger("braintrust").error(
            "Span customizers are not yet supported with OTel compat mode (BRAINTRUST_OTEL_COMPAT)"
        )
        return
    for customizer in snapshot:
        _validate_customizer(customizer)
    _span_customizers = snapshot


def _get_span_customizers() -> tuple[SpanCustomizer, ...]:
    return _span_customizers


def _validate_customizer(customizer: Any) -> None:
    # Reject misconfiguration eagerly rather than stripping records at export time.
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


class _StrippedSpanExportData(dict[str, Any]):
    """SDK-only marker: do not merge or mask a memoized failed record."""


def _customize_span_export(
    data: SpanExportData, customizers: Sequence[SpanCustomizer] | None = None
) -> SpanExportData:
    if customizers is None:
        customizers = _span_customizers
    if not customizers or isinstance(data, _StrippedSpanExportData):
        return data

    # Capture identity and timing before even looking up a hook: user-defined
    # descriptors and earlier hooks can fail or mutate the record in place.
    protected = _snapshot_fields(data, _PROTECTED_FIELDS)
    # Non-content fields that keep a stripped initial record recognizable.
    preserved = _snapshot_fields(data, ("created",))
    timing = _snapshot_fields(data.get("metrics"), ("start", "end"))
    span_type = _snapshot_fields(data.get("span_attributes"), ("type",))
    try:
        # Protocol containers may alias span state; don't expose those to hooks.
        data = _restore_protocol_fields(data, protected)
        for customizer in customizers:
            hook = getattr(customizer, "on_span_export", None)
            if hook is None:
                continue
            result = hook(data)
            if inspect.isawaitable(result):
                # Do not execute asynchronous hooks or emit unawaited coroutine warnings.
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Span customizers must return a synchronous dict")
            if not isinstance(result, dict):
                raise TypeError("Span customizers must return a dict")
            # Always a fresh plain dict, even for dict subclasses.
            data = _restore_protocol_fields(result, protected)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # Memoize this stripped result in the caller's LazyValue, including
        # GeneratorExit and CancelledError, without rejecting other records.
        stripped = _StrippedSpanExportData(protected)
        stripped.update(preserved)
        if timing:
            stripped["metrics"] = timing
        if span_type:
            stripped["span_attributes"] = span_type
        stripped["context"] = {"customizer_error": True}
        _log_customizer_failure()
        return stripped
    return data


def _snapshot_fields(source: Any, keys: Iterable[str]) -> dict[str, Any]:
    # Best-effort fallback; only process-control exceptions may escape.
    snapshot: dict[str, Any] = {}
    if not isinstance(source, dict):
        return snapshot
    for key in keys:
        try:
            if key not in source:
                continue
            value = source[key]
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            continue
        try:
            snapshot[key] = _copy_protocol_value(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            # e.g. RecursionError on a self-referential container.
            snapshot[key] = value
    return snapshot


def _log_customizer_failure() -> None:
    global _failure_logged
    with _failure_log_lock:
        if _failure_logged:
            return
        _failure_logged = True
    # Neither exception details nor record data are safe to include here.
    logging.getLogger("braintrust").error("Span customizer failed; stripping outgoing span content.")
