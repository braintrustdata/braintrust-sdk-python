"""Explicit, spec-compliant batch tracing helpers for Anthropic Message Batches.

These helpers never call Anthropic APIs, never poll, and never fetch results.
The caller owns all provider API calls and passes the resulting batch objects,
input requests, and result entries or JSONL streams to Braintrust.
"""

import hashlib
import io
import json
import logging
import math
import tempfile
import time
import uuid
from collections.abc import AsyncIterable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from braintrust._instrumentation import provider_content_allowed
from braintrust.bt_json import bt_dumps
from braintrust.integrations.anthropic._utils import _try_to_dict, extract_anthropic_usage
from braintrust.integrations.anthropic.tracing import (
    _INSTRUMENTATION,
    _get_input_from_kwargs,
    _get_metadata_from_kwargs,
    _message_output,
    _usage_served_service_tier,
)
from braintrust.integrations.utils import _service_tier_metadata
from braintrust.logger import (
    NOOP_SPAN,
    _internal_start_span_with_initial_merge,
    get_span_parent_object,
    update_span,
)
from braintrust.span_identifier_v4 import SpanComponentsV4
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


log = logging.getLogger(__name__)

_CONTEXT_VERSION = 1
_BATCH_SPAN_NAME = "anthropic.batch"
_CHILD_SPAN_NAME = "anthropic.messages.create"
_VALID_RESULT_TYPES = frozenset({"succeeded", "errored", "canceled", "expired"})
_REQUEST_COUNT_FIELDS = ("processing", "succeeded", "errored", "canceled", "expired")


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _deterministic_digest(*parts: str) -> bytes:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.digest()


def _digest_uuid(digest: bytes) -> str:
    return str(uuid.UUID(bytes=digest[:16], version=4))


def _digest_hex(digest: bytes, num_bytes: int) -> str:
    return digest[:num_bytes].hex()


def _request_ids_digest(custom_ids: Iterable[str]) -> str:
    return _deterministic_digest(*sorted(custom_ids)).hex()


def _parse_timestamp(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        ts = dt.timestamp()
        return ts if math.isfinite(ts) else None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts if math.isfinite(ts) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
            return ts if math.isfinite(ts) else None
        except ValueError:
            return None
    return None


def _extract_request_counts(batch: Any) -> dict[str, int] | None:
    raw_counts = _get_field(batch, "request_counts")
    if raw_counts is None:
        return None
    counts: dict[str, int] = {}
    for field in _REQUEST_COUNT_FIELDS:
        val = _get_field(raw_counts, field)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            counts[field] = val
    return counts or None


def _iter_jsonl_lines(stream: Iterable[str | bytes]) -> Iterable[Any]:
    for raw_line in stream:
        if isinstance(raw_line, (bytes, bytearray, memoryview)):
            line = bytes(raw_line).decode("utf-8").strip()
        elif isinstance(raw_line, str):
            line = raw_line.strip()
        else:
            raise TypeError(f"Unsupported JSONL line type: {type(raw_line).__name__}")
        if not line:
            continue
        yield json.loads(line)


def _iter_records(source: Any) -> Iterable[Any]:
    if isinstance(source, str):
        yield from _iter_jsonl_lines(io.StringIO(source))
        return
    if isinstance(source, (bytes, bytearray, memoryview)):
        yield from _iter_jsonl_lines(io.TextIOWrapper(io.BytesIO(bytes(source)), encoding="utf-8"))
        return
    if isinstance(source, io.IOBase):
        yield from _iter_jsonl_lines(source)
        return
    if isinstance(source, Iterable) and not isinstance(source, Mapping):
        for item in source:
            if isinstance(item, str):
                stripped = item.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                stripped = bytes(item).decode("utf-8").strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
            else:
                yield item
        return
    raise TypeError(f"Unsupported Anthropic batch record source: {type(source).__name__}")


async def _aiter_records(source: Any) -> AsyncIterable[Any]:
    if isinstance(source, AsyncIterable):
        async for item in source:
            if isinstance(item, str):
                stripped = item.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                stripped = bytes(item).decode("utf-8").strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
            else:
                yield item
        return
    for item in _iter_records(source):
        yield item


def _strip_propagated_event(slug: str) -> str:
    components = SpanComponentsV4.from_str(slug)
    return SpanComponentsV4(
        object_type=components.object_type,
        object_id=components.object_id,
        compute_object_metadata_args=components.compute_object_metadata_args,
        row_id=components.row_id,
        span_id=components.span_id,
        root_span_id=components.root_span_id,
    ).to_str()


def _task_ids(batch_id: str) -> tuple[str, str, str]:
    digest = _deterministic_digest("anthropic:batch", batch_id)
    return _digest_uuid(digest), _digest_hex(digest, 8), _digest_hex(digest, 16)


def _child_ids(batch_id: str, custom_id: str) -> tuple[str, str]:
    digest = _deterministic_digest("anthropic:batch:request", batch_id, custom_id)
    return _digest_uuid(digest), _digest_hex(digest, 8)


def _child_slug(task_components: SpanComponentsV4, batch_id: str, custom_id: str) -> str:
    row_id, span_id = _child_ids(batch_id, custom_id)
    return SpanComponentsV4(
        object_type=task_components.object_type,
        object_id=task_components.object_id,
        compute_object_metadata_args=task_components.compute_object_metadata_args,
        row_id=row_id,
        span_id=span_id,
        root_span_id=task_components.root_span_id,
    ).to_str()


def _extract_error_message(error_obj: Any, default: str) -> str:
    if error_obj is None:
        return default
    if isinstance(error_obj, str):
        return error_obj or default
    inner = _get_field(error_obj, "error")
    target = inner if inner is not None else error_obj
    err_type = _get_field(target, "type")
    err_msg = _get_field(target, "message")
    if isinstance(err_type, str) and err_type and isinstance(err_msg, str) and err_msg:
        return f"{err_type}: {err_msg}"
    if isinstance(err_msg, str) and err_msg:
        return err_msg
    if isinstance(err_type, str) and err_type:
        return err_type
    return default


def _build_succeeded_update(message: Any, end_time: float) -> dict[str, Any]:
    usage = _get_field(message, "usage")
    metrics, metadata = extract_anthropic_usage(usage)
    served_service_tier = _usage_served_service_tier(usage)
    if served_service_tier is not None:
        metadata["served_service_tier"] = served_service_tier

    model = _get_field(message, "model")
    if isinstance(model, str) and model:
        metadata["model"] = model

    diagnostics = _get_field(message, "diagnostics")
    cache_miss_reason = _get_field(diagnostics, "cache_miss_reason") if diagnostics is not None else None
    reason_type = _get_field(cache_miss_reason, "type") if cache_miss_reason is not None else None
    if reason_type is not None:
        metadata["cache_miss_reason"] = reason_type
    cache_missed_input_tokens = (
        _get_field(cache_miss_reason, "cache_missed_input_tokens") if cache_miss_reason is not None else None
    )
    if is_numeric(cache_missed_input_tokens):
        metadata["cache_missed_input_tokens"] = cache_missed_input_tokens

    metrics["end"] = end_time
    output = _message_output(message)
    event: dict[str, Any] = {"metrics": metrics}
    if output is not None:
        event["output"] = output
    if metadata:
        event["metadata"] = metadata
    return event


def _parse_context(context: str, expected_batch_id: str) -> dict[str, Any] | None:
    if not isinstance(context, str) or not context:
        return None
    try:
        parsed = json.loads(context)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("version") != _CONTEXT_VERSION:
        return None
    if parsed.get("batch_id") != expected_batch_id:
        return None

    start_time = parsed.get("start_time")
    if not isinstance(start_time, (int, float)) or isinstance(start_time, bool) or not math.isfinite(start_time):
        return None

    request_count = parsed.get("request_count")
    if not isinstance(request_count, int) or isinstance(request_count, bool) or request_count <= 0:
        return None

    digest = parsed.get("request_ids_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    try:
        bytes.fromhex(digest)
    except ValueError:
        return None

    task_slug = parsed.get("task")
    if not isinstance(task_slug, str) or not task_slug:
        return None
    try:
        task_components = SpanComponentsV4.from_str(task_slug)
    except Exception:
        return None

    expected_row_id, expected_span_id, _ = _task_ids(expected_batch_id)
    if (
        task_components.row_id != expected_row_id
        or task_components.span_id != expected_span_id
        or not task_components.root_span_id
    ):
        return None

    return {
        "batch_id": expected_batch_id,
        "task_slug": task_slug,
        "task_components": task_components,
        "start_time": float(start_time),
        "request_count": request_count,
        "request_ids_digest": digest,
    }


def _start_batch_trace_impl(batch: Any, requests: Any) -> tuple[str, set[str]] | None:
    batch_id = _get_field(batch, "id")
    if not isinstance(batch_id, str) or not batch_id:
        log.debug("Anthropic batch trace skipped: missing batch id")
        return None

    parent = get_span_parent_object()
    if parent == NOOP_SPAN:
        return None
    parent_slug = parent.to_str() if isinstance(parent, SpanComponentsV4) else parent.export()
    if not parent_slug:
        return None

    parent_components = SpanComponentsV4.from_str(parent_slug)
    start_time = _parse_timestamp(_get_field(batch, "created_at"))
    if start_time is None:
        start_time = time.time()

    task_row_id, task_span_id, task_root_span_id = _task_ids(batch_id)
    task_metadata: dict[str, Any] = {
        "provider": "anthropic",
        "batch_id": batch_id,
    }
    processing_status = _get_field(batch, "processing_status")
    if isinstance(processing_status, str) and processing_status:
        task_metadata["processing_status"] = processing_status
    request_counts = _extract_request_counts(batch)
    if request_counts is not None:
        task_metadata["request_counts"] = request_counts

    task_span = _internal_start_span_with_initial_merge(
        _BATCH_SPAN_NAME,
        parent=parent_slug,
        span_id=task_span_id,
        root_span_id=None if parent_components.span_id else task_root_span_id,
        start_time=start_time,
        type=SpanTypeAttribute.TASK,
        internal={"instrumentation": _INSTRUMENTATION},
        id=task_row_id,
        metadata=task_metadata,
    )
    if task_span == NOOP_SPAN:
        return None

    task_slug = task_span.export()
    seen_ids: set[str] = set()

    for request in _iter_records(requests):
        custom_id = _get_field(request, "custom_id")
        if not isinstance(custom_id, str) or not custom_id or custom_id in seen_ids:
            raise ValueError("Invalid or duplicate custom_id in Anthropic batch requests")

        raw_params = _get_field(request, "params")
        params = _try_to_dict(raw_params)
        if not isinstance(params, dict) or not isinstance(params.get("model"), str) or not params["model"]:
            raise ValueError(f"Invalid params in Anthropic batch request {custom_id!r}")

        child_row_id, child_span_id = _child_ids(batch_id, custom_id)
        child_input = _get_input_from_kwargs(params)
        child_metadata = _get_metadata_from_kwargs(params)
        child_metadata.update(_service_tier_metadata(requested=params.get("service_tier")))
        child_metadata["batch_id"] = batch_id
        child_metadata["custom_id"] = custom_id

        _internal_start_span_with_initial_merge(
            _CHILD_SPAN_NAME,
            parent=task_slug,
            span_id=child_span_id,
            start_time=start_time,
            type=SpanTypeAttribute.LLM,
            internal={"instrumentation": _INSTRUMENTATION},
            id=child_row_id,
            input=child_input,
            metadata=child_metadata,
        )
        seen_ids.add(custom_id)

    if not seen_ids:
        raise ValueError("Anthropic batch requests must not be empty")

    context_payload = {
        "version": _CONTEXT_VERSION,
        "batch_id": batch_id,
        "task": _strip_propagated_event(task_slug),
        "start_time": start_time,
        "request_count": len(seen_ids),
        "request_ids_digest": _request_ids_digest(seen_ids),
    }
    return json.dumps(context_payload, separators=(",", ":")), seen_ids


def _prepare_completion(
    batch: Any,
    context: str | None,
    requests: Any | None,
) -> tuple[dict[str, Any], set[str] | None, float] | None:
    batch_id = _get_field(batch, "id")
    if not isinstance(batch_id, str) or not batch_id:
        log.debug("Anthropic batch completion skipped: missing batch id")
        return None

    processing_status = _get_field(batch, "processing_status")
    if processing_status != "ended":
        return None

    known_ids: set[str] | None = None
    parsed: dict[str, Any] | None = None
    if context is not None:
        parsed = _parse_context(context, batch_id)
        if parsed is None:
            log.debug("Invalid or mismatched Anthropic batch trace context for batch %s", batch_id)
            if requests is None:
                return None

    if parsed is None:
        if requests is None:
            log.debug("Anthropic batch completion skipped: neither valid context nor requests provided")
            return None
        started = _start_batch_trace_impl(batch, requests)
        if started is None:
            return None
        created_context, known_ids = started
        parsed = _parse_context(created_context, batch_id)
        if parsed is None:
            return None

    ended_at = _parse_timestamp(_get_field(batch, "ended_at"))
    end_time = max(parsed["start_time"], ended_at if ended_at is not None else time.time())
    return parsed, known_ids, end_time


def _apply_provider_privacy(event: dict[str, Any]) -> dict[str, Any]:
    policy = provider_content_allowed.get()
    if policy is None or policy():
        return event
    redacted = {k: v for k, v in event.items() if k not in ("input", "output", "error")}
    metadata = redacted.get("metadata")
    if isinstance(metadata, dict):
        redacted["metadata"] = {k: metadata[k] for k in ("model", "provider") if k in metadata}
    return redacted


def _build_child_update_event(
    entry: Any,
    *,
    batch_id: str,
    task_components: SpanComponentsV4,
    end_time: float,
    seen_ids: set[str],
    known_ids: set[str] | None,
    expected_count: int,
) -> tuple[str, dict[str, Any]] | None:
    custom_id = _get_field(entry, "custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        log.debug("Anthropic batch completion paused: malformed result missing custom_id")
        return None
    if custom_id in seen_ids:
        log.debug("Anthropic batch completion paused: duplicate custom_id %r", custom_id)
        return None
    if known_ids is not None and custom_id not in known_ids:
        log.debug("Anthropic batch completion paused: unknown custom_id %r", custom_id)
        return None
    if len(seen_ids) >= expected_count:
        log.debug("Anthropic batch completion paused: result count exceeds started request count (%d)", expected_count)
        return None

    result = _get_field(entry, "result")
    result_type = _get_field(result, "type")
    if result_type not in _VALID_RESULT_TYPES:
        log.debug("Anthropic batch completion paused: invalid result type %r for %r", result_type, custom_id)
        return None

    exported = _child_slug(task_components, batch_id, custom_id)
    if result_type == "succeeded":
        message = _get_field(result, "message")
        if message is None:
            log.debug("Anthropic batch completion paused: missing message for succeeded result %r", custom_id)
            return None
        event = _build_succeeded_update(message, end_time)
    elif result_type == "errored":
        error_msg = _extract_error_message(_get_field(result, "error"), "Anthropic batch request errored")
        event = {"error": error_msg, "metrics": {"end": end_time}}
    elif result_type == "canceled":
        event = {"error": "Anthropic batch request canceled", "metrics": {"end": end_time}}
    else:
        event = {"error": "Anthropic batch request expired", "metrics": {"end": end_time}}

    seen_ids.add(custom_id)
    return exported, _apply_provider_privacy(event)


_SPOOL_MAX_BYTES = 1024 * 1024


def _finalize_task_span(
    batch: Any,
    parsed: dict[str, Any],
    seen_ids: set[str],
    end_time: float,
    staged_updates: "tempfile.SpooledTemporaryFile[str] | None" = None,
) -> None:
    if len(seen_ids) != parsed["request_count"] or _request_ids_digest(seen_ids) != parsed["request_ids_digest"]:
        log.debug(
            "Anthropic batch %s results incomplete or mismatched (%d/%d); leaving task pending",
            parsed["batch_id"],
            len(seen_ids),
            parsed["request_count"],
        )
        return

    request_counts = _extract_request_counts(batch)
    if request_counts is not None:
        if request_counts.get("processing", 0) != 0:
            log.debug("Anthropic batch %s still has processing requests; leaving task pending", parsed["batch_id"])
            return
        total = sum(request_counts.get(k, 0) for k in ("succeeded", "errored", "canceled", "expired"))
        if total != parsed["request_count"]:
            log.debug(
                "Anthropic batch %s request_counts total (%d) != started count (%d); leaving task pending",
                parsed["batch_id"],
                total,
                parsed["request_count"],
            )
            return

    if staged_updates is not None:
        staged_updates.seek(0)
        for line in staged_updates:
            exported, child_event = json.loads(line)
            update_span(exported=exported, **child_event)

    task_metadata: dict[str, Any] = {
        "provider": "anthropic",
        "batch_id": parsed["batch_id"],
        "processing_status": "ended",
    }
    if request_counts is not None:
        task_metadata["request_counts"] = request_counts

    task_error: str | None = None
    if request_counts is not None and request_counts.get("succeeded", 0) == 0:
        if (
            request_counts.get("canceled", 0) > 0
            and request_counts.get("errored", 0) == 0
            and request_counts.get("expired", 0) == 0
        ):
            task_error = "Anthropic batch canceled"
        elif (
            request_counts.get("expired", 0) > 0
            and request_counts.get("errored", 0) == 0
            and request_counts.get("canceled", 0) == 0
        ):
            task_error = "Anthropic batch expired"

    event: dict[str, Any] = {
        "metrics": {"end": end_time},
        "metadata": task_metadata,
    }
    if task_error is not None:
        event["error"] = task_error
    update_span(exported=parsed["task_slug"], **_apply_provider_privacy(event))


def start_anthropic_batch_trace(
    *,
    batch: Any,
    requests: Any,
) -> str | None:
    """Start pending Braintrust spans after Anthropic accepts a Message Batch.

    Does not call Anthropic APIs. Creates one pending ``anthropic.batch`` task
    span and one pending ``anthropic.messages.create`` child LLM span per
    request. Persist the returned opaque context string with the batch and pass
    it to :func:`complete_anthropic_batch_trace` (or
    :func:`complete_anthropic_batch_trace_async`) when results are collected,
    including from another process.

    Flush the logger before handing off work to another process.
    """
    try:
        started = _start_batch_trace_impl(batch, requests)
        return started[0] if started is not None else None
    except Exception as e:
        log.debug("Could not start Anthropic batch trace: %s", e)
        return None


def complete_anthropic_batch_trace(
    *,
    batch: Any,
    results: Any | None = None,
    context: str | None = None,
    requests: Any | None = None,
) -> None:
    """Complete a Braintrust Anthropic Message Batch trace with collected results.

    Does not poll Anthropic or fetch results. Accepts already-obtained result
    entries as a sync iterable (such as ``client.messages.batches.results(...)``
    or a list of result objects/dicts), a JSONL string/bytes payload, or a
    file-like stream.

    Non-terminal batches and batches with incomplete or mismatched results
    remain pending and retryable. Repeating completion with the same context
    and results updates the same spans idempotently.

    If submission was not traced, omit ``context`` and supply ``requests`` to
    capture the entire batch in collect-only mode under the active parent.
    """
    try:
        prepared = _prepare_completion(batch, context, requests)
        if prepared is None:
            return
        parsed, known_ids, end_time = prepared
        if results is None:
            log.debug("Anthropic batch %s has no results supplied; leaving task pending", parsed["batch_id"])
            return

        seen_ids: set[str] = set()
        staged_updates = (
            tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_BYTES, mode="w+", encoding="utf-8")
            if known_ids is None
            else None
        )
        try:
            for entry in _iter_records(results):
                built = _build_child_update_event(
                    entry,
                    batch_id=parsed["batch_id"],
                    task_components=parsed["task_components"],
                    end_time=end_time,
                    seen_ids=seen_ids,
                    known_ids=known_ids,
                    expected_count=parsed["request_count"],
                )
                if built is None:
                    return
                exported, child_event = built
                if staged_updates is not None:
                    staged_updates.write(bt_dumps([exported, child_event], separators=(",", ":")) + "\n")
                else:
                    update_span(exported=exported, **child_event)

            _finalize_task_span(batch, parsed, seen_ids, end_time, staged_updates=staged_updates)
        finally:
            if staged_updates is not None:
                staged_updates.close()
    except Exception as e:
        log.debug("Could not complete Anthropic batch trace: %s", e)


async def complete_anthropic_batch_trace_async(
    *,
    batch: Any,
    results: Any | None = None,
    context: str | None = None,
    requests: Any | None = None,
) -> None:
    """Async variant of :func:`complete_anthropic_batch_trace` for async result streams.

    Streams an ``AsyncIterable`` (such as ``await async_client.messages.batches.results(...)``)
    incrementally without buffering all results in memory. Also accepts any
    sync iterable, JSONL string/bytes payload, or file-like stream.
    """
    try:
        prepared = _prepare_completion(batch, context, requests)
        if prepared is None:
            return
        parsed, known_ids, end_time = prepared
        if results is None:
            log.debug("Anthropic batch %s has no results supplied; leaving task pending", parsed["batch_id"])
            return

        seen_ids: set[str] = set()
        staged_updates = (
            tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_BYTES, mode="w+", encoding="utf-8")
            if known_ids is None
            else None
        )
        try:
            async for entry in _aiter_records(results):
                built = _build_child_update_event(
                    entry,
                    batch_id=parsed["batch_id"],
                    task_components=parsed["task_components"],
                    end_time=end_time,
                    seen_ids=seen_ids,
                    known_ids=known_ids,
                    expected_count=parsed["request_count"],
                )
                if built is None:
                    return
                exported, child_event = built
                if staged_updates is not None:
                    staged_updates.write(bt_dumps([exported, child_event], separators=(",", ":")) + "\n")
                else:
                    update_span(exported=exported, **child_event)

            _finalize_task_span(batch, parsed, seen_ids, end_time, staged_updates=staged_updates)
        finally:
            if staged_updates is not None:
                staged_updates.close()
    except Exception as e:
        log.debug("Could not complete Anthropic batch trace: %s", e)
