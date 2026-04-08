"""Shared tracing utilities for Braintrust SDK integrations.

These helpers are common building blocks used across multiple provider
integrations. Keeping them here avoids duplication and makes behavioral fixes
propagate to all providers at once.

Names are prefixed with ``_`` so that consumer modules can import them
directly without aliasing (e.g. ``from braintrust.integrations.utils import
_try_to_dict``).
"""

import base64
import re
import time
import warnings
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

from braintrust.logger import Attachment, Span
from braintrust.util import is_numeric


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$")


def _try_to_dict(obj: Any) -> dict[str, Any] | Any:
    """Best-effort conversion of an SDK response object to a plain dict.

    Tries, in order:
      1. ``model_dump()`` (Pydantic v2)
      2. ``dict()``       (Pydantic v1 / legacy)
      3. returns *obj* unchanged

    Pydantic serializer warnings (common with generic/discriminated-union
    models such as OpenAI's ``ParsedResponse[T]``) are suppressed.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
                return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _camel_to_snake(value: str) -> str:
    """Convert a camelCase or PascalCase string into snake_case."""
    out = []
    for char in value:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out).lstrip("_")


def _is_supported_metric_value(value: Any) -> bool:
    """Return ``True`` for numeric metric values, excluding booleans."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _convert_data_url_to_attachment(data_url: str, filename: str | None = None) -> Attachment | str:
    """Convert a ``data:<mime>;base64,…`` URL into an :class:`Attachment`.

    Returns the original *data_url* string unchanged when it does not match
    the expected format or cannot be decoded.
    """
    match = _DATA_URL_RE.match(data_url)
    if not match:
        return data_url

    mime_type, base64_data = match.groups()

    try:
        binary_data = base64.b64decode(base64_data)
    except Exception:
        return data_url

    if filename is None:
        extension = mime_type.split("/")[1] if "/" in mime_type else "bin"
        prefix = "image" if mime_type.startswith("image/") else "file"
        filename = f"{prefix}.{extension}"

    return Attachment(data=binary_data, filename=filename, content_type=mime_type)


def _is_not_given(value: object) -> bool:
    """Return ``True`` when *value* is a provider ``NOT_GIVEN`` sentinel.

    Works by type-name inspection so that Braintrust does not need a
    direct import dependency on any provider SDK.
    """
    if value is None:
        return False
    try:
        return type(value).__name__ == "NotGiven"
    except Exception:
        return False


def _serialize_response_format(response_format: Any) -> Any:
    """Serialize a Pydantic ``BaseModel`` subclass into a JSON-schema dict.

    Non-Pydantic values pass through unchanged. Used when logging
    ``response_format`` parameters so the span metadata contains a
    readable schema rather than a Python class reference.
    """
    try:
        from pydantic import BaseModel
    except ImportError:
        return response_format

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return dict(
            type="json_schema",
            json_schema=dict(
                name=response_format.__name__,
                schema=response_format.model_json_schema(),
            ),
        )
    return response_format


def _prettify_response_params(params: dict[str, Any], *, drop_not_given: bool = False) -> dict[str, Any]:
    """Return a shallow copy of traced request params with logging-friendly values."""
    ret = params.copy()
    if drop_not_given:
        ret = {key: value for key, value in ret.items() if not _is_not_given(value)}

    if "response_format" in ret:
        ret["response_format"] = _serialize_response_format(ret["response_format"])
    return ret


def _parse_openai_usage_metrics(
    usage: Any,
    *,
    token_name_map: Mapping[str, str],
    token_prefix_map: Mapping[str, str],
) -> dict[str, Any]:
    """Parse usage payloads that follow OpenAI's ``*_tokens`` conventions."""
    metrics: dict[str, Any] = {}

    if not usage:
        return metrics

    usage = _try_to_dict(usage)
    if not isinstance(usage, dict):
        return metrics

    for name, value in usage.items():
        if name.endswith("_tokens_details"):
            if not isinstance(value, dict):
                continue
            raw_prefix = name[: -len("_tokens_details")]
            prefix = token_prefix_map.get(raw_prefix, raw_prefix)
            for nested_name, nested_value in value.items():
                if is_numeric(nested_value):
                    metrics[f"{prefix}_{nested_name}"] = nested_value
        elif is_numeric(value):
            metrics[token_name_map.get(name, name)] = value

    return metrics


def _timing_metrics(start_time: float, end_time: float, first_token_time: float | None = None) -> dict[str, float]:
    """Build a standard ``start / end / duration`` metrics dict.

    Optionally includes ``time_to_first_token`` when *first_token_time*
    is provided.
    """
    metrics: dict[str, float] = {
        "start": start_time,
        "end": end_time,
        "duration": end_time - start_time,
    }
    if first_token_time is not None:
        metrics["time_to_first_token"] = first_token_time - start_time
    return metrics


def _merge_timing_and_usage_metrics(
    start_time: float,
    usage: Any,
    usage_parser: Callable[[Any], dict[str, Any]],
    first_token_time: float | None = None,
) -> dict[str, Any]:
    """Combine standard timing metrics with provider-specific usage parsing."""
    return {
        **_timing_metrics(start_time, time.time(), first_token_time),
        **usage_parser(usage),
    }


def _log_and_end_span(
    span: Span,
    *,
    output: Any = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log *output*, *metrics* and *metadata* (when present) then end the span."""
    event: dict[str, Any] = {}
    if output is not None:
        event["output"] = output
    if metrics:
        event["metrics"] = metrics
    if metadata:
        event["metadata"] = metadata
    if event:
        span.log(**event)
    span.end()


def _log_error_and_end_span(span: Span, error: BaseException) -> None:
    """Log an error to *span* and immediately end it."""
    span.log(error=error)
    span.end()
