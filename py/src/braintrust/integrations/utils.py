"""Shared tracing utilities for Braintrust SDK integrations.

These helpers are common building blocks used across multiple provider
integrations. Keeping them here avoids duplication and makes behavioral fixes
propagate to all providers at once.

Names are prefixed with ``_`` so that consumer modules can import them
directly without aliasing (e.g. ``from braintrust.integrations.utils import
_try_to_dict``).
"""

import base64
import binascii
import mimetypes
import os
import re
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from braintrust.logger import Attachment, Span
from braintrust.util import is_numeric


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$")

# Keep these overrides narrow and deterministic across platforms. Python's
# mimetypes registry varies by OS (notably on Windows), which can otherwise
# produce verbose vendor-subtype suffixes instead of common file extensions.
_KNOWN_ATTACHMENT_EXTENSIONS = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}


def _try_to_dict(obj: Any) -> dict[str, Any] | Any:
    """Best-effort conversion of an SDK response object to a plain dict.

    Tries, in order:
      1. ``model_dump(mode="python")`` (preferred for Pydantic v2 objects)
      2. ``model_dump()``               (fallback for SDKs with custom signatures)
      3. ``to_dict()``                  (used by some provider SDK response objects)
      4. ``dict()``                     (Pydantic v1 / legacy)
      5. ``vars(obj)``                  (plain Python attribute bags)
      6. returns *obj* unchanged

    Only dict-like conversion results are accepted; non-dict results are
    ignored so later fallbacks still run.

    Pydantic serializer warnings (common with generic/discriminated-union
    models such as OpenAI's ``ParsedResponse[T]``) are suppressed.
    """
    if isinstance(obj, dict):
        return obj

    model_dump = getattr(obj, "model_dump", None)

    def _call_model_dump_python() -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
            return model_dump(mode="python")

    def _call_model_dump() -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
            return model_dump()

    to_dict = getattr(obj, "to_dict", None)
    dict_method = getattr(obj, "dict", None)

    converters: list[Callable[[], Any]] = []
    if callable(model_dump):
        converters.extend((_call_model_dump_python, _call_model_dump))
    if callable(to_dict):
        converters.append(to_dict)
    if callable(dict_method):
        converters.append(dict_method)
    converters.append(lambda: vars(obj))

    for converter in converters:
        try:
            result = converter()
        except Exception:
            continue
        if isinstance(result, dict):
            return result

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


def _attachment_filename_for_mime_type(mime_type: str, *, prefix: str = "file") -> str:
    """Return a stable filename for *mime_type* using *prefix*.

    Examples:
    - ``image/png`` with prefix ``image`` -> ``image.png``
    - ``application/pdf`` with prefix ``document`` -> ``document.pdf``
    - ``image/svg+xml`` with prefix ``file`` -> ``file.svg``
    - ``application/vnd.openxmlformats-officedocument.spreadsheetml.sheet``
      with prefix ``file`` -> ``file.xlsx``
    """
    extension = _KNOWN_ATTACHMENT_EXTENSIONS.get(mime_type)
    if extension is None:
        guessed_extension = mimetypes.guess_extension(mime_type)
        if guessed_extension:
            extension = guessed_extension.lstrip(".")
        else:
            extension = mime_type.split("/", 1)[1] if "/" in mime_type else "bin"
            extension = extension.split("+", 1)[0]
    return f"{prefix}.{extension}"


@dataclass(frozen=True)
class _ResolvedAttachment:
    attachment: Attachment

    @property
    def mime_type(self) -> str:
        return self.attachment.reference.get("content_type") or "application/octet-stream"

    @property
    def filename(self) -> str:
        return self.attachment.reference.get("filename") or "file"

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def multimodal_part_payload(self) -> dict[str, Any]:
        if self.is_image:
            return {"image_url": {"url": self.attachment}}
        return {"file": {"file_data": self.attachment, "filename": self.filename}}


def _label_for_mime_type(mime_type: str, label: str | None) -> str:
    if label is not None:
        return label
    return "image" if mime_type.startswith("image/") else "file"


def _default_attachment_filename(
    mime_type: str,
    *,
    filename: str | None = None,
    label: str | None = None,
    prefix: str | None = None,
) -> str:
    return filename or _attachment_filename_for_mime_type(
        mime_type,
        prefix=prefix or _label_for_mime_type(mime_type, label),
    )


def _resolved_attachment_from_bytes(
    data: bytes | bytearray,
    mime_type: str,
    *,
    filename: str | None = None,
    label: str | None = None,
    prefix: str | None = None,
) -> _ResolvedAttachment:
    resolved_filename = _default_attachment_filename(mime_type, filename=filename, label=label, prefix=prefix)
    attachment = Attachment(
        data=data if isinstance(data, bytes) else bytes(data),
        filename=resolved_filename,
        content_type=mime_type,
    )
    return _ResolvedAttachment(attachment=attachment)


def _resolved_attachment_from_base64(
    data: str,
    mime_type: str,
    *,
    filename: str | None = None,
    label: str | None = None,
    prefix: str | None = None,
) -> _ResolvedAttachment | None:
    raw_data = data.partition(",")[2] if data.startswith("data:") else data

    try:
        decoded = base64.b64decode(raw_data, validate=True)
    except (binascii.Error, ValueError):
        return None

    return _resolved_attachment_from_bytes(decoded, mime_type, filename=filename, label=label, prefix=prefix)


def _materialize_attachment(
    value: Any,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
    label: str | None = None,
    prefix: str | None = None,
) -> _ResolvedAttachment | None:
    """Resolve common attachment inputs into a concrete attachment object.

    Supports existing :class:`Attachment` objects, bytes-like data, raw base64
    strings, data URLs, filesystem paths, file-like objects, and common
    ``(filename, value, content_type)`` tuple inputs.
    """
    if value is None:
        return None

    if isinstance(value, Attachment):
        ref_ct = value.reference.get("content_type")
        ref_fn = value.reference.get("filename")
        resolved_mime_type = mime_type or ref_ct or "application/octet-stream"
        resolved_filename = (
            filename
            or ref_fn
            or _default_attachment_filename(
                resolved_mime_type,
                label=label,
                prefix=prefix,
            )
        )
        if ref_ct != resolved_mime_type or ref_fn != resolved_filename:
            attachment = Attachment(
                data=value.data,
                filename=resolved_filename,
                content_type=resolved_mime_type,
            )
            return _ResolvedAttachment(attachment=attachment)
        return _ResolvedAttachment(attachment=value)

    if isinstance(value, tuple):
        tuple_filename = value[0] if value and isinstance(value[0], (str, os.PathLike)) else None
        tuple_value = value[1] if len(value) > 1 else None
        tuple_content_type = value[2] if len(value) > 2 and isinstance(value[2], str) else None
        return _materialize_attachment(
            tuple_value,
            mime_type=mime_type or tuple_content_type,
            filename=filename or (os.path.basename(os.fspath(tuple_filename)) if tuple_filename is not None else None),
            label=label,
            prefix=prefix,
        )

    if isinstance(value, (bytes, bytearray)):
        resolved_mime_type = (
            mime_type
            or (mimetypes.guess_type(filename)[0] if filename is not None else None)
            or "application/octet-stream"
        )
        return _resolved_attachment_from_bytes(
            value, resolved_mime_type, filename=filename, label=label, prefix=prefix
        )

    if isinstance(value, (str, os.PathLike)):
        path_or_data = os.fspath(value)
        data_url_match = _DATA_URL_RE.match(path_or_data) if isinstance(value, str) else None
        if data_url_match:
            data_url_mime_type, _ = data_url_match.groups()
            return _resolved_attachment_from_base64(
                path_or_data,
                mime_type or data_url_mime_type,
                filename=filename,
                label=label,
                prefix=prefix,
            )

        try:
            with open(path_or_data, "rb") as file_obj:
                data = file_obj.read()
        except OSError:
            if isinstance(value, str) and mime_type is not None:
                return _resolved_attachment_from_base64(
                    value,
                    mime_type,
                    filename=filename,
                    label=label,
                    prefix=prefix,
                )
            return None

        resolved_filename = filename or os.path.basename(path_or_data)
        resolved_mime_type = mime_type or mimetypes.guess_type(resolved_filename)[0] or "application/octet-stream"
        return _resolved_attachment_from_bytes(
            data,
            resolved_mime_type,
            filename=resolved_filename,
            label=label,
            prefix=prefix,
        )

    read = getattr(value, "read", None)
    if callable(read):
        file_name_attr = getattr(value, "name", None)
        resolved_filename = filename or (os.path.basename(file_name_attr) if isinstance(file_name_attr, str) else None)
        resolved_mime_type = (
            mime_type
            or (mimetypes.guess_type(resolved_filename)[0] if resolved_filename is not None else None)
            or "application/octet-stream"
        )

        position = None
        try:
            position = value.tell()
        except Exception:
            pass

        try:
            data = value.read()
        finally:
            if position is not None:
                try:
                    value.seek(position)
                except Exception:
                    pass

        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, (bytes, bytearray)):
            return _resolved_attachment_from_bytes(
                data,
                resolved_mime_type,
                filename=resolved_filename,
                label=label,
                prefix=prefix,
            )
        return None

    return None


_AUDIO_FORMAT_TO_MIME_TYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


def _infer_audio_mime_type(response: Any, response_format: Any = None) -> str:
    raw_response = getattr(response, "response", None)
    if raw_response is None and isinstance(response, Mapping):
        raw_response = response.get("response")

    headers = getattr(raw_response, "headers", None)
    if headers is not None:
        content_type = headers.get("content-type")
        if isinstance(content_type, str) and content_type:
            return content_type.split(";", 1)[0].strip()

    if isinstance(response_format, str) and response_format:
        normalized = response_format.lower()
        return _AUDIO_FORMAT_TO_MIME_TYPE.get(
            normalized,
            normalized if "/" in normalized else f"audio/{normalized}",
        )

    return "application/octet-stream"


def _extract_audio_output(
    response: Any,
    *,
    response_format: Any = None,
    prefix: str = "generated_audio",
) -> dict[str, Any]:
    audio_bytes = getattr(response, "content", None)
    if not isinstance(audio_bytes, (bytes, bytearray)) and isinstance(response, Mapping):
        raw_response = response.get("response")
        audio_bytes = getattr(raw_response, "content", None)

    if not isinstance(audio_bytes, (bytes, bytearray)):
        return {"type": "audio"}

    mime_type = _infer_audio_mime_type(response, response_format)
    resolved_attachment = _materialize_attachment(
        audio_bytes,
        mime_type=mime_type,
        prefix=prefix,
    )
    if resolved_attachment is None:
        return {
            "type": "audio",
            "mime_type": mime_type,
            "audio_size_bytes": len(audio_bytes),
        }

    return {
        "type": "audio",
        "mime_type": resolved_attachment.mime_type,
        "audio_size_bytes": len(audio_bytes),
        **resolved_attachment.multimodal_part_payload,
    }


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
