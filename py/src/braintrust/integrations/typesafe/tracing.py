"""TypeSafe-specific tracing helpers."""

import os
import re
import sys
from collections.abc import Mapping
from typing import Any

from braintrust.integrations.question import (
    _ANSWER_FIELDS,
    _QUESTION_FIELDS,
    _question_span_input,
    _question_span_output,
)
from braintrust.integrations.utils import (
    _log_and_end_span,
    _log_error_and_end_span,
    _materialize_attachment,
    _try_to_dict,
)
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "typesafe"
_MISSING = object()


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


def _tagged_payload(value: Any, fields: tuple[str, ...], *, type_suffix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {field: value[field] for field in fields if field in value}

    kind = value.__class__.__name__.removesuffix(type_suffix).lower()
    payload = {"type": kind}
    for field in fields:
        if field == "type":
            continue
        field_value = getattr(value, field, _MISSING)
        if field_value is not _MISSING:
            payload[field] = field_value
    return payload


def _question_payload(value: Any) -> dict[str, Any]:
    return _tagged_payload(value, _QUESTION_FIELDS)


def _answer_payload(value: Any) -> dict[str, Any]:
    return _tagged_payload(value, _ANSWER_FIELDS, type_suffix="Answer")


def _call_value(args: Any, kwargs: dict[str, Any], name: str, position: int) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


def _request_parts(instance: Any, args: Any, kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
    state = _call_value(args, kwargs, "state", 0)
    questions = _call_value(args, kwargs, "questions", 1)
    model = kwargs.get("model")

    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, Mapping):
        state = extra_body.get("state", state)
        questions = extra_body.get("questions", questions)
        model = extra_body.get("model", model)

    if not isinstance(model, str):
        config = getattr(instance, "_config", None)
        model = getattr(config, "default_model", None)
    return state, questions, model


def _image_payload(value: Any) -> Any:
    # Only interpret TypeSafe's Image objects as files/buffers. Arbitrary JSON
    # strings in the request must never be treated as local paths.
    image_type = getattr(sys.modules.get("typesafe_sdk"), "Image", ())
    if isinstance(value, image_type):
        data = value.data
        if isinstance(data, os.PathLike):
            with open(data, "rb") as image_file:
                data = image_file.read()
        else:
            data = bytes(data)
        mime_type = (
            "image/png"
            if data.startswith(b"\x89PNG\r\n\x1a\n")
            else "image/jpeg"
            if data.startswith(b"\xff\xd8\xff")
            else None
        )
        if mime_type is not None:
            resolved = _materialize_attachment(data, mime_type=mime_type)
            if resolved is not None:
                return resolved.attachment
    if isinstance(value, Mapping) and "data_base64" in value:
        image_format = value.get("format")
        if image_format in ("png", "jpg", "jpeg") and isinstance(value["data_base64"], str):
            resolved = _materialize_attachment(
                value["data_base64"], mime_type="image/png" if image_format == "png" else "image/jpeg"
            )
            if resolved is not None:
                return resolved.attachment
    return value


def _span_input(state: Any, questions: Any, assets: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(questions, Mapping):
        questions = {identifier: _question_payload(value) for identifier, value in questions.items()}
    references = [re.escape(key) for key in assets if isinstance(key, str) and key]
    references.sort(key=len, reverse=True)
    pattern = re.compile("(" + "|".join(references) + ")") if references else None
    captured_images: dict[int, Any] = {}

    def capture(value):
        key = id(value)
        if key in captured_images:
            return captured_images[key]
        captured = _image_payload(value)
        if captured is not value:
            captured_images[key] = captured
        return captured

    def resolve(reference):
        asset = assets[reference]
        captured = capture(asset)
        return reference if captured is asset else captured

    def media_transform(value):
        if isinstance(value, str) and assets:
            if value in assets:
                return resolve(value)
            # TypeSafe expands references inside text as well as whole values.
            if pattern is not None:
                parts = pattern.split(value)
                if len(parts) > 1:
                    return [resolve(part) if part in assets else part for part in parts if part]
        return capture(value)

    return _question_span_input(state, questions, media_transform=media_transform)


def _token_metrics(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    metrics: dict[str, int] = {}
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0:
        metrics["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens >= 0:
        metrics["completion_tokens"] = completion_tokens
    if "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


def _log_response(span: Any, response: Any, span_input: Any) -> None:
    answers = _try_to_dict(getattr(response, "answers", None))
    if isinstance(answers, Mapping):
        answers = {identifier: _answer_payload(value) for identifier, value in answers.items()}
    response_model = getattr(response, "model", None)
    _log_and_end_span(
        span,
        output=_question_span_output(answers, span_input),
        metrics=_token_metrics(response),
        metadata={"model": response_model} if isinstance(response_model, str) else None,
    )


def _start_system_one_span(instance: Any, args: Any, kwargs: dict[str, Any]):
    state, questions, model = _request_parts(instance, args, kwargs)
    metadata = {"provider": "typesafe"}
    if isinstance(model, str):
        metadata["model"] = model
    assets = dict(kwargs.get("assets") or {})
    attached = getattr(instance, "_attached", None)
    if attached is not None:
        assets.update(attached.snapshot())
    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, Mapping) and isinstance(extra_body.get("assets"), Mapping):
        assets = dict(extra_body["assets"])
    # Images may also be referenced in text by their SDK-generated reference.
    assets.update({str(image): image for image in list(assets.values()) if not isinstance(image, Mapping)})
    span_input = _span_input(state, questions, assets)
    return start_span(
        name="typesafe.systemOne",
        type=SpanTypeAttribute.DECISION,
        input=span_input,
        metadata=metadata,
    ), span_input


def _system_one_wrapper(wrapped, instance, args, kwargs):
    span, span_input = _start_system_one_span(instance, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_response(span, response, span_input)
    return response


async def _async_system_one_wrapper(wrapped, instance, args, kwargs):
    span, span_input = _start_system_one_span(instance, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_response(span, response, span_input)
    return response
