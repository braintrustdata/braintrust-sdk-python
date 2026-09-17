"""TypeSafe-specific tracing helpers."""

from collections.abc import Mapping
from typing import Any

from braintrust.integrations.utils import _log_and_end_span, _log_error_and_end_span
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "typesafe"
_MISSING = object()
_QUESTION_FIELDS = ("type", "instructions", "criteria")
_ANSWER_FIELDS = ("type", "noul", "choice", "score", "confidence", "legend", "probabilities")


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


def _items_with_ids(values: Any, payload_fn) -> Any:
    if not isinstance(values, Mapping):
        return values
    return [{**payload_fn(value), "id": identifier} for identifier, value in values.items()]


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


def _span_input(state: Any, questions: Any) -> dict[str, Any]:
    return {
        "state": state,
        "questions": _items_with_ids(questions, _question_payload),
    }


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


def _log_response(span: Any, response: Any) -> None:
    answers = getattr(response, "answers", None)
    response_model = getattr(response, "model", None)
    _log_and_end_span(
        span,
        output={"answers": _items_with_ids(answers, _answer_payload)},
        metrics=_token_metrics(response),
        metadata={"model": response_model} if isinstance(response_model, str) else None,
    )


def _start_system_one_span(instance: Any, args: Any, kwargs: dict[str, Any]):
    state, questions, model = _request_parts(instance, args, kwargs)
    metadata = {"provider": "typesafe"}
    if isinstance(model, str):
        metadata["model"] = model
    return start_span(
        name="typesafe.systemOne",
        type=SpanTypeAttribute.QUESTION,
        input=_span_input(state, questions),
        metadata=metadata,
    )


def _system_one_wrapper(wrapped, instance, args, kwargs):
    span = _start_system_one_span(instance, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_response(span, response)
    return response


async def _async_system_one_wrapper(wrapped, instance, args, kwargs):
    span = _start_system_one_span(instance, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_response(span, response)
    return response
