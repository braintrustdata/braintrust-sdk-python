"""Tracing helpers for the Instructor integration.

The Instructor integration emits a single ``task``-typed parent span per
``Instructor.create*`` call.  It does **not** emit ``llm``-typed spans —
the existing provider integrations (OpenAI / Anthropic / Cohere / …) own
LLM telemetry on the underlying ``client.chat.completions.create`` /
``client.messages.create`` call that Instructor invokes internally.
Putting token / usage metrics on this parent span would double-count
against those provider children in dashboard aggregations.

Parent span shape:

- ``type``  : ``task``
- ``name``  : ``instructor.create`` (or ``.create_with_completion`` etc.)
- ``input`` : ``{"response_model": <name>, "mode": <str>, "messages": [...]}``
- ``output``: the extracted Pydantic model (or list of models for
  iterable/partial helpers).  This is Instructor's product and is not
  present on any provider child span.
- ``metadata``: ``response_model`` (class name), ``mode``, ``max_retries``,
  ``retry_count``, ``validation_errors``.
- ``metrics``: empty.  Token usage stays on the provider child span.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


log = logging.getLogger(__name__)


# Hooks moved to ``instructor.core.hooks`` in 1.10+; both paths still work in
# 1.x.  Prefer the canonical path and fall back to the deprecated alias.
try:
    from instructor.core.hooks import HookName  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - older instructor
    try:
        from instructor.hooks import HookName  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - instructor missing
        HookName = None  # type: ignore[assignment]


def _response_model_name(model: Any) -> str | None:
    """Return a short name for the response model, suitable as metadata."""
    if model is None:
        return None
    if isinstance(model, type):
        return model.__name__
    # Iterable[Person] / Partial[Person] expose __args__
    inner = getattr(model, "__args__", None)
    if inner:
        first = inner[0]
        if isinstance(first, type):
            return first.__name__
    return repr(model)


def _mode_name(instance: Any) -> str | None:
    """Read the configured Instructor mode from the client instance."""
    mode = getattr(instance, "mode", None)
    if mode is None:
        return None
    return getattr(mode, "name", None) or str(mode)


def _max_retries_value(max_retries: Any) -> Any:
    """Normalize Instructor's ``max_retries`` argument for metadata logging.

    ``max_retries`` may be an ``int`` or a ``tenacity.Retrying`` /
    ``AsyncRetrying`` instance.  Log the int directly; for Retrying objects
    extract a stop attempt number when one is configured, otherwise log the
    repr so users can see what was passed.
    """
    if isinstance(max_retries, int):
        return max_retries
    stop = getattr(max_retries, "stop", None)
    max_attempt = getattr(stop, "max_attempt_number", None)
    if isinstance(max_attempt, int):
        return max_attempt
    return repr(max_retries)


def _extract_output(result: Any) -> Any:
    """Return the Instructor-produced model while leaving serialization to Braintrust."""
    if isinstance(result, tuple) and len(result) == 2:
        # create_with_completion returns (model, raw_completion). The model is
        # Instructor's output; provider integrations own the raw completion span.
        return result[0]
    if isinstance(result, list):
        return [_extract_output(item) for item in result]
    return result


def _build_input(response_model: Any, instance: Any, args: Any, kwargs: Any) -> dict[str, Any]:
    messages = kwargs.get("messages")
    if messages is None and len(args) >= 2:
        messages = args[1]
    return {
        "response_model": _response_model_name(response_model),
        "mode": _mode_name(instance),
        "messages": messages,
    }


def _build_metadata(
    *,
    response_model: Any,
    instance: Any,
    max_retries: Any,
    retry_count: int,
    validation_errors: list[str],
) -> dict[str, Any]:
    return {
        "response_model": _response_model_name(response_model),
        "mode": _mode_name(instance),
        "max_retries": _max_retries_value(max_retries),
        "retry_count": retry_count,
        "validation_errors": list(validation_errors),
    }


class _CallTracker:
    """Per-call hook listeners that count retries and capture validation errors.

    Instructor emits ``PARSE_ERROR`` on every Pydantic validation failure that
    triggers a retry, and ``COMPLETION_RESPONSE`` after every successful
    underlying provider call.  Registering listeners on the client's existing
    ``hooks`` object lets us observe retry behavior without touching the
    retry loop itself; the listeners are removed in ``finally`` so per-call
    state never leaks across calls.
    """

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.validation_errors: list[str] = []
        self._hooks = getattr(instance, "hooks", None)
        self._registered: list[tuple[Any, Any]] = []

    def __enter__(self) -> "_CallTracker":
        if self._hooks is None or HookName is None:
            return self

        def _on_parse_error(error: Any) -> None:
            self.validation_errors.append(str(error))

        try:
            self._hooks.on(HookName.PARSE_ERROR, _on_parse_error)
            self._registered.append((HookName.PARSE_ERROR, _on_parse_error))
        except Exception:  # pragma: no cover - defensive
            log.debug("braintrust instructor: hooks.on failed", exc_info=True)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._hooks is None:
            return
        for event, handler in self._registered:
            try:
                self._hooks.off(event, handler)
            except Exception:  # pragma: no cover - defensive
                log.debug("braintrust instructor: hooks.off failed", exc_info=True)
        self._registered.clear()

    @property
    def retry_count(self) -> int:
        return len(self.validation_errors)


def _extract_response_model(args: Any, kwargs: Any) -> Any:
    if "response_model" in kwargs:
        return kwargs["response_model"]
    if args:
        return args[0]
    return None


def _make_sync_wrapper(span_name: str):
    def _wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        response_model = _extract_response_model(args, kwargs)
        max_retries = kwargs.get("max_retries", 3)
        with (
            _CallTracker(instance) as tracker,
            start_span(
                name=span_name,
                type=SpanTypeAttribute.TASK,
                input=_build_input(response_model, instance, args, kwargs),
            ) as span,
        ):
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.log(
                    error=exc,
                    metadata=_build_metadata(
                        response_model=response_model,
                        instance=instance,
                        max_retries=max_retries,
                        retry_count=tracker.retry_count,
                        validation_errors=tracker.validation_errors,
                    ),
                )
                raise
            span.log(
                output=_extract_output(result),
                metadata=_build_metadata(
                    response_model=response_model,
                    instance=instance,
                    max_retries=max_retries,
                    retry_count=tracker.retry_count,
                    validation_errors=tracker.validation_errors,
                ),
            )
            return result

    return _wrapper


def _make_async_wrapper(span_name: str):
    async def _wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        response_model = _extract_response_model(args, kwargs)
        max_retries = kwargs.get("max_retries", 3)
        with (
            _CallTracker(instance) as tracker,
            start_span(
                name=span_name,
                type=SpanTypeAttribute.TASK,
                input=_build_input(response_model, instance, args, kwargs),
            ) as span,
        ):
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.log(
                    error=exc,
                    metadata=_build_metadata(
                        response_model=response_model,
                        instance=instance,
                        max_retries=max_retries,
                        retry_count=tracker.retry_count,
                        validation_errors=tracker.validation_errors,
                    ),
                )
                raise
            span.log(
                output=_extract_output(result),
                metadata=_build_metadata(
                    response_model=response_model,
                    instance=instance,
                    max_retries=max_retries,
                    retry_count=tracker.retry_count,
                    validation_errors=tracker.validation_errors,
                ),
            )
            return result

    return _wrapper


def _make_sync_iterable_wrapper(span_name: str):
    def _wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        response_model = _extract_response_model(args, kwargs)
        max_retries = kwargs.get("max_retries", 3)

        def _iterate() -> Iterator[Any]:
            output = []
            with (
                _CallTracker(instance) as tracker,
                start_span(
                    name=span_name,
                    type=SpanTypeAttribute.TASK,
                    input=_build_input(response_model, instance, args, kwargs),
                ) as span,
            ):
                try:
                    result = wrapped(*args, **kwargs)
                    for item in result:
                        output.append(_extract_output(item))
                        yield item
                except Exception as exc:
                    span.log(
                        error=exc,
                        metadata=_build_metadata(
                            response_model=response_model,
                            instance=instance,
                            max_retries=max_retries,
                            retry_count=tracker.retry_count,
                            validation_errors=tracker.validation_errors,
                        ),
                    )
                    raise
                span.log(
                    output=output,
                    metadata=_build_metadata(
                        response_model=response_model,
                        instance=instance,
                        max_retries=max_retries,
                        retry_count=tracker.retry_count,
                        validation_errors=tracker.validation_errors,
                    ),
                )

        return _iterate()

    return _wrapper


def _make_async_iterable_wrapper(span_name: str):
    def _wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        response_model = _extract_response_model(args, kwargs)
        max_retries = kwargs.get("max_retries", 3)

        async def _iterate() -> AsyncIterator[Any]:
            output = []
            with (
                _CallTracker(instance) as tracker,
                start_span(
                    name=span_name,
                    type=SpanTypeAttribute.TASK,
                    input=_build_input(response_model, instance, args, kwargs),
                ) as span,
            ):
                try:
                    result = wrapped(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    async for item in result:
                        output.append(_extract_output(item))
                        yield item
                except Exception as exc:
                    span.log(
                        error=exc,
                        metadata=_build_metadata(
                            response_model=response_model,
                            instance=instance,
                            max_retries=max_retries,
                            retry_count=tracker.retry_count,
                            validation_errors=tracker.validation_errors,
                        ),
                    )
                    raise
                span.log(
                    output=output,
                    metadata=_build_metadata(
                        response_model=response_model,
                        instance=instance,
                        max_retries=max_retries,
                        retry_count=tracker.retry_count,
                        validation_errors=tracker.validation_errors,
                    ),
                )

        return _iterate()

    return _wrapper


# Public wrapper callables, one per Instructor entry point.  Names match the
# patcher targets in ``patchers.py``.
_create_wrapper = _make_sync_wrapper("instructor.create")
_create_with_completion_wrapper = _make_sync_wrapper("instructor.create_with_completion")
_create_partial_wrapper = _make_sync_iterable_wrapper("instructor.create_partial")
_create_iterable_wrapper = _make_sync_iterable_wrapper("instructor.create_iterable")

_async_create_wrapper = _make_async_wrapper("instructor.create")
_async_create_with_completion_wrapper = _make_async_wrapper("instructor.create_with_completion")
_async_create_partial_wrapper = _make_async_iterable_wrapper("instructor.create_partial")
_async_create_iterable_wrapper = _make_async_iterable_wrapper("instructor.create_iterable")
