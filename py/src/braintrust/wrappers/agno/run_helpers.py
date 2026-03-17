import time
from inspect import isawaitable
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute

from .utils import (
    extract_metadata,
    extract_metrics,
    is_async_iterator,
    is_sync_iterator,
    omit,
    trace_async_stream_result,
    trace_sync_stream_result,
)


def run_public_dispatch_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    *,
    default_name: str,
    metadata_component: str,
) -> Any:
    """Trace a public synchronous `run(...)` dispatch method.

    Handles both non-streaming return values and synchronous streaming iterators.
    For iterator results, span lifecycle is delegated to `trace_sync_stream_result`.
    """
    component_name = getattr(instance, "name", None) or default_name
    input_arg = args[0] if len(args) > 0 else kwargs.get("input")
    input_data = {"input": input_arg}
    metadata = {**omit(kwargs, ["input"]), **extract_metadata(instance, metadata_component)}

    span = start_span(
        name=f"{component_name}.run",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=metadata,
    )
    span.set_current()
    start = time.time()
    try:
        result = wrapped(*args, **kwargs)
        if is_sync_iterator(result):
            return trace_sync_stream_result(result, span, start)
        span.log(
            output=result,
            metrics=extract_metrics(result),
        )
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise


def arun_public_dispatch_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    *,
    default_name: str,
    metadata_component: str,
) -> Any:
    """Trace a public `arun(...)` dispatch method across async return contracts.

    Supports all observed `arun` dispatcher behaviors:
    - immediate return value
    - awaitable returning a value
    - direct async iterator
    - awaitable returning an async iterator

    If an async iterator is returned (directly or after await), span lifecycle is
    delegated to `trace_async_stream_result` so the span remains open until stream
    consumption completes.
    """
    component_name = getattr(instance, "name", None) or default_name
    input_arg = args[0] if len(args) > 0 else kwargs.get("input")
    input_data = {"input": input_arg}
    metadata = {**omit(kwargs, ["input"]), **extract_metadata(instance, metadata_component)}

    span = start_span(
        name=f"{component_name}.arun",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=metadata,
    )
    span.set_current()
    start = time.time()
    try:
        result = wrapped(*args, **kwargs)

        if isawaitable(result):

            async def _trace_awaitable():
                should_end_span = True
                try:
                    awaited = await result
                    if is_async_iterator(awaited):
                        should_end_span = False
                        return trace_async_stream_result(awaited, span, start)
                    span.log(
                        output=awaited,
                        metrics=extract_metrics(awaited),
                    )
                    return awaited
                except Exception as e:
                    span.log(error=str(e))
                    raise
                finally:
                    if should_end_span:
                        span.unset_current()
                        span.end()

            return _trace_awaitable()

        if is_async_iterator(result):
            return trace_async_stream_result(result, span, start)

        span.log(
            output=result,
            metrics=extract_metrics(result),
        )
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise
