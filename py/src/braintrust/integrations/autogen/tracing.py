"""AutoGen-specific tracing helpers."""

from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones


_agent_public_run_depth: ContextVar[int] = ContextVar("braintrust_autogen_agent_public_run_depth", default=0)


def _agent_metadata(instance: Any) -> dict[str, Any]:
    return clean_nones(
        {
            "component": "agent",
            "agent_name": getattr(instance, "name", None),
            "agent_description": getattr(instance, "description", None),
            "agent_class": instance.__class__.__name__,
        }
    )


def _team_metadata(instance: Any) -> dict[str, Any]:
    participants = getattr(instance, "_participants", None) or getattr(instance, "participants", None)
    return clean_nones(
        {
            "component": "team",
            "team_name": getattr(instance, "name", None),
            "team_description": getattr(instance, "description", None),
            "team_class": instance.__class__.__name__,
            "participant_names": [
                getattr(participant, "name", participant.__class__.__name__) for participant in participants
            ]
            if participants
            else None,
        }
    )


def _task_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    return clean_nones(
        {
            "task": kwargs.get("task"),
            "output_task_messages": kwargs.get("output_task_messages"),
        }
    )


def _messages_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = args[0] if args else kwargs.get("messages")
    return clean_nones({"messages": messages})


async def _agent_run_wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    name = getattr(instance, "name", None) or instance.__class__.__name__
    token = _agent_public_run_depth.set(_agent_public_run_depth.get() + 1)
    try:
        with start_span(
            name=f"{name}.run",
            type=SpanTypeAttribute.TASK,
            input=_task_input(kwargs),
            metadata=_agent_metadata(instance),
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.log(error=exc)
                raise
            span.log(output=result)
            return result
    finally:
        _agent_public_run_depth.reset(token)


async def _agent_run_stream_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    name = getattr(instance, "name", None) or instance.__class__.__name__
    token = _agent_public_run_depth.set(_agent_public_run_depth.get() + 1)
    try:
        with start_span(
            name=f"{name}.run_stream",
            type=SpanTypeAttribute.TASK,
            input=_task_input(kwargs),
            metadata=_agent_metadata(instance),
        ) as span:
            events = []
            try:
                async for event in wrapped(*args, **kwargs):
                    events.append(event)
                    yield event
            except Exception as exc:
                span.log(error=exc, output={"events": events})
                raise
            span.log(output={"events": events})
    finally:
        _agent_public_run_depth.reset(token)


async def _agent_on_messages_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if _agent_public_run_depth.get() > 0:
        return await wrapped(*args, **kwargs)
    name = getattr(instance, "name", None) or instance.__class__.__name__
    with start_span(
        name=f"{name}.run",
        type=SpanTypeAttribute.TASK,
        input=_messages_input(args, kwargs),
        metadata=_agent_metadata(instance),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.log(error=exc)
            raise
        span.log(output=result)
        return result


async def _agent_on_messages_stream_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    if _agent_public_run_depth.get() > 0:
        async for event in wrapped(*args, **kwargs):
            yield event
        return
    name = getattr(instance, "name", None) or instance.__class__.__name__
    with start_span(
        name=f"{name}.run_stream",
        type=SpanTypeAttribute.TASK,
        input=_messages_input(args, kwargs),
        metadata=_agent_metadata(instance),
    ) as span:
        events = []
        try:
            async for event in wrapped(*args, **kwargs):
                events.append(event)
                yield event
        except Exception as exc:
            span.log(error=exc, output={"events": events})
            raise
        span.log(output={"events": events})


async def _team_run_wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    name = getattr(instance, "name", None) or instance.__class__.__name__
    with start_span(
        name=f"{name}.run",
        type=SpanTypeAttribute.TASK,
        input=_task_input(kwargs),
        metadata=_team_metadata(instance),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.log(error=exc)
            raise
        span.log(output=result)
        return result


async def _team_run_stream_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    name = getattr(instance, "name", None) or instance.__class__.__name__
    with start_span(
        name=f"{name}.run_stream",
        type=SpanTypeAttribute.TASK,
        input=_task_input(kwargs),
        metadata=_team_metadata(instance),
    ) as span:
        events = []
        try:
            async for event in wrapped(*args, **kwargs):
                events.append(event)
                yield event
        except Exception as exc:
            span.log(error=exc, output={"events": events})
            raise
        span.log(output={"events": events})


def _tool_metadata(instance: Any) -> dict[str, Any]:
    return clean_nones(
        {
            "component": "tool",
            "tool_name": getattr(instance, "name", None),
            "tool_description": getattr(instance, "description", None),
            "tool_class": instance.__class__.__name__,
        }
    )


async def _tool_run_wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    name = getattr(instance, "name", None) or instance.__class__.__name__
    tool_args = args[0] if args else kwargs.get("args")
    with start_span(
        name=f"{name}.run",
        type=SpanTypeAttribute.TOOL,
        input=tool_args,
        metadata=_tool_metadata(instance),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.log(error=exc)
            raise
        span.log(output=result)
        return result
