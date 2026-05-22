"""ADK-specific span creation, metadata extraction, stream handling, and output normalization."""

import contextvars
import inspect
import logging
import time
from collections.abc import Iterable, Mapping
from contextlib import aclosing
from functools import lru_cache
from itertools import chain
from typing import Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.utils import _materialize_attachment
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_content(content: Any) -> Any:
    """Serialize Google ADK Content/Part objects, converting binary data to Attachments."""
    if content is None:
        return None

    # Handle Content objects with parts
    if hasattr(content, "parts") and content.parts:
        serialized_parts = []
        for part in content.parts:
            serialized_parts.append(_serialize_part(part))

        result = {"parts": serialized_parts}
        if hasattr(content, "role"):
            result["role"] = content.role
        return result

    # Handle single Part
    return _serialize_part(content)


def _serialize_part(part: Any) -> Any:
    """Serialize a single Part object, handling binary data."""
    if part is None:
        return None

    # If it's already a dict, return as-is
    if isinstance(part, dict):
        return part

    # Handle Part objects with inline_data (binary data like images)
    if hasattr(part, "inline_data") and part.inline_data:
        inline_data = part.inline_data
        if hasattr(inline_data, "data") and hasattr(inline_data, "mime_type"):
            data = inline_data.data
            mime_type = inline_data.mime_type

            if isinstance(data, bytes):
                resolved_attachment = _materialize_attachment(data, mime_type=mime_type)
                if resolved_attachment is not None:
                    return resolved_attachment.multimodal_part_payload

    # Handle Part objects with file_data (file references)
    if hasattr(part, "file_data") and part.file_data:
        file_data = part.file_data
        result = {"file_data": {}}
        if hasattr(file_data, "file_uri"):
            result["file_data"]["file_uri"] = file_data.file_uri
        if hasattr(file_data, "mime_type"):
            result["file_data"]["mime_type"] = file_data.mime_type
        return result

    # Handle text parts
    if hasattr(part, "text") and part.text is not None:
        result = {"text": part.text}
        if hasattr(part, "thought") and part.thought:
            result["thought"] = part.thought
        return result

    # Try standard serialization methods
    return bt_safe_deep_copy(part)


@lru_cache(maxsize=128)
def _serialize_pydantic_schema(schema_class: Any) -> dict[str, Any]:
    """
    Serialize a Pydantic model class to its full JSON schema.

    Returns the complete schema including descriptions, constraints, and nested definitions
    so engineers can see exactly what structured output schema was used.
    """
    try:
        from pydantic import BaseModel

        if inspect.isclass(schema_class) and issubclass(schema_class, BaseModel):
            # Return the full JSON schema - includes all field info, descriptions, constraints, etc.
            return schema_class.model_json_schema()
    except (ImportError, AttributeError, TypeError):
        pass
    # If not a Pydantic model, return class name
    return {"__class__": schema_class.__name__ if inspect.isclass(schema_class) else str(type(schema_class).__name__)}


def _capture_config(config: Any) -> dict[str, Any] | Any:
    """
    Capture the ADK config fields that make LLM spans readable.

    Google ADK uses these fields for schemas:
    - response_schema, response_json_schema (in GenerateContentConfig for LLM requests)
    - input_schema, output_schema (in agent config)
    """
    if config is None or not config:
        return config

    config_fields = [
        "system_instruction",
        "response_mime_type",
        "response_schema",
        "response_json_schema",
        "input_schema",
        "output_schema",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "candidate_count",
    ]
    captured: dict[str, Any] = dict(config) if isinstance(config, dict) else {}

    for field in config_fields:
        value = _get_field(config, field)
        if value is None:
            continue
        if inspect.isclass(value):
            try:
                from pydantic import BaseModel

                if issubclass(value, BaseModel):
                    captured[field] = _serialize_pydantic_schema(value)
                    continue
            except (TypeError, ImportError):
                pass
        captured[field] = value

    return captured or config


def _omit(obj: Any, keys: Iterable[str]):
    return {k: v for k, v in obj.items() if k not in keys}


def _extract_metrics(response: Any) -> dict[str, float] | None:
    """Extract token usage metrics from Google GenAI response."""
    if not response:
        return None

    usage_metadata = getattr(response, "usage_metadata", None)
    if not usage_metadata:
        return None

    metrics: dict[str, float] = {}

    # Core token counts
    if hasattr(usage_metadata, "prompt_token_count") and usage_metadata.prompt_token_count is not None:
        metrics["prompt_tokens"] = float(usage_metadata.prompt_token_count)

    if hasattr(usage_metadata, "candidates_token_count") and usage_metadata.candidates_token_count is not None:
        metrics["completion_tokens"] = float(usage_metadata.candidates_token_count)

    if hasattr(usage_metadata, "total_token_count") and usage_metadata.total_token_count is not None:
        metrics["tokens"] = float(usage_metadata.total_token_count)

    # Cached token metrics
    if hasattr(usage_metadata, "cached_content_token_count") and usage_metadata.cached_content_token_count is not None:
        metrics["prompt_cached_tokens"] = float(usage_metadata.cached_content_token_count)

    # Reasoning token metrics (thoughts_token_count)
    if hasattr(usage_metadata, "thoughts_token_count") and usage_metadata.thoughts_token_count is not None:
        metrics["completion_reasoning_tokens"] = float(usage_metadata.thoughts_token_count)

    return metrics if metrics else None


def _extract_model_name(response: Any, llm_request: Any, instance: Any) -> str | None:
    """Extract model name from Google GenAI response, request, or flow instance."""
    # Try to get from response first
    if response:
        model_version = getattr(response, "model_version", None)
        if model_version:
            return model_version

    # Try to get from llm_request
    if llm_request:
        if hasattr(llm_request, "model") and llm_request.model:
            return str(llm_request.model)

    # Try to get from instance (flow's llm)
    if instance:
        if hasattr(instance, "llm"):
            llm = instance.llm
            if hasattr(llm, "model") and llm.model:
                return str(llm.model)

        # Try to get model from instance directly
        if hasattr(instance, "model") and instance.model:
            return str(instance.model)

    return None


def _get_field(value: Any, field: str, default: Any = None) -> Any:
    return value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)


def _part_has_field(part: Any, *field_names: str) -> bool:
    return any(_get_field(part, field_name) is not None for field_name in field_names)


def _capture_llm_request_input(llm_request: Any) -> Any:
    """Capture the ADK request fields that make LLM spans readable."""
    if llm_request is None:
        return None

    contents = _get_field(llm_request, "contents")
    config = _get_field(llm_request, "config")
    model = _get_field(llm_request, "model")
    live_connect_config = _get_field(llm_request, "live_connect_config")

    captured: dict[str, Any] = {}
    if model:
        captured["model"] = model
    if contents:
        captured["contents"] = (
            [_serialize_content(c) for c in contents] if isinstance(contents, list) else _serialize_content(contents)
        )
    if config:
        captured["config"] = _capture_config(config)
    if live_connect_config is not None or hasattr(llm_request, "live_connect_config") or isinstance(llm_request, dict):
        captured["live_connect_config"] = live_connect_config

    return captured or llm_request


def _event_output_with_content(last_event: Any, event_with_content: Any | None) -> Any:
    if event_with_content is None or _get_field(last_event, "content") is not None:
        return last_event

    content = _get_field(event_with_content, "content")
    if content is None:
        return last_event

    if isinstance(last_event, dict):
        return {**last_event, "content": content}

    # Keep the original event instead of recursively serializing it; add the
    # captured content alongside it so Braintrust can serialize both values.
    return {"event": last_event, "content": content}


def _determine_llm_call_type(llm_request: Any, model_response: Any = None) -> str:
    """
    Determine the type of LLM call based on the request and response content.

    Returns:
        - "tool_selection" if the LLM selected a tool to call in its response
        - "response_generation" if the LLM is generating a response after tool execution
        - "direct_response" if there are no tools involved or tools available but not used
    """
    try:
        has_function_response = any(
            _part_has_field(part, "function_response", "functionResponse")
            for content in (_get_field(llm_request, "contents", []) or [])
            for part in (_get_field(content, "parts", []) or [])
        )

        response_has_function_call = False
        if model_response:
            if hasattr(model_response, "get_function_calls"):
                try:
                    function_calls = model_response.get_function_calls()
                    response_has_function_call = bool(function_calls)
                except Exception:
                    pass

            if not response_has_function_call:
                content = _get_field(model_response, "content")
                response_has_function_call = any(
                    _part_has_field(part, "function_call", "functionCall")
                    for part in chain(
                        _get_field(content, "parts", []) or [], _get_field(model_response, "parts", []) or []
                    )
                )

        if has_function_response:
            return "response_generation"
        elif response_has_function_call:
            return "tool_selection"
        else:
            return "direct_response"

    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Thread-bridge helper (wrapt-style wrapper)
# ---------------------------------------------------------------------------


def _create_thread_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """wrapt wrapper for ``create_thread`` that copies context into new threads."""
    ctx = contextvars.copy_context()

    # ``create_thread(target, ...)`` — target may be positional or keyword.
    if args:
        target = args[0]
        rest_args = args[1:]
    else:
        target = kwargs.pop("target")
        rest_args = args

    def _run_in_context(*target_args: Any, **target_kwargs: Any) -> Any:
        return ctx.run(target, *target_args, **target_kwargs)

    return wrapped(_run_in_context, *rest_args, **kwargs)


# ---------------------------------------------------------------------------
# wrapt wrapper functions (used by patchers)
# ---------------------------------------------------------------------------


async def _agent_run_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    parent_context = args[0] if len(args) > 0 else kwargs.get("parent_context")

    async def _trace():
        with start_span(
            name=f"agent_run [{instance.name}]",
            type=SpanTypeAttribute.TASK,
            metadata={"parent_context": parent_context, **_omit(kwargs, ["parent_context"])},
        ) as agent_span:
            last_event = None
            async with aclosing(wrapped(*args, **kwargs)) as agen:
                async for event in agen:
                    if event.is_final_response():
                        last_event = event
                    yield event
            if last_event:
                agent_span.log(output=last_event)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _flow_run_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")

    async def _trace():
        with start_span(
            name="call_llm",
            type=SpanTypeAttribute.TASK,
            metadata={
                "invocation_context": invocation_context,
                **_omit(kwargs, ["invocation_context"]),
            },
        ) as llm_span:
            last_event = None
            async with aclosing(wrapped(*args, **kwargs)) as agen:
                async for event in agen:
                    last_event = event
                    yield event
            if last_event:
                llm_span.log(output=last_event)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _flow_call_llm_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
    llm_request = args[1] if len(args) > 1 else kwargs.get("llm_request")
    model_response_event = args[2] if len(args) > 2 else kwargs.get("model_response_event")

    async def _trace():
        # Capture only the fields we need to alter: contents may contain binary
        # data that should become Attachments, and config may contain Pydantic
        # schema classes that are clearer as JSON schema.
        captured_request = _capture_llm_request_input(llm_request)

        # Extract model name from request or instance
        model_name = _extract_model_name(None, llm_request, instance)

        # Create span BEFORE execution so child spans (like mcp_tool) have proper parent
        # Start with generic name - we'll update it after we see the response
        with start_span(
            name="llm_call",
            type=SpanTypeAttribute.LLM,
            input=captured_request,
            metadata={
                "invocation_context": invocation_context,
                "model_response_event": model_response_event,
                "flow_class": instance.__class__.__name__,
                "model": model_name,
                **_omit(kwargs, ["invocation_context", "model_response_event", "flow_class", "llm_call_type"]),
            },
        ) as llm_span:
            # Execute the LLM call and yield events while span is active
            last_event = None
            event_with_content = None
            start_time = time.time()
            first_token_time = None

            async with aclosing(wrapped(*args, **kwargs)) as agen:
                async for event in agen:
                    # Record time to first token
                    if first_token_time is None:
                        first_token_time = time.time()

                    last_event = event
                    if hasattr(event, "content") and event.content is not None:
                        event_with_content = event
                    yield event

            # After execution, update span with correct call type and output
            if last_event:
                # We need to check if we should merge content from an earlier event.
                output = _event_output_with_content(last_event, event_with_content)

                # Extract metrics from response
                metrics = _extract_metrics(last_event)

                # Add time to first token if we captured it
                if first_token_time is not None:
                    if metrics is None:
                        metrics = {}
                    metrics["time_to_first_token"] = first_token_time - start_time

                # Determine the actual call type based on the response
                call_type = _determine_llm_call_type(llm_request, last_event)

                # Update span name with the specific call type now that we know it
                llm_span.set_attributes(
                    name=f"llm_call [{call_type}]",
                    span_attributes={"llm_call_type": call_type},
                )

                # Log output and metrics (span.log will handle serialization)
                llm_span.log(output=output, metrics=metrics)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _runner_run_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    user_id = kwargs.get("user_id")
    session_id = kwargs.get("session_id")
    new_message = kwargs.get("new_message")
    state_delta = kwargs.get("state_delta")

    # Serialize new_message before any dict conversion to handle binary data
    serialized_message = _serialize_content(new_message) if new_message else None

    async def _trace():
        with start_span(
            name=f"invocation [{instance.app_name}]",
            type=SpanTypeAttribute.TASK,
            input={"new_message": serialized_message},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "state_delta": state_delta,
                **_omit(kwargs, ["user_id", "session_id", "new_message", "state_delta"]),
            },
        ) as runner_span:
            last_event = None
            async with aclosing(wrapped(*args, **kwargs)) as agen:
                async for event in agen:
                    if event.is_final_response():
                        last_event = event
                    yield event
            if last_event:
                runner_span.log(output=last_event)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _tool_call_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    tool = args[0] if len(args) > 0 else kwargs.get("tool")
    tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})

    # MCP tools already have a dedicated wrapper. Skip here to avoid duplicate tool spans.
    if tool is not None and getattr(tool.__class__, "__module__", "").startswith("google.adk.tools.mcp_tool"):
        return await wrapped(*args, **kwargs)

    tool_name = getattr(tool, "name", tool.__class__.__name__ if tool is not None else "unknown")

    with start_span(
        name=f"tool [{tool_name}]",
        type=SpanTypeAttribute.TOOL,
        input={"tool_name": tool_name, "arguments": tool_args},
        metadata={"tool_class": tool.__class__.__name__ if tool is not None else None},
    ) as tool_span:
        try:
            result = await wrapped(*args, **kwargs)
            tool_span.log(output=result)
            return result
        except Exception as e:
            tool_span.log(error=str(e))
            raise


async def _mcp_tool_run_async_wrapper_async(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    # Extract tool information
    tool_name = instance.name
    tool_args = kwargs.get("args", {})

    with start_span(
        name=f"mcp_tool [{tool_name}]",
        type=SpanTypeAttribute.TOOL,
        input={"tool_name": tool_name, "arguments": tool_args},
        metadata=_omit(kwargs, ["args"]),
    ) as tool_span:
        try:
            result = await wrapped(*args, **kwargs)
            tool_span.log(output=result)
            return result
        except Exception as e:
            # Log error to span but re-raise for ADK to handle
            tool_span.log(error=str(e))
            raise
