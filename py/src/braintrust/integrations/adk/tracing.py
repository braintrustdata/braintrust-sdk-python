"""ADK-specific span creation, metadata extraction, stream handling, and output normalization."""

import asyncio
import contextvars
import inspect
import json
import logging
import time
from contextlib import aclosing, contextmanager
from functools import lru_cache
from itertools import chain
from typing import Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.utils import _materialize_attachment
from braintrust.logger import start_span as _bt_start_span


_INSTRUMENTATION = "adk-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


@contextmanager
def _start_stream_span(*args, **kwargs):
    """Keep ADK stream cleanup signals from being logged as span failures."""
    cleanup_signal = None
    with start_span(*args, **kwargs) as span:
        try:
            yield span
        except (GeneratorExit, asyncio.CancelledError) as exc:
            cleanup_signal = exc

    if cleanup_signal is not None:
        raise cleanup_signal


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


_CAPTURED_CONFIG_FIELDS = (
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
)


def _capture_config(config: Any) -> dict[str, Any] | Any:
    """Capture the ADK config fields that make LLM spans readable."""
    if config is None or not config:
        return config

    captured: dict[str, Any] = {}
    for field in _CAPTURED_CONFIG_FIELDS:
        value = getattr(config, field, None)
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


def _part_has_field(part: Any, *field_names: str) -> bool:
    return any(getattr(part, field_name, None) is not None for field_name in field_names)


def _capture_llm_request_input(llm_request: Any) -> Any:
    """Capture the ADK request fields that make LLM spans readable."""
    if llm_request is None:
        return None

    contents = getattr(llm_request, "contents", None)
    config = getattr(llm_request, "config", None)
    model = getattr(llm_request, "model", None)

    captured: dict[str, Any] = {}
    if model:
        captured["model"] = model
    if contents:
        captured["contents"] = (
            [_serialize_content(c) for c in contents] if isinstance(contents, list) else _serialize_content(contents)
        )
    if config:
        captured["config"] = _capture_config(config)
    if hasattr(llm_request, "live_connect_config"):
        captured["live_connect_config"] = getattr(llm_request, "live_connect_config", None)

    return captured or llm_request


def _extract_tool_metadata(llm_request: Any) -> dict[str, Any]:
    """Extract tool definitions and tool_config from an ADK LLM request for metadata.tools.

    Google-native shape is preserved; ADK is Google-backed and metadata.provider="google"
    lets the UI apply the Google normalizer.
    """
    if llm_request is None:
        return {}
    config = getattr(llm_request, "config", None)
    if config is None:
        return {}
    result: dict[str, Any] = {}
    tools = getattr(config, "tools", None)
    if tools:
        result["tools"] = tools
    tool_config = getattr(config, "tool_config", None)
    if tool_config:
        result["tool_config"] = tool_config
    return result


def _event_output_with_content(last_event: Any, event_with_content: Any | None) -> Any:
    if event_with_content is None or getattr(last_event, "content", None) is not None:
        return last_event

    content = getattr(event_with_content, "content", None)
    if content is None:
        return last_event

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
            for content in (getattr(llm_request, "contents", None) or [])
            for part in (getattr(content, "parts", None) or [])
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
                content = getattr(model_response, "content", None)
                response_has_function_call = any(
                    _part_has_field(part, "function_call", "functionCall")
                    for part in chain(
                        getattr(content, "parts", None) or [], getattr(model_response, "parts", None) or []
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
    with _start_stream_span(
        name=f"agent_run [{instance.name}]",
        type=SpanTypeAttribute.TASK,
        metadata={"agent_name": instance.name},
    ) as agent_span:
        last_event = None
        async with aclosing(wrapped(*args, **kwargs)) as agen:
            async for event in agen:
                if event.is_final_response():
                    last_event = event
                yield event
        if last_event:
            agent_span.log(output=last_event)


async def _workflow_node_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Trace ADK 2.x workflow nodes, including the Workflow root node."""
    if any(
        cls.__name__ == "BaseAgent" and cls.__module__.startswith("google.adk.agents")
        for cls in instance.__class__.__mro__
    ):
        # Newer ADK agent classes share BaseNode.run; their existing agent span
        # is the canonical task span and must remain the direct child of Runner.
        async with aclosing(wrapped(*args, **kwargs)) as agen:
            async for event in agen:
                yield event
        return

    node_name = getattr(instance, "name", instance.__class__.__name__)
    node_input = kwargs.get("node_input", args[1] if len(args) > 1 else None)
    is_workflow = any(
        cls.__name__ == "Workflow" and cls.__module__.startswith("google.adk.workflow")
        for cls in instance.__class__.__mro__
    )
    span_name = f"workflow [{node_name}]" if is_workflow else f"workflow_node [{node_name}]"

    async def _trace():
        with _start_stream_span(
            name=span_name,
            type=SpanTypeAttribute.TASK,
            input=node_input,
            metadata={"node_name": node_name, "node_class": instance.__class__.__name__},
        ) as node_span:
            last_output = None
            async with aclosing(wrapped(*args, **kwargs)) as agen:
                async for event in agen:
                    event_output = getattr(event, "output", None)
                    if event_output is not None:
                        last_output = event_output
                    yield event
            if last_output is not None:
                node_span.log(output=last_output)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _workflow_tool_node_run_impl_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Trace tools executed directly by ADK 2.x workflow tool nodes."""
    tool = getattr(instance, "tool", None)
    if tool is None:
        async with aclosing(wrapped(*args, **kwargs)) as agen:
            async for event in agen:
                yield event
        return

    # MCP tools already have their own wrapper around run_async.
    if getattr(tool.__class__, "__module__", "").startswith("google.adk.tools.mcp_tool"):
        async with aclosing(wrapped(*args, **kwargs)) as agen:
            async for event in agen:
                yield event
        return

    tool_name = getattr(tool, "name", tool.__class__.__name__)
    node_input = kwargs.get("node_input", args[1] if len(args) > 1 else None)
    tool_args = node_input
    if hasattr(tool_args, "parts"):
        tool_args = "".join(getattr(part, "text", "") or "" for part in (tool_args.parts or []))
    if hasattr(tool_args, "model_dump"):
        tool_args = tool_args.model_dump()
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            pass
    if not isinstance(tool_args, dict):
        tool_args = {}

    async def _trace():
        with _start_stream_span(
            name=f"tool [{tool_name}]",
            type=SpanTypeAttribute.TOOL,
            input={"tool_name": tool_name, "arguments": tool_args},
            metadata={"tool_class": tool.__class__.__name__},
        ) as tool_span:
            last_output = None
            try:
                async with aclosing(wrapped(*args, **kwargs)) as agen:
                    async for event in agen:
                        if getattr(event, "output", None) is not None:
                            last_output = event.output
                        yield event
            except Exception as error:
                tool_span.log(error=error)
                raise
            if last_output is not None:
                tool_span.log(output=last_output)

    async with aclosing(_trace()) as agen:
        async for event in agen:
            yield event


async def _flow_run_async_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    async def _trace():
        with _start_stream_span(
            name="call_llm",
            type=SpanTypeAttribute.TASK,
            metadata={"flow_class": instance.__class__.__name__},
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
    llm_request = args[1] if len(args) > 1 else kwargs.get("llm_request")

    async def _trace():
        # Capture only the fields we need to alter: contents may contain binary
        # data that should become Attachments, and config may contain Pydantic
        # schema classes that are clearer as JSON schema.
        captured_request = _capture_llm_request_input(llm_request)

        # Extract model name from request or instance
        model_name = _extract_model_name(None, llm_request, instance)

        metadata: dict[str, Any] = {
            "flow_class": instance.__class__.__name__,
            "model": model_name,
            "provider": "google",
        }
        metadata.update(_extract_tool_metadata(llm_request))

        # Create span BEFORE execution so child spans (like mcp_tool) have proper parent
        # Start with generic name - we'll update it after we see the response
        # A task, not an llm span: the provider client ADK calls (google-genai,
        # litellm, ...) owns the llm span and its usage.
        with _start_stream_span(
            name="llm_call",
            type=SpanTypeAttribute.TASK,
            input=captured_request,
            metadata=metadata,
        ) as model_call_span:
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

                # Determine the actual call type based on the response
                call_type = _determine_llm_call_type(llm_request, last_event)

                # Update span name with the specific call type now that we know it
                model_call_span.set_attributes(
                    name=f"llm_call [{call_type}]",
                    span_attributes={"llm_call_type": call_type},
                )

                model_call_span.log(output=output, metrics={"time_to_first_token": first_token_time - start_time})

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
        with _start_stream_span(
            name=f"invocation [{instance.app_name}]",
            type=SpanTypeAttribute.TASK,
            input={"new_message": serialized_message},
            metadata={
                "app_name": instance.app_name,
                "user_id": user_id,
                "session_id": session_id,
                "state_delta": state_delta,
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
            tool_span.log(error=e)
            raise


async def _mcp_tool_run_async_wrapper_async(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    # Extract tool information
    tool_name = instance.name
    tool_args = kwargs.get("args", {})

    with start_span(
        name=f"mcp_tool [{tool_name}]",
        type=SpanTypeAttribute.TOOL,
        input={"tool_name": tool_name, "arguments": tool_args},
        metadata={"tool_class": instance.__class__.__name__},
    ) as tool_span:
        try:
            result = await wrapped(*args, **kwargs)
            tool_span.log(output=result)
            return result
        except Exception as e:
            # Log error to span but re-raise for ADK to handle
            tool_span.log(error=e)
            raise
