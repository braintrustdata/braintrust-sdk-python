"""Observe chat consumption without changing ChatResponse or its cursors.

The subprocess reports execution steps and turn-level usage, not individual
model requests. This is partial agentic instrumentation: the spec's per-call
LLM spans and built-in tool return values cannot be recovered from this wire
protocol. Model steps are tasks; only the chat task owns aggregate usage.
The separate send/receive API is deliberately excluded from instrumentation.
"""

import contextvars
import logging
import time
import weakref
from typing import Any

from braintrust.integrations.utils import _extract_google_usage_metadata_metrics, _materialize_attachment
from braintrust.logger import start_span as _bt_start_span


logger = logging.getLogger(__name__)
_INSTRUMENTATION = "antigravity-auto"
# Metadata allowlist: model, tools, conversation_id, trajectory_id, step_index.
# Function definitions allow only name, description and parameters. Never log
# AgentConfig wholesale: endpoints and other config fields contain credentials.
_starting_chat: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "braintrust_antigravity_starting_chat", default=None
)
_agent_config: contextvars.ContextVar[Any] = contextvars.ContextVar("braintrust_antigravity_config", default=None)
_pending: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
# The event processor and tool runner dispatch from tasks created before chat().
# Their ownership changes only when the connection begins sending a new turn.
_execution_chats: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


def _finish_pending(conversation, error=None):
    if conversation is None:
        return
    state_ref = _pending.pop(conversation, None)
    state = state_ref() if state_ref is not None else None
    if state is not None and not state.done:
        state.finish(error)


def _materialize_media(value):
    from google.antigravity import types

    if isinstance(value, (types.Image, types.Document, types.Audio, types.Video)):
        try:
            resolved = _materialize_attachment(value.data, mime_type=value.mime_type)
        except Exception:
            logger.debug("Could not materialize Antigravity media", exc_info=True)
            return value
        if resolved is None:
            return value
        part = {"type": "image_url" if resolved.is_image else "file", **resolved.multimodal_part_payload}
        if value.description is not None:
            part["description"] = value.description
        return part
    if isinstance(value, dict):
        return {key: _materialize_media(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_media(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_media(item) for item in value)
    return value


def _prompt_messages(prompt):
    parts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": part} if isinstance(part, str) else _materialize_media(part)
                for part in parts
                if part is not None
            ],
        }
    ]


class _ChatState:
    def __init__(self, conversation, prompt):
        self.conversation = weakref.ref(conversation)
        self.started = time.monotonic()
        messages = _prompt_messages(prompt)
        metadata = {}
        config = _agent_config.get()
        if config is not None:
            instructions = config.system_instructions
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            model = getattr(config, "model", None)
            if isinstance(model, str):
                metadata["model"] = model
            capabilities = getattr(config, "capabilities", None)
            enabled_tools = getattr(capabilities, "enabled_tools", None)
            if enabled_tools:
                metadata["tools"] = [{"type": getattr(tool, "value", tool)} for tool in enabled_tools]
        self.client_tool_names: set[str] = set()
        tool_runner = getattr(conversation.connection, "_tool_runner", None)
        if tool_runner is not None:
            self.client_tool_names.update(tool_runner.tool_names)
            tools = metadata.get("tools", [])
            try:
                from google.antigravity.connections.local.interactions_config_converter import (
                    callable_to_function_tool_dict,
                )

                for tool in tool_runner.tools.values():
                    definition = callable_to_function_tool_dict(tool, tool_runner=tool_runner)
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                k: definition[k] for k in ("name", "description", "parameters") if k in definition
                            },
                        }
                    )
            except Exception:
                logger.debug("Could not read Antigravity tool definitions", exc_info=True)
            if tools:
                metadata["tools"] = tools
        self.span = start_span(name="antigravity.chat", type="task", input=messages, metadata=metadata)
        self.steps: dict[Any, Any] = {}
        self.text: list[str] = []
        self.structured_output = None
        self.done = False
        self.sent = False
        self.usage_start = None
        self.error_logged = False
        self.first_token = False

    def activate(self):
        conversation = self.conversation()
        if conversation is None:
            return
        self.usage_start = conversation.total_usage
        _pending[conversation] = weakref.ref(self)
        connection = conversation.connection
        for owner in (getattr(connection, "_processor", None), getattr(connection, "_tool_runner", None)):
            if owner is not None:
                _execution_chats[owner] = weakref.ref(self)

    def observe_step(self, step):
        key = (step.trajectory_id, step.step_index)
        # Mirror receive_chunks(): an implicit SDK drain consumes steps without
        # producing ChatResponse chunks, but those steps still belong here.
        if step.source == "MODEL" and step.target == "TARGET_USER" and step.status != "ERROR" and step.content_delta:
            self.text.append(step.content_delta)
            if not self.first_token:
                self.span.log(metrics={"time_to_first_token": time.monotonic() - self.started})
                self.first_token = True
        if step.type == "FINISH" and step.status == "DONE" and step.structured_output is not None:
            self.structured_output = step.structured_output
        if step.source == "SYSTEM" and step.error:
            self.span.log(error=step.error)
        if step.source in ("USER", "SYSTEM") or step.type == "UNKNOWN":
            return
        # The SDK emits both step updates and a separate call notification
        # for Python tools. ToolRunner.execute owns their spans and results.
        if step.type == "TOOL_CALL":
            if not step.tool_calls or all(call.name in self.client_tool_names for call in step.tool_calls):
                return
        entry = self.steps.get(key)
        if entry is None:
            call = step.tool_calls[0] if step.tool_calls else None
            span = self.span.start_span(
                name=getattr(call.name, "value", call.name) if call else f"antigravity.{step.type.value.lower()}",
                type="tool" if call else "task",
                input=call.args if call else None,
                metadata={"trajectory_id": step.trajectory_id, "step_index": step.step_index},
                internal={"instrumentation": _INSTRUMENTATION},
            )
            entry = self.steps[key] = {"span": span, "done": False, "tool": call is not None}
        if entry["done"]:
            return
        span = entry["span"]
        if (step.content or step.thinking) and not entry["tool"]:
            span.log(output={"role": "assistant", "content": step.content, "reasoning": step.thinking})
        if step.error:
            span.log(error=step.error)
        if step.status in ("DONE", "ERROR", "CANCELED"):
            if step.structured_output is not None:
                span.log(output=step.structured_output)
            span.end()
            entry["done"] = True

    def finish(self, error=None):
        # cancel() ends timing before the SDK cursor raises its cancellation
        # exception. Preserve that exact exception if consumption observes it.
        if error is not None and not self.error_logged:
            self.span.log(error=error)
            self.error_logged = True
        if self.done:
            return
        self.done = True
        output = (
            self.structured_output
            if self.structured_output is not None
            else [{"role": "assistant", "content": "".join(self.text)}]
        )
        self.span.log(output=output)
        conversation = self.conversation()
        metrics = {}
        metadata = {}
        try:
            if conversation is not None and self.sent:
                usage = conversation.total_usage
                # Conversation.send resets last_turn_usage before calling the
                # transport. Keep our own baseline so a drained turn retains
                # its usage even after that reset.
                if self.usage_start is not None and usage.total_token_count != self.usage_start.total_token_count:
                    metrics = _extract_google_usage_metadata_metrics(usage - self.usage_start)
                metrics = {
                    k: v for k, v in metrics.items() if isinstance(v, int) and not isinstance(v, bool) and v >= 0
                }
                metadata = {"conversation_id": conversation.conversation_id}
        except Exception:
            logger.debug("Could not read Antigravity turn usage", exc_info=True)
        self.span.log(metrics=metrics, metadata=metadata)
        for entry in self.steps.values():
            if not entry["done"]:
                entry["span"].end()
                entry["done"] = True
        self.span.end()


class _ChatStream:
    """Wrap only the upstream cursor; preserve the SDK's shared buffering."""

    def __init__(self, stream, state):
        self.stream = stream
        self.state = state
        self._finalizer = weakref.finalize(self, state.finish)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            self.state.span.set_current()
            try:
                return await self.stream.__anext__()
            finally:
                self.state.span.unset_current()
        except StopAsyncIteration:
            self.state.finish()
            raise
        except BaseException as exc:
            self.state.finish(exc)
            raise

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            self.state.finish()


async def _chat_wrapper(wrapped, instance, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    state = None
    try:
        state = _ChatState(instance, prompt)
    except Exception:
        logger.debug("Could not start Antigravity trace", exc_info=True)
    token = _starting_chat.set(state)
    try:
        response = await wrapped(*args, **kwargs)
    except BaseException as exc:
        if state is not None:
            state.finish(exc)
        raise
    finally:
        _starting_chat.reset(token)
    try:
        if state is None:
            return response
        state.sent = True
        response._chunk_stream = _ChatStream(response._chunk_stream, state)
    except Exception:
        if state is not None:
            state.finish()
        logger.debug("Could not instrument Antigravity response", exc_info=True)
    return response


async def _connection_send_wrapper(wrapped, instance, args, kwargs):
    try:
        state = _starting_chat.get()
        previous_ref = _execution_chats.get(instance._processor)
        previous = previous_ref() if previous_ref is not None else None
        if previous is not state:
            if previous is not None:
                # Conversation.send() has now drained/waited for the old turn,
                # before the connection can dispatch work for the next prompt.
                _finish_pending(previous.conversation())
            conversation = state.conversation() if state is not None else None
            if conversation is not None and conversation.connection is instance:
                state.activate()
    except Exception:
        logger.debug("Could not trace Antigravity turn boundary", exc_info=True)
    return await wrapped(*args, **kwargs)


async def _emit_step_wrapper(wrapped, instance, args, kwargs):
    state_ref = _execution_chats.get(instance)
    state = state_ref() if state_ref is not None else None
    if state is not None and not state.done:
        try:
            state.observe_step(args[0] if args else kwargs["step_obj"])
        except Exception:
            logger.debug("Could not trace Antigravity step", exc_info=True)
    return await wrapped(*args, **kwargs)


async def _lifecycle_wrapper(wrapped, instance, args, kwargs):
    error = None
    try:
        return await wrapped(*args, **kwargs)
    except BaseException as exc:
        error = exc
        raise
    finally:
        _finish_pending(instance, error)


async def _agent_chat_wrapper(wrapped, instance, args, kwargs):
    token = _agent_config.set(instance._config)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _agent_config.reset(token)


async def _agent_exit_wrapper(wrapped, instance, args, kwargs):
    return await _lifecycle_wrapper(wrapped, instance._conversation, args, kwargs)


async def _tool_execute_wrapper(wrapped, instance, args, kwargs):
    state_ref = _execution_chats.get(instance)
    state = state_ref() if state_ref is not None else None
    if state is None or state.done:
        return await wrapped(*args, **kwargs)
    name = args[0] if args else kwargs.get("tool_name")
    tool_input = dict(kwargs)
    if not args:
        tool_input.pop("tool_name", None)
    with state.span.start_span(
        name=name, type="tool", input=tool_input, internal={"instrumentation": _INSTRUMENTATION}
    ) as span:
        result = await wrapped(*args, **kwargs)
        try:
            output = _materialize_media(result)
        except Exception:
            logger.debug("Could not materialize Antigravity tool output", exc_info=True)
            output = result
        span.log(output=output)
        return result
