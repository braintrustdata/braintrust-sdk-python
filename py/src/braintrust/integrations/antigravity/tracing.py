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
_active_chat: contextvars.ContextVar[Any] = contextvars.ContextVar("braintrust_antigravity_chat", default=None)
_agent_config: contextvars.ContextVar[Any] = contextvars.ContextVar("braintrust_antigravity_config", default=None)
_pending: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_tool_chats: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


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


def _prompt_messages(prompt):
    parts = []
    for part in prompt if isinstance(prompt, (list, tuple)) else [prompt]:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif hasattr(part, "data") and hasattr(part, "mime_type"):
            resolved = _materialize_attachment(part.data, mime_type=part.mime_type)
            if resolved is not None:
                if part.mime_type.startswith("image/"):
                    parts.append({"type": "image_url", "image_url": {"url": resolved.attachment}})
                else:
                    parts.append(
                        {"type": "file", "file": {"file_data": resolved.attachment, "filename": resolved.filename}}
                    )
            else:
                parts.append(part)
        elif part is not None:
            parts.append(part)
    return [{"role": "user", "content": parts}]


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
        self.error_logged = False
        self.first_token = False
        if tool_runner is not None:
            _tool_chats[tool_runner] = weakref.ref(self)

    def observe_step(self, step):
        key = (step.trajectory_id, step.step_index)
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

    def observe_chunk(self, chunk):
        if self.done:
            return
        # Thought and Text both have text; only Text is conversational output.
        from google.antigravity.types import Text

        if isinstance(chunk, Text) and chunk.text:
            self.text.append(chunk.text)
            if not self.first_token:
                self.span.log(metrics={"time_to_first_token": time.monotonic() - self.started})
                self.first_token = True

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
                metrics = _extract_google_usage_metadata_metrics(conversation.last_turn_usage)
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
        token = _active_chat.set(self.state)
        try:
            self.state.span.set_current()
            try:
                chunk = await self.stream.__anext__()
            finally:
                self.state.span.unset_current()
            try:
                self.state.observe_chunk(chunk)
            except Exception:
                logger.debug("Could not trace Antigravity chunk", exc_info=True)
            return chunk
        except StopAsyncIteration:
            self.state.finish()
            raise
        except BaseException as exc:
            self.state.finish(exc)
            raise
        finally:
            _active_chat.reset(token)

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            self.state.finish()


async def _chat_wrapper(wrapped, instance, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    _finish_pending(instance)
    state = None
    try:
        state = _ChatState(instance, prompt)
        _pending[instance] = weakref.ref(state)
    except Exception:
        logger.debug("Could not start Antigravity trace", exc_info=True)
    try:
        response = await wrapped(*args, **kwargs)
    except BaseException as exc:
        if state is not None:
            state.finish(exc)
        raise
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


async def _receive_steps_wrapper(wrapped, instance, args, kwargs):
    state = _active_chat.get()
    async for step in wrapped(*args, **kwargs):
        if state is not None and not state.done:
            try:
                state.observe_step(step)
            except Exception:
                logger.debug("Could not trace Antigravity step", exc_info=True)
        yield step


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
    state_ref = _tool_chats.get(instance)
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
        span.log(output=result)
        return result
