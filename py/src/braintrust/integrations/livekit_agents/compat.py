"""Private hooks qualified against livekit-agents 1.8.9–1.8.10."""

import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from livekit.agents.llm import ChatContext


STATE_ATTR = "__braintrust_livekit_cascade__"


def arguments(wrapped, args, kwargs):
    return inspect.signature(wrapped).bind_partial(*args, **kwargs).arguments


def state_for(instance):
    session = getattr(instance, "_session", instance)
    return getattr(session, STATE_ATTR, None)


def chat_messages(context: "ChatContext") -> list[dict[str, Any]]:
    # Capture observed full context at the operation boundary. Never retain it
    # for later relationship resolution, and never serialize inline media.
    messages: list[dict[str, Any]] = []
    for item in context.items:
        if item.type == "message":
            messages.append({"role": item.role, "content": item.raw_text_content})
        elif item.type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {"name": item.name, "arguments": item.arguments},
                        }
                    ],
                }
            )
        elif item.type == "function_call_output":
            messages.append({"role": "tool", "content": item.output, "tool_call_id": item.call_id})
    return messages


def detach_output(tap):
    downstream = tap.next_in_chain
    if downstream is not None:
        downstream.off("playback_finished", tap._dispatch_playback_finished)
        downstream.off("playback_started", tap._forward_next_playback_started)
        downstream.off("playback_progressed", tap._forward_next_playback_progressed)


# These concrete adapters call APIs covered by our provider integrations.
# This is a capability decision, independent of setup order or observed spans.
_PROVIDER_LLM_ADAPTERS = frozenset(
    {
        "livekit.plugins.openai.llm",
        "livekit.plugins.anthropic.llm",
        "livekit.plugins.google.llm",
    }
)


def provider_llm_expected(node: Any) -> bool:
    agent = getattr(node, "__self__", None)
    activity = getattr(agent, "_activity", None)
    model = getattr(activity, "llm", None)
    return any(cls.__module__ in _PROVIDER_LLM_ADAPTERS for cls in type(model).__mro__)


def llm_metadata(values: dict[str, Any]) -> dict[str, Any]:
    from livekit.agents.llm._provider_format.openai import to_fnc_ctx

    metadata = {key: values[key] for key in ("model", "provider") if values.get(key)}
    tools = to_fnc_ctx(values["tool_ctx"], strict=False)
    if tools:
        metadata["tools"] = tools
    choice = getattr(values.get("model_settings"), "tool_choice", None)
    if isinstance(choice, (str, dict)):
        metadata["tool_choice"] = choice
    return metadata
