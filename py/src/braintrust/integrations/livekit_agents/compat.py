"""Private hooks qualified against livekit-agents==1.8.10 only."""

import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from livekit.agents.llm import ChatContext


QUALIFIED_VERSION = "1.8.10"
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
