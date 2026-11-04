"""Auto setup uses the same framework-level cascade hooks after imports."""

import inspect

from braintrust.auto import auto_instrument
from livekit.agents import AgentSession
from livekit.agents.voice import generation
from livekit.agents.voice.agent_activity import AgentActivity
from wrapt import FunctionWrapper


assert not isinstance(inspect.getattr_static(AgentSession, "start"), FunctionWrapper)
assert auto_instrument().get("livekit_agents") is True
assert isinstance(inspect.getattr_static(AgentSession, "start"), FunctionWrapper)
assert isinstance(inspect.getattr_static(AgentSession, "_aclose_impl"), FunctionWrapper)
assert isinstance(inspect.getattr_static(AgentActivity, "push_audio"), FunctionWrapper)
assert isinstance(generation._llm_inference_task, FunctionWrapper)
assert isinstance(generation._tts_inference_task, FunctionWrapper)
assert auto_instrument().get("livekit_agents") is True
print("SUCCESS")
