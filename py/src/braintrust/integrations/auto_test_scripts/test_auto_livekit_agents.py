"""Test auto_instrument for LiveKit Agents."""

import inspect

from braintrust.auto import auto_instrument
from wrapt import FunctionWrapper


def _is_braintrust_wrapped(target, attr: str) -> bool:
    return isinstance(inspect.getattr_static(target, attr, None), FunctionWrapper)


# Import the provider classes before auto-instrumentation to verify setup handles
# normal user import order in a fresh process.
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.inference.llm import LLMStream  # noqa: E402
from livekit.agents.stt import STT  # noqa: E402
from livekit.agents.tts import TTS  # noqa: E402
from livekit.agents.voice import generation  # noqa: E402
from livekit.agents.voice.io import AudioOutput  # noqa: E402


assert not _is_braintrust_wrapped(AgentSession, "run")
assert not _is_braintrust_wrapped(AgentSession, "_on_audio_output_changed")
assert not _is_braintrust_wrapped(AgentSession, "_update_user_state")
assert not isinstance(generation._execute_tools_task, FunctionWrapper)
assert not _is_braintrust_wrapped(LLMStream, "_run")
assert not _is_braintrust_wrapped(STT, "recognize")
assert not _is_braintrust_wrapped(TTS, "synthesize")
assert not _is_braintrust_wrapped(AudioOutput, "capture_frame")

results = auto_instrument()
assert results.get("livekit_agents") is True
assert _is_braintrust_wrapped(AgentSession, "run")
assert _is_braintrust_wrapped(AgentSession, "_on_audio_output_changed")
assert _is_braintrust_wrapped(AgentSession, "_update_user_state")
assert isinstance(generation._execute_tools_task, FunctionWrapper)
assert _is_braintrust_wrapped(LLMStream, "_run")
assert _is_braintrust_wrapped(STT, "recognize")
assert _is_braintrust_wrapped(TTS, "synthesize")
assert _is_braintrust_wrapped(AudioOutput, "capture_frame")

# Idempotent.
results2 = auto_instrument()
assert results2.get("livekit_agents") is True

print("SUCCESS")
