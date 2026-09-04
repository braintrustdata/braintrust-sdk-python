"""Test auto_instrument for LiveKit Agents."""

import inspect

from braintrust.integrations.test_utils import run_auto_smoke

# Import the provider classes before auto-instrumentation to verify setup handles
# normal user import order in a fresh process.
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.inference.llm import LLMStream  # noqa: E402
from livekit.agents.stt import STT  # noqa: E402
from livekit.agents.tts import TTS  # noqa: E402
from livekit.agents.voice import generation  # noqa: E402
from livekit.agents.voice.io import AudioOutput  # noqa: E402
from wrapt import FunctionWrapper


def _attr_wrapped(target, attr: str) -> bool:
    return isinstance(inspect.getattr_static(target, attr, None), FunctionWrapper)


_WRAP_TARGETS = (
    (AgentSession, "run"),
    (AgentSession, "_on_audio_output_changed"),
    (AgentSession, "_update_user_state"),
    (LLMStream, "_run"),
    (STT, "recognize"),
    (TTS, "synthesize"),
    (AudioOutput, "capture_frame"),
)


def _is_patched() -> bool:
    return all(_attr_wrapped(target, attr) for target, attr in _WRAP_TARGETS) and isinstance(
        generation._execute_tools_task, FunctionWrapper
    )


run_auto_smoke("livekit_agents", is_patched=_is_patched)
print("SUCCESS")
