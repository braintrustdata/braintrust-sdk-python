"""Auto setup before customer imports reaches the same cascade hooks."""

import asyncio
import inspect

from braintrust.auto import auto_instrument
from wrapt import FunctionWrapper


assert auto_instrument().get("livekit_agents") is True
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.voice import generation  # noqa: E402


assert isinstance(inspect.getattr_static(AgentSession, "start"), FunctionWrapper)
assert isinstance(generation._llm_inference_task, FunctionWrapper)
from test_auto_livekit_agents import assert_exported_eou_wait  # noqa: E402


asyncio.run(assert_exported_eou_wait())
print("SUCCESS")
