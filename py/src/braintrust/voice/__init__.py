"""Voice evals with managed audio transports and distributed tracing.

Install braintrust[voice], plus your agent framework. Use voice_task in an Eval,
forward its call context to the agent, and use connect(call) in that process.
"""

from ._call import run_voice_call as run_voice_call
from ._call import voice_task as voice_task
from ._livekit import livekit_agent as livekit_agent
from ._session import VoiceSession as VoiceSession
from ._session import connect as connect
from ._types import LiveKitAgent as LiveKitAgent
from ._types import VoiceCall as VoiceCall
from ._types import VoiceCallResult as VoiceCallResult
