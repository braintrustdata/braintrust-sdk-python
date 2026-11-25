"""Voice evals with managed audio transports and distributed tracing.

Install braintrust[voice]. Use voice_task in an Eval,
forward its call context to the agent, and use connect(call) in that process.
"""

from ._call import run_voice_call as run_voice_call
from ._call import voice_task as voice_task
from ._session import VoiceSession as VoiceSession
from ._session import connect as connect
from ._types import VoiceCall as VoiceCall
from ._types import VoiceCallResult as VoiceCallResult
