"""Pipecat cascade tracing.

``observer`` routes native frames to three trackers sharing a session clock and
span factory: ``_user`` (user turns, speech, recognition), ``_agent`` (agent
turns and the work they own) and ``_context`` (model input correlation).
"""

from .observer import BraintrustPipecatObserver


__all__ = ["BraintrustPipecatObserver"]
