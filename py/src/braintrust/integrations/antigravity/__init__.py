"""Braintrust tracing for the Google Antigravity SDK.

Instrument before creating an Agent, then consume its ChatResponse normally.
"""

from typing import Any

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AntigravityIntegration
from .patchers import AgentPatcher


__all__ = ["AntigravityIntegration", "setup_antigravity", "wrap_antigravity_agent"]


def setup_antigravity(
    api_key: str | None = None,
    project_id: str | None = None,
    project: str | None = None,
) -> bool:
    """Enable tracing for Antigravity chat responses."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project, api_key=api_key, project_id=project_id)
    return AntigravityIntegration.setup()


def wrap_antigravity_agent(agent_class: Any) -> Any:
    """Instrument the Conversation surface used by an Antigravity Agent.

    Patches in place, including Agent classes imported before this call.
    """
    AntigravityIntegration.setup()
    return AgentPatcher.wrap_target(agent_class)
