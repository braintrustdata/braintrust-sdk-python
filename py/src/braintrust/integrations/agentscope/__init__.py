"""Braintrust integration for AgentScope."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AgentScopeIntegration


__all__ = ["AgentScopeIntegration", "setup_agentscope"]


def setup_agentscope(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Setup Braintrust integration with AgentScope."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return AgentScopeIntegration.setup()
