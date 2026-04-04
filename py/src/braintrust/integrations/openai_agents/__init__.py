"""Braintrust integration for the OpenAI Agents SDK."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import OpenAIAgentsIntegration


try:
    from .tracing import BraintrustTracingProcessor
except ImportError as exc:  # pragma: no cover - optional dependency not installed
    _IMPORT_ERROR = exc

    class BraintrustTracingProcessor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("openai-agents is required for braintrust.integrations.openai_agents") from _IMPORT_ERROR


__all__ = ["BraintrustTracingProcessor", "OpenAIAgentsIntegration", "setup_openai_agents"]


def setup_openai_agents(
    api_key: str | None = None,
    project_id: str | None = None,
    project: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Setup Braintrust tracing for the OpenAI Agents SDK."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name or project, api_key=api_key, project_id=project_id)

    return OpenAIAgentsIntegration.setup()
