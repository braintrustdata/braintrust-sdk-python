"""Braintrust integration for Strands Agents."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import StrandsIntegration
from .patchers import wrap_strands_tracer


__all__ = ["StrandsIntegration", "setup_strands", "wrap_strands_tracer"]


def setup_strands(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust tracing for Strands Agents."""
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return StrandsIntegration.setup()
