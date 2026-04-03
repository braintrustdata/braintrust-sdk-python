"""Braintrust integration for LangSmith."""

import logging
import os

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import LangSmithIntegration


logger = logging.getLogger(__name__)

__all__ = [
    "LangSmithIntegration",
    "setup_langsmith",
]


def setup_langsmith(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    standalone: bool = False,
) -> bool:
    """Setup Braintrust integration with LangSmith."""
    resolved_project_name = project_name or os.environ.get("LANGCHAIN_PROJECT")
    if current_span() == NOOP_SPAN:
        init_logger(project=resolved_project_name, api_key=api_key, project_id=project_id)

    try:
        import langsmith  # noqa: F401
    except ImportError as exc:
        logger.error("Failed to import langsmith: %s", exc)
        logger.error("langsmith is not installed. Please install it with: pip install langsmith")
        return False

    logger.info("LangSmith integration with Braintrust enabled")
    return LangSmithIntegration.setup(standalone=True if standalone else None)
