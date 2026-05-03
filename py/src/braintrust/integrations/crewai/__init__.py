"""Braintrust integration for CrewAI.

Public entry points:

- :func:`setup_crewai` — mirrors :func:`setup_agno`/``setup_dspy`` ergonomics.
  Initializes a Braintrust logger (when one is not already set) and
  registers the CrewAI event-bus listener.
- :func:`patch_crewai` — thin setup-only helper (no logger init), matches
  the ``patch_*`` naming used elsewhere.
- :class:`CrewAIIntegration` — used by :func:`braintrust.auto_instrument`.
- :class:`BraintrustCrewAIListener` — exposed for advanced users who want
  to register the listener manually on their own event-bus instance (e.g.
  for tests).
"""

import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import CrewAIIntegration
from .tracing import BraintrustCrewAIListener


logger = logging.getLogger(__name__)


__all__ = [
    "BraintrustCrewAIListener",
    "CrewAIIntegration",
    "patch_crewai",
    "setup_crewai",
]


def setup_crewai(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust tracing for CrewAI.

    Initializes a Braintrust logger (unless one is already active) and
    registers :class:`BraintrustCrewAIListener` on the singleton
    ``crewai_event_bus``.  Safe to call multiple times.

    Args:
        api_key: Braintrust API key (optional, ``BRAINTRUST_API_KEY`` env var works too).
        project_id: Braintrust project id (optional).
        project_name: Braintrust project name (optional, ``BRAINTRUST_PROJECT`` env var works too).

    Returns:
        ``True`` on successful (or already-registered) setup, ``False`` when
        CrewAI is not importable at the required minimum version.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return CrewAIIntegration.setup()


def patch_crewai() -> bool:
    """Register the Braintrust CrewAI listener without initializing a logger.

    Equivalent to ``CrewAIIntegration.setup()``. Use this when the calling
    code already sets up Braintrust (e.g. via :func:`braintrust.init_logger`)
    and only needs CrewAI tracing wired up.

    Returns:
        ``True`` if CrewAI was patched (or already patched), ``False`` when
        CrewAI is not installed at the required minimum version.
    """
    return CrewAIIntegration.setup()
