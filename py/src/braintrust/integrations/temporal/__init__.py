"""Braintrust integration for Temporal workflows and activities."""

import logging
from typing import TYPE_CHECKING, Any

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import TemporalIntegration


if TYPE_CHECKING:
    from .plugin import BraintrustInterceptor, BraintrustPlugin


logger = logging.getLogger(__name__)

__all__ = [
    "BraintrustInterceptor",
    "BraintrustPlugin",
    "TemporalIntegration",
    "setup_temporal",
]


def setup_temporal(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust auto-instrumentation for Temporal."""
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return TemporalIntegration.setup()


def __getattr__(name: str) -> Any:
    if name in {"BraintrustInterceptor", "BraintrustPlugin"}:
        from .plugin import BraintrustInterceptor, BraintrustPlugin

        return {"BraintrustInterceptor": BraintrustInterceptor, "BraintrustPlugin": BraintrustPlugin}[name]
    raise AttributeError(name)
