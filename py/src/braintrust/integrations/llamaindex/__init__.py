"""Braintrust integration for LlamaIndex."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import LlamaIndexIntegration


try:
    from .event_handler import BraintrustEventHandler
    from .span_handler import BraintrustSpanHandler
except ImportError as exc:
    _IMPORT_ERROR = exc

    class BraintrustSpanHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("llama-index-core is required for braintrust.integrations.llamaindex") from _IMPORT_ERROR

    class BraintrustEventHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("llama-index-core is required for braintrust.integrations.llamaindex") from _IMPORT_ERROR


__all__ = [
    "BraintrustEventHandler",
    "BraintrustSpanHandler",
    "LlamaIndexIntegration",
    "setup_llamaindex",
]


def setup_llamaindex(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Setup Braintrust integration with LlamaIndex.

    Registers Braintrust span and event handlers on LlamaIndex's root
    dispatcher, enabling automatic tracing of all LlamaIndex operations.

    Args:
        api_key: Braintrust API key. If not provided, uses BRAINTRUST_API_KEY env var.
        project_id: Braintrust project ID.
        project_name: Braintrust project name.

    Returns:
        True if the integration was successfully set up.
    """
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return LlamaIndexIntegration.setup()
