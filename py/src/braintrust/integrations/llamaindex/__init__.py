"""Braintrust integration for LlamaIndex."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import LlamaIndexIntegration


try:
    from .span_handler import BraintrustSpanHandler
except ImportError as exc:
    _IMPORT_ERROR = exc

    class BraintrustSpanHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("llama-index-core is required for braintrust.integrations.llamaindex") from _IMPORT_ERROR


__all__ = [
    "BraintrustSpanHandler",
    "LlamaIndexIntegration",
    "setup_llamaindex",
]


def setup_llamaindex(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return LlamaIndexIntegration.setup()
