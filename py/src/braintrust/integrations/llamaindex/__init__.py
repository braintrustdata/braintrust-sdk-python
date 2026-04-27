"""Braintrust integration for LlamaIndex."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import LlamaIndexIntegration


_IMPORT_ERROR: ImportError | None = None
try:
    from .tracing import BraintrustSpanHandler as _BraintrustSpanHandler
except ImportError as exc:
    _IMPORT_ERROR = exc
    _BraintrustSpanHandler = None


if _BraintrustSpanHandler is None:

    class BraintrustSpanHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            message = "llama-index-core is required for braintrust.integrations.llamaindex"
            if _IMPORT_ERROR is not None:
                raise ImportError(message) from _IMPORT_ERROR
            raise ImportError(message)

else:
    BraintrustSpanHandler = _BraintrustSpanHandler


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
