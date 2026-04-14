"""Braintrust integration for LangChain."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import LangChainIntegration


try:
    from .callbacks import BraintrustCallbackHandler
    from .context import set_global_handler
except ImportError as exc:  # pragma: no cover - optional dependency not installed
    _IMPORT_ERROR = exc

    class BraintrustCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain-core is required for braintrust.integrations.langchain") from _IMPORT_ERROR

    def set_global_handler(handler):  # type: ignore[no-redef]
        raise ImportError("langchain-core is required for braintrust.integrations.langchain") from _IMPORT_ERROR


__all__ = [
    "BraintrustCallbackHandler",
    "LangChainIntegration",
    "set_global_handler",
    "setup_langchain",
]


def setup_langchain(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Setup Braintrust integration with LangChain."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return LangChainIntegration.setup()
