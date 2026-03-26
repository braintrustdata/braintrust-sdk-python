import logging

from braintrust.integrations.langchain.callbacks import BraintrustCallbackHandler
from braintrust.integrations.langchain.context import clear_global_handler, set_global_handler


__all__ = ["BraintrustCallbackHandler", "set_global_handler", "clear_global_handler", "BraintrustTracer"]

_logger = logging.getLogger(__name__)


class BraintrustTracer(BraintrustCallbackHandler):
    """Deprecated: use BraintrustCallbackHandler instead."""

    def __init__(self, *args, **kwargs):
        _logger.warning(
            "BraintrustTracer is deprecated, use BraintrustCallbackHandler instead. "
            "Update your imports to: from braintrust.integrations.langchain import BraintrustCallbackHandler"
        )
        super().__init__(*args, **kwargs)
