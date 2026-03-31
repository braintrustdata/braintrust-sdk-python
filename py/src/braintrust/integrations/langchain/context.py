from contextvars import ContextVar

from braintrust.integrations.langchain.callbacks import BraintrustCallbackHandler
from langchain_core.tracers.context import register_configure_hook


__all__ = ["set_global_handler", "clear_global_handler", "get_global_handler"]


braintrust_callback_handler_var: ContextVar[BraintrustCallbackHandler | None] = ContextVar(
    "braintrust_callback_handler", default=None
)


def set_global_handler(handler: BraintrustCallbackHandler):
    braintrust_callback_handler_var.set(handler)


def get_global_handler() -> BraintrustCallbackHandler | None:
    return braintrust_callback_handler_var.get()


def clear_global_handler():
    braintrust_callback_handler_var.set(None)


register_configure_hook(
    context_var=braintrust_callback_handler_var,
    inheritable=True,
)
