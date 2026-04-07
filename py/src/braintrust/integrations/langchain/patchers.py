"""LangChain patchers."""

from braintrust.integrations.base import CallbackPatcher


try:
    from .callbacks import BraintrustCallbackHandler
    from .context import get_global_handler, set_global_handler
except ImportError:  # pragma: no cover - optional dependency not installed
    BraintrustCallbackHandler = None
    get_global_handler = None
    set_global_handler = None


def _has_global_handler() -> bool:
    if BraintrustCallbackHandler is None or get_global_handler is None:
        return False
    return isinstance(get_global_handler(), BraintrustCallbackHandler)


def _set_default_global_handler() -> None:
    if BraintrustCallbackHandler is None or get_global_handler is None or set_global_handler is None:
        raise ImportError("langchain_core is not installed")

    if isinstance(get_global_handler(), BraintrustCallbackHandler):
        return

    set_global_handler(BraintrustCallbackHandler())


class GlobalHandlerPatcher(CallbackPatcher):
    """Install the default Braintrust callback handler into LangChain's configure hook."""

    name = "langchain.global_handler"
    target_module = "langchain_core.tracers.context"
    callback = _set_default_global_handler
    state_getter = _has_global_handler
