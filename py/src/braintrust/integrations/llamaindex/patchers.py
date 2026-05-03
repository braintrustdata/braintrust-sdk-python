"""LlamaIndex patchers."""

from braintrust.integrations.base import CallbackPatcher


try:
    from .tracing import BraintrustSpanHandler
except ImportError:
    BraintrustSpanHandler = None


def _has_braintrust_handlers() -> bool:
    if BraintrustSpanHandler is None:
        return False

    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher()
    return any(isinstance(h, BraintrustSpanHandler) for h in dispatcher.span_handlers)


def _register_braintrust_handlers() -> None:
    if BraintrustSpanHandler is None:
        raise ImportError("llama-index-core is not installed")

    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher()

    if any(isinstance(h, BraintrustSpanHandler) for h in dispatcher.span_handlers):
        return

    dispatcher.add_span_handler(BraintrustSpanHandler())


class DispatcherHandlerPatcher(CallbackPatcher):
    name = "llamaindex.dispatcher_handler"
    target_module = "llama_index.core.instrumentation"
    callback = _register_braintrust_handlers
    state_getter = _has_braintrust_handlers
