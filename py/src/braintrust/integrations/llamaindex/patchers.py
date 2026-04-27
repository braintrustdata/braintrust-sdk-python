"""LlamaIndex patchers.

Registers Braintrust span and event handlers on LlamaIndex's root dispatcher
for automatic instrumentation of all LlamaIndex operations.
"""

from braintrust.integrations.base import CallbackPatcher


try:
    from .event_handler import BraintrustEventHandler
    from .span_handler import BraintrustSpanHandler
except ImportError:
    BraintrustSpanHandler = None
    BraintrustEventHandler = None


def _has_braintrust_handlers() -> bool:
    if BraintrustSpanHandler is None:
        return False
    try:
        from llama_index.core.instrumentation import get_dispatcher

        dispatcher = get_dispatcher()
        return any(isinstance(h, BraintrustSpanHandler) for h in dispatcher.span_handlers)
    except Exception:
        return False


def _register_braintrust_handlers() -> None:
    if BraintrustSpanHandler is None or BraintrustEventHandler is None:
        raise ImportError("llama-index-core is not installed")

    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher()

    if any(isinstance(h, BraintrustSpanHandler) for h in dispatcher.span_handlers):
        return

    dispatcher.add_span_handler(BraintrustSpanHandler())
    dispatcher.add_event_handler(BraintrustEventHandler())


class DispatcherHandlerPatcher(CallbackPatcher):
    """Register Braintrust handlers on LlamaIndex's root dispatcher."""

    name = "llamaindex.dispatcher_handler"
    target_module = "llama_index.core.instrumentation"
    callback = _register_braintrust_handlers
    state_getter = _has_braintrust_handlers
