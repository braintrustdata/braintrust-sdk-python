"""LlamaIndex integration definition."""

from braintrust.integrations.base import BaseIntegration

from .patchers import DispatcherHandlerPatcher


class LlamaIndexIntegration(BaseIntegration):
    """Braintrust instrumentation for LlamaIndex via dispatcher handlers.

    Registers a BraintrustSpanHandler and BraintrustEventHandler on
    LlamaIndex's root dispatcher to automatically trace all instrumented
    operations: LLM calls, query engines, retrievers, synthesizers,
    agents, embeddings, and more.
    """

    name = "llamaindex"
    import_names = ("llama_index.core",)
    min_version = "0.13.0"
    patchers = (DispatcherHandlerPatcher,)
