"""Pipecat integration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    ContextAddMessagePatcher,
    EOUWaitPatcher,
    LLMServicePushFramePatcher,
    OpenAILivePatcher,
    OpenAIRealtimePatcher,
    PipelineWorkerPatcher,
    STTServiceProcessGeneratorPatcher,
)


class PipecatIntegration(BaseIntegration):
    """Braintrust instrumentation for Pipecat AI pipelines."""

    name = "pipecat"
    import_names = ("pipecat",)
    distribution_names = ("pipecat-ai",)
    min_version = "1.3.0"
    patchers = (
        EOUWaitPatcher,
        PipelineWorkerPatcher,
        ContextAddMessagePatcher,
        LLMServicePushFramePatcher,
        STTServiceProcessGeneratorPatcher,
        OpenAILivePatcher,
        OpenAIRealtimePatcher,
    )
