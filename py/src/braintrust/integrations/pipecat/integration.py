"""Pipecat integration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    EOUWaitPatcher,
    LLMServicePushFramePatcher,
    OpenAILivePatcher,
    PipelineWorkerPatcher,
    SegmentedSTTPushSegmentFramesPatcher,
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
        LLMServicePushFramePatcher,
        STTServiceProcessGeneratorPatcher,
        SegmentedSTTPushSegmentFramesPatcher,
        OpenAILivePatcher,
    )
