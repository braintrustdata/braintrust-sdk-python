"""Pipecat integration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import PipelineWorkerPatcher


class PipecatIntegration(BaseIntegration):
    """Braintrust instrumentation for Pipecat AI pipelines."""

    name = "pipecat"
    import_names = ("pipecat",)
    distribution_names = ("pipecat-ai",)
    min_version = "1.3.0"
    patchers = (PipelineWorkerPatcher,)
