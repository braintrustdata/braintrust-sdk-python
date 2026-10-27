"""Hugging Face Transformers integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import PIPELINE_PATCHERS


class TransformersIntegration(BaseIntegration):
    """Braintrust instrumentation for local Transformers pipelines."""

    name = "transformers"
    import_names = ("transformers",)
    distribution_names = ("transformers",)
    min_version = "4.42.0"
    patchers = PIPELINE_PATCHERS
