"""DSPy integration — orchestration class and setup entry-point."""

from braintrust.integrations.base import BaseIntegration

from .patchers import DSPyConfigurePatcher


class DSPyIntegration(BaseIntegration):
    """Braintrust instrumentation for DSPy."""

    name = "dspy"
    import_names = ("dspy",)
    patchers = (DSPyConfigurePatcher,)
