"""Strands Agents integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import TracerPatcher


class StrandsIntegration(BaseIntegration):
    """Braintrust instrumentation for Strands Agents."""

    name = "strands"
    import_names = ("strands",)
    min_version = "1.20.0"
    patchers = (TracerPatcher,)
