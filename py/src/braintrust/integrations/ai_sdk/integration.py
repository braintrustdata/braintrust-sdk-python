"""Vercel AI SDK for Python integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import TelemetryAdapterPatcher


class AISDKIntegration(BaseIntegration):
    """Braintrust instrumentation for Vercel AI SDK for Python."""

    name = "ai_sdk"
    import_names = ("ai",)
    distribution_names = ("ai",)
    min_version = "0.4.0"
    patchers = (TelemetryAdapterPatcher,)
