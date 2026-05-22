"""Temporal integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import ClientConnectPatcher, WorkerInitPatcher


class TemporalIntegration(BaseIntegration):
    """Braintrust instrumentation for Temporal workflows and activities."""

    name = "temporal"
    import_names = ("temporalio",)
    min_version = "1.19.0"
    patchers = (ClientConnectPatcher, WorkerInitPatcher)
