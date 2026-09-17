"""TypeSafe integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import TypeSafePatcher


class TypeSafeIntegration(BaseIntegration):
    """Braintrust instrumentation for the TypeSafe Python SDK."""

    name = "typesafe"
    import_names = ("typesafe_sdk",)
    distribution_names = ("typesafe-sdk",)
    min_version = "0.6.0"
    patchers = (TypeSafePatcher,)
