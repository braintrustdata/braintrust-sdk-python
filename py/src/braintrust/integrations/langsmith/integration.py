"""LangSmith integration definition."""

from typing import Any

from braintrust.integrations.base import BaseIntegration

from .patchers import ClientEvaluationPatcher, ModuleEvaluationPatcher, TraceablePatcher
from .tracing import _set_langsmith_standalone_override


class LangSmithIntegration(BaseIntegration):
    """Braintrust instrumentation for LangSmith migration helpers."""

    name = "langsmith"
    import_names = ("langsmith",)
    patchers = (
        TraceablePatcher,
        ClientEvaluationPatcher,
        ModuleEvaluationPatcher,
    )

    @classmethod
    def setup(
        cls,
        *,
        target: Any | None = None,
        standalone: bool | None = None,
    ) -> bool:
        _set_langsmith_standalone_override(standalone)
        return super().setup(target=target)
