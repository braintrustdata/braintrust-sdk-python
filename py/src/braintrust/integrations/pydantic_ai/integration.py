"""Pydantic AI integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentPatcher,
    DirectModelRequestPatcher,
    DirectModelRequestStreamPatcher,
    DirectModelRequestStreamSyncPatcher,
    DirectModelRequestSyncPatcher,
    DirectPrepareModelPatcher,
    StreamedResponseSyncStartProducerPatcher,
    ToolManagerFunctionToolPatcher,
)


class PydanticAIIntegration(BaseIntegration):
    """Braintrust instrumentation for Pydantic AI."""

    name = "pydantic_ai"
    import_names = ("pydantic_ai",)
    min_version = "1.10.0"
    patchers = (
        StreamedResponseSyncStartProducerPatcher,
        AgentPatcher,
        DirectPrepareModelPatcher,
        DirectModelRequestPatcher,
        DirectModelRequestSyncPatcher,
        DirectModelRequestStreamPatcher,
        DirectModelRequestStreamSyncPatcher,
        ToolManagerFunctionToolPatcher,
    )
