"""Pydantic AI integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .harness_patchers import StepPersistencePatcher
from .patchers import (
    AgentPatcher,
    DeferredToolRequestsPatcher,
    DirectModelRequestPatcher,
    DirectModelRequestStreamPatcher,
    DirectModelRequestStreamSyncPatcher,
    DirectModelRequestSyncPatcher,
    DirectPrepareModelPatcher,
    RetryModelPatcher,
    StreamedResponseSyncStartProducerPatcher,
    SyncStreamBridgeInitPatcher,
    ToolCallClassificationPatcher,
    ToolManagerFunctionToolPatcher,
    ValidateStreamOutputPatcher,
)


class PydanticAIIntegration(BaseIntegration):
    """Braintrust instrumentation for Pydantic AI."""

    name = "pydantic_ai"
    import_names = ("pydantic_ai",)
    min_version = "1.10.0"
    patchers = (
        SyncStreamBridgeInitPatcher,
        StreamedResponseSyncStartProducerPatcher,
        AgentPatcher,
        RetryModelPatcher,
        DirectPrepareModelPatcher,
        DirectModelRequestPatcher,
        DirectModelRequestSyncPatcher,
        DirectModelRequestStreamPatcher,
        DirectModelRequestStreamSyncPatcher,
        ToolManagerFunctionToolPatcher,
        StepPersistencePatcher,
        DeferredToolRequestsPatcher,
        ToolCallClassificationPatcher,
        ValidateStreamOutputPatcher,
    )
