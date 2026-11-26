"""Pydantic AI integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .harness import (
    HarnessCompactionPatcher,
    HarnessGuardrailBlockPatcher,
    HarnessMemoryInjectionPatcher,
    HarnessModelRouterPatcher,
)
from .harness_patchers import StepPersistencePatcher
from .patchers import (
    AgentPatcher,
    BackgroundToolExecutionPatcher,
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
        ToolCallClassificationPatcher,
        BackgroundToolExecutionPatcher,
        HarnessCompactionPatcher,
        HarnessMemoryInjectionPatcher,
        HarnessGuardrailBlockPatcher,
        HarnessModelRouterPatcher,
    )
