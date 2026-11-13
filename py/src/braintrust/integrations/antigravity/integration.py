"""Antigravity integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import AgentPatcher, ChatPatcher, ConversationLifecyclePatcher, ReceiveStepsPatcher, ToolExecutePatcher


class AntigravityIntegration(BaseIntegration):
    """Trace Google Antigravity's direct, streaming chat API."""

    name = "antigravity"
    import_names = ("google.antigravity",)
    distribution_names = ("google-antigravity",)
    min_version = "0.1.23"
    patchers = (AgentPatcher, ChatPatcher, ReceiveStepsPatcher, ToolExecutePatcher, ConversationLifecyclePatcher)
