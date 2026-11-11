"""Framework privacy policy inherited by automatic provider telemetry."""

from collections.abc import Callable
from contextvars import ContextVar

from .span_customizer import SpanCustomizer, SpanExportData


class _ProviderPrivacyCustomizer(SpanCustomizer):
    """Recheck the owning framework's policy when a provider record resolves."""

    def __init__(self, policy: Callable[[], bool]) -> None:
        self.policy = policy

    def on_span_export(self, data: SpanExportData) -> SpanExportData:
        if not self.policy():
            for field in ("input", "output", "error"):
                data.pop(field, None)
        return data


provider_content_allowed: ContextVar[Callable[[], bool] | None] = ContextVar(
    "braintrust_provider_content_allowed", default=None
)

# Explicit provider allowlist. User/tool/framework spans remain visible.
AUTOMATIC_PROVIDERS = frozenset(
    {
        "openai-auto",
        "anthropic-auto",
        "google-genai-auto",
        "litellm-auto",
        "cohere-auto",
        "mistral-auto",
        "bedrock-runtime-auto",
        "openrouter-auto",
        "huggingface-hub-auto",
        "ai-sdk-auto",
    }
)
