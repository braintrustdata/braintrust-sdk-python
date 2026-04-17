from braintrust.integrations.openai import OpenAIIntegration, wrap_openai
from braintrust.integrations.openai.tracing import NamedWrapper


def patch_openai() -> bool:
    """Patch OpenAI globally for Braintrust tracing."""
    return OpenAIIntegration.setup()


__all__ = [
    "NamedWrapper",
    "patch_openai",
    "wrap_openai",
]
