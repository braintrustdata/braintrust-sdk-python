from braintrust.integrations.openai import OpenAIIntegration, wrap_openai


def patch_openai() -> bool:
    """Patch OpenAI globally for Braintrust tracing."""
    return OpenAIIntegration.setup()


__all__ = [
    "patch_openai",
    "wrap_openai",
]
