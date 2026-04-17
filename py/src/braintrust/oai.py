from braintrust.integrations.openai import OpenAIIntegration, wrap_openai

# NamedWrapper is re-exported here for back-compat. autoevals imports it from
# `braintrust.oai` to detect OpenAI clients that the Braintrust SDK has already
# wrapped (see autoevals/oai.py); if the import fails, autoevals falls back to
# an internal stub and its isinstance check misses real wrapped clients,
# causing scorer LLM calls to be re-wrapped without tracing.
from braintrust.integrations.openai.tracing import NamedWrapper


def patch_openai() -> bool:
    """Patch OpenAI globally for Braintrust tracing."""
    return OpenAIIntegration.setup()


__all__ = [
    "NamedWrapper",
    "patch_openai",
    "wrap_openai",
]
