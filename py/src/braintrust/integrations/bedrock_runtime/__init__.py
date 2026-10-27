"""Braintrust integration for boto3 Bedrock Runtime."""

from .integration import BedrockRuntimeIntegration
from .patchers import wrap_bedrock_client


__all__ = ["BedrockRuntimeIntegration", "setup_bedrock", "wrap_bedrock"]


def setup_bedrock() -> bool:
    """Set up Braintrust tracing for Bedrock Runtime.

    Returns True if patched or already patched, False if botocore is missing or too old.
    """
    return BedrockRuntimeIntegration.setup()


def wrap_bedrock(client):
    """Instrument a Bedrock Runtime client in place."""
    return wrap_bedrock_client(client)
