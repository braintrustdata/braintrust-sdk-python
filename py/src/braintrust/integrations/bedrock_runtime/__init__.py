"""Public entry points for the boto3 Bedrock Runtime integration."""

from .integration import BedrockRuntimeIntegration
from .patchers import wrap_bedrock_client


__all__ = ["BedrockRuntimeIntegration", "setup_bedrock", "wrap_bedrock"]


def setup_bedrock() -> bool:
    """Patch botocore client creation to auto-wrap Bedrock Runtime clients."""
    return BedrockRuntimeIntegration.setup()


def wrap_bedrock(client):
    """Instrument a boto3 Bedrock Runtime client instance in place."""
    return wrap_bedrock_client(client)
