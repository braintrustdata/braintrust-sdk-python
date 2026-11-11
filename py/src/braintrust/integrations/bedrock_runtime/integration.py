"""boto3/aiobotocore Bedrock Runtime integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import AsyncBedrockClientCreatorPatcher, BedrockClientCreatorPatcher


class BedrockRuntimeIntegration(BaseIntegration):
    """Braintrust instrumentation for boto3/aiobotocore Bedrock Runtime clients."""

    name = "bedrock_runtime"
    import_names = ("botocore",)
    distribution_names = ("botocore",)
    # Botocore 1.34.116 is the first release whose Bedrock Runtime service model
    # exposes both Converse and ConverseStream.
    min_version = "1.34.116"
    patchers = (BedrockClientCreatorPatcher, AsyncBedrockClientCreatorPatcher)
