"""boto3 Bedrock Runtime integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import BedrockClientCreatorPatcher


class BedrockRuntimeIntegration(BaseIntegration):
    """Braintrust instrumentation for boto3 Bedrock Runtime clients."""

    name = "bedrock_runtime"
    import_names = ("botocore",)
    distribution_names = ("botocore",)
    # Botocore 1.34.116 is the first release whose Bedrock Runtime service model
    # exposes both Converse and ConverseStream.
    min_version = "1.34.116"
    patchers = (BedrockClientCreatorPatcher,)
