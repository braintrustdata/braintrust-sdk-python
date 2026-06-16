"""Patchers for the boto3 Bedrock Runtime integration."""

from typing import Any

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _converse_stream_wrapper,
    _converse_wrapper,
    _invoke_model_stream_wrapper,
    _invoke_model_wrapper,
)


def _is_bedrock_runtime_client(client: Any) -> bool:
    meta = getattr(client, "meta", None)
    service_model = getattr(meta, "service_model", None) or getattr(client, "_service_model", None)
    return getattr(service_model, "service_name", None) == "bedrock-runtime"


def wrap_bedrock_client(client: Any) -> Any:
    """Instrument a boto3/botocore Bedrock Runtime client instance in place."""
    if not _is_bedrock_runtime_client(client):
        return client
    return BedrockRuntimeClientMethodsPatcher.wrap_target(client)


def _create_client_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    client = wrapped(*args, **kwargs)
    return wrap_bedrock_client(client)


class ConversePatcher(FunctionWrapperPatcher):
    name = "bedrock_runtime.converse"
    target_path = "converse"
    wrapper = _converse_wrapper


class ConverseStreamPatcher(FunctionWrapperPatcher):
    name = "bedrock_runtime.converse_stream"
    target_path = "converse_stream"
    wrapper = _converse_stream_wrapper


class InvokeModelPatcher(FunctionWrapperPatcher):
    name = "bedrock_runtime.invoke_model"
    target_path = "invoke_model"
    wrapper = _invoke_model_wrapper


class InvokeModelWithResponseStreamPatcher(FunctionWrapperPatcher):
    name = "bedrock_runtime.invoke_model_with_response_stream"
    target_path = "invoke_model_with_response_stream"
    wrapper = _invoke_model_stream_wrapper


class BedrockRuntimeClientMethodsPatcher(CompositeFunctionWrapperPatcher):
    name = "bedrock_runtime.client_methods"
    sub_patchers = (
        ConversePatcher,
        ConverseStreamPatcher,
        InvokeModelPatcher,
        InvokeModelWithResponseStreamPatcher,
    )


class BedrockClientCreatorPatcher(FunctionWrapperPatcher):
    """Patch botocore's dynamic client factory to wrap Bedrock Runtime clients."""

    name = "bedrock_runtime.client_creator"
    target_module = "botocore.client"
    target_path = "ClientCreator.create_client"
    wrapper = _create_client_wrapper
