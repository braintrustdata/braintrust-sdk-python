"""Explicit execution ownership for application-defined Pipecat services."""

from typing import Any, Literal


ModelOperationKind = Literal["llm_response", "stt", "tts"]
_BINDINGS = "_braintrust_model_clients"

# These namespaces identify actual clients covered by Braintrust integrations,
# independently of whether those integrations are currently enabled.
_SDK_OPERATIONS = {
    "openai": {"llm_response", "stt", "tts"},
    "anthropic": {"llm_response"},
    "google.genai": {"llm_response", "tts"},
    "mistralai": {"llm_response", "stt", "tts"},
    "cohere": {"llm_response", "stt"},
    "huggingface_hub": {"llm_response"},
    # Bedrock sessions: the Bedrock integration traces the clients they create.
    "aiobotocore": {"llm_response"},
    "aioboto3": {"llm_response"},
    "botocore": {"llm_response"},
    "boto3": {"llm_response"},
}


def set_model_client(service: Any, *, operation: ModelOperationKind, client: Any) -> None:
    """Declare the client that actually executes a custom service's operation.

    Call before running the service. Pass the actual provider SDK client, or
    ``None`` when the service executes its own HTTP/WebSocket request. An unused
    client or a client used only for configuration must not be bound here.

    Built-in compatible services are detected automatically. A custom service
    that overrides their execution path should bind its client explicitly;
    Braintrust does not inspect application bytecode or guess dynamic dispatch.
    A covered SDK retains ownership even when its integration is disabled.
    Update the binding before an operation if the service switches clients.
    """
    if operation not in {"llm_response", "stt", "tts"}:
        raise ValueError(f"Unsupported Pipecat operation: {operation}")
    bindings = dict(getattr(service, _BINDINGS, {}))
    bindings[operation] = client
    setattr(service, _BINDINGS, bindings)


def sdk_client_covers(client: Any, operation: str) -> bool:
    return any(
        operation in operations and (base.__module__ == namespace or base.__module__.startswith(namespace + "."))
        for base in type(client).__mro__
        for namespace, operations in _SDK_OPERATIONS.items()
    )


def provider_client_expected(service: Any, operation: str) -> bool:
    bindings = getattr(service, _BINDINGS, {})
    if operation in bindings:
        return sdk_client_covers(bindings[operation], operation)
    from ._compat import executing_client

    known, client = executing_client(service, operation)
    if known:
        return sdk_client_covers(client, operation)
    # Guessing could duplicate billable usage from a disabled provider
    # integration. Require an explicit binding for unknown execution paths.
    if not getattr(service, "_braintrust_ownership_warning", False):
        from .tracing._common import LOG

        LOG.warning(
            "Unknown Pipecat model execution path on %s; use "
            "braintrust.integrations.pipecat.set_model_client() to enable ownership tracing",
            type(service).__name__,
        )
        service._braintrust_ownership_warning = True
    return True
