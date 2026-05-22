"""Execute BTX LLM-span specs in-process using the Braintrust Python SDK.

Returns the root span ID so the test can fetch the spans back from Braintrust
via BTQL (VCR-off mode) or drain the memory logger (VCR-on mode).

Client selection
----------------
The BTX_CLIENT environment variable (set by the nox session) controls which
SDK client is used to make provider API calls.  When absent, the client
defaults to the same name as the provider (e.g. provider=openai → client=openai).

Current client identifiers:
  openai    — braintrust.wrap_openai(openai.OpenAI())
  anthropic — braintrust.wrap_anthropic(anthropic.Anthropic())

Adding a new client (e.g. "openrouter" targeting the openai provider):
  1. Add a branch in _build_client() that constructs the wrapped client.
  2. Add a nox session test_btx_openrouter_openai parametrized over the
     openrouter version matrix, passing BTX_PROVIDERS=openai BTX_CLIENT=openrouter.
  3. The rest of the framework (spec loading, cassette routing, validation)
     requires no changes.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import braintrust
from braintrust import logger, wrap_openai

from .spec_loader import LlmSpanSpec


def execute_spec(spec: LlmSpanSpec, *, live: bool = False) -> str:
    """Execute all requests in *spec* under a Braintrust parent span.

    Args:
        spec: The spec to execute.
        live: When True (VCR-off mode), calls ``braintrust.init_logger()`` so
              spans are sent to the real backend.  When False (VCR mode), uses
              the module-level logger already initialised by ``init_test_logger``
              in the test fixture — no network calls to Braintrust are made.

    Returns:
        The root span ID, used by the test to fetch spans from Braintrust
        (live mode) or correlate in-memory spans (VCR mode).
    """
    # BTX_CLIENT defaults to BTX_PROVIDER. Both are set by the nox session,
    # so "test_btx_openai" (which sets BTX_PROVIDER=openai) implies client=openai
    # without needing BTX_CLIENT to be set explicitly.
    client_id = os.environ.get("BTX_CLIENT") or os.environ.get("BTX_PROVIDER")
    if not client_id:
        raise RuntimeError(
            "BTX_PROVIDER environment variable is not set. Run btx tests via nox: nox -s 'test_btx_openai(latest)'"
        )
    client = _build_client(client_id)

    if live:
        project = os.environ.get("BRAINTRUST_PROJECT") or os.environ.get(
            "BRAINTRUST_DEFAULT_PROJECT_NAME", "python-unit-test"
        )
        project_id = os.environ.get("BRAINTRUST_PROJECT_ID") or os.environ.get("BRAINTRUST_DEFAULT_PROJECT_ID")
        bt_logger = braintrust.init_logger(project=project, project_id=project_id or None)
        root_span_id: str = ""
        with bt_logger.start_span(name=spec.name) as root_span:
            root_span_id = root_span.root_span_id
            _dispatch(spec, client)
        bt_logger.flush()
        return root_span_id
    else:
        # VCR mode: use the global logger already set up by init_test_logger().
        root_span_id = ""
        with logger.start_span(name=spec.name) as root_span:
            root_span_id = root_span.root_span_id
            _dispatch(spec, client)
        return root_span_id


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _build_client(client_id: str) -> Any:
    """Construct and return a Braintrust-wrapped provider client.

    Args:
        client_id: The BTX client identifier (e.g. "openai", "anthropic").
                   Matches the nox session name component after "test_btx_".
    """
    if client_id == "openai":
        import openai as _openai

        return wrap_openai(_openai.OpenAI())

    if client_id == "anthropic":
        import anthropic as _anthropic
        from braintrust import wrap_anthropic

        return wrap_anthropic(_anthropic.Anthropic())

    raise NotImplementedError(
        f"BTX executor: unknown client {client_id!r}. Add a branch in _build_client() to support this client."
    )


# ---------------------------------------------------------------------------
# Dispatch by provider + endpoint
# ---------------------------------------------------------------------------


def _dispatch(spec: LlmSpanSpec, client: Any) -> None:
    provider = spec.provider
    endpoint = spec.endpoint

    if provider == "openai" and endpoint == "/v1/chat/completions":
        _execute_chat_completions(spec.requests, client)

    elif provider == "openai" and endpoint == "/v1/responses":
        _execute_responses(spec.requests, client)

    elif provider == "anthropic" and endpoint == "/v1/messages":
        _execute_anthropic_messages(spec.requests, client)

    else:
        raise NotImplementedError(f"BTX executor: provider={provider!r} endpoint={endpoint!r} not implemented")


# ---------------------------------------------------------------------------
# OpenAI /v1/chat/completions
# ---------------------------------------------------------------------------


def _execute_chat_completions(requests: list[dict[str, Any]], client: Any) -> None:
    conversation_history: list[dict[str, Any]] = []

    for req in requests:
        full_req = copy.deepcopy(req)
        if conversation_history:
            full_req["messages"] = conversation_history + full_req.get("messages", [])

        is_streaming = full_req.pop("stream", False)
        conversation_history.extend(req.get("messages", []))

        if is_streaming:
            # Use the context-manager streaming form (openai >= 1.x) to
            # retrieve the accumulated final completion for conversation history.
            # Fall back to create(stream=True) for older SDK versions.
            if hasattr(client.chat.completions, "stream"):
                with client.chat.completions.stream(**full_req) as stream:
                    final = stream.get_final_completion()
                if final.choices:
                    msg = final.choices[0].message
                    conversation_history.append({"role": "assistant", "content": msg.content or ""})
            else:
                response = client.chat.completions.create(stream=True, **full_req)
                for _ in response:
                    pass
                # Can't retrieve final message from raw iterator; history omitted.
        else:
            response = client.chat.completions.create(**full_req)
            if response.choices:
                msg = response.choices[0].message
                conversation_history.append({"role": "assistant", "content": msg.content or ""})


# ---------------------------------------------------------------------------
# OpenAI /v1/responses  (Responses API — reasoning spec)
# ---------------------------------------------------------------------------


def _execute_responses(requests: list[dict[str, Any]], client: Any) -> None:
    conversation_history: list[Any] = []

    for req in requests:
        full_req = copy.deepcopy(req)
        if conversation_history:
            full_req["input"] = conversation_history + list(full_req.get("input", []))

        response = client.responses.create(**full_req)

        conversation_history.extend(req.get("input", []))
        if hasattr(response, "output"):
            conversation_history.extend(response.output)


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------


def _execute_anthropic_messages(requests: list[dict[str, Any]], client: Any) -> None:
    """Execute Anthropic messages requests.

    Handles streaming (stream=True) by consuming the stream context manager.
    Multi-turn is not in the current spec but conversation history is tracked
    for future use.
    """
    conversation_history: list[dict[str, Any]] = []

    for req in requests:
        full_req = copy.deepcopy(req)

        if conversation_history:
            full_req["messages"] = conversation_history + full_req.get("messages", [])

        is_streaming = full_req.get("stream", False)
        conversation_history.extend(req.get("messages", []))

        if is_streaming:
            with client.messages.create(**full_req) as stream:
                final = stream.get_final_message()
            if hasattr(final, "content") and final.content:
                text_blocks = [b.text for b in final.content if hasattr(b, "text")]
                conversation_history.append({"role": "assistant", "content": " ".join(text_blocks)})
        else:
            response = client.messages.create(**full_req)
            if hasattr(response, "content") and response.content:
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                conversation_history.append({"role": "assistant", "content": " ".join(text_blocks)})
