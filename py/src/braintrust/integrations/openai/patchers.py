"""OpenAI patchers and public helpers."""

import inspect
from typing import Any

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher
from wrapt import BoundFunctionWrapper, FunctionWrapper

from .tracing import (
    _chat_completion_create_wrapper,
    _chat_completion_parse_wrapper,
    _embedding_create_wrapper,
    _moderation_create_wrapper,
    _responses_create_wrapper,
    _responses_parse_wrapper,
    _responses_raw_create_wrapper,
    _responses_raw_parse_wrapper,
)


# ---------------------------------------------------------------------------
# Factory — single source of truth for each traced method
# ---------------------------------------------------------------------------


def _make_method_patchers(
    *,
    name_prefix: str,
    target_module: str,
    sync_class: str,
    async_class: str,
    method: str,
    wrapper: Any,
    wrap_name: str,
) -> tuple[type[FunctionWrapperPatcher], type[FunctionWrapperPatcher], type[FunctionWrapperPatcher]]:
    """Create sync, async, and instance-level patchers for one method.

    Returns ``(sync_patcher, async_patcher, instance_patcher)``:

    * *sync_patcher* / *async_patcher* — module-level patchers used by
      ``OpenAIIntegration.setup()`` to instrument class methods in-place.
    * *instance_patcher* — used by ``wrap_openai()`` to instrument a specific
      client instance's resource method.

    All three share the same *wrapper* callback, ensuring identical tracing
    regardless of which code path activates the instrumentation.
    """
    sync_patcher: type[FunctionWrapperPatcher] = type(
        f"{name_prefix}.sync",
        (FunctionWrapperPatcher,),
        {
            "__module__": __name__,
            "name": f"{name_prefix}.sync",
            "target_module": target_module,
            "target_path": f"{sync_class}.{method}",
            "wrapper": wrapper,
        },
    )
    async_patcher: type[FunctionWrapperPatcher] = type(
        f"{name_prefix}.async",
        (FunctionWrapperPatcher,),
        {
            "__module__": __name__,
            "name": f"{name_prefix}.async",
            "target_module": target_module,
            "target_path": f"{async_class}.{method}",
            "wrapper": wrapper,
        },
    )
    instance_patcher: type[FunctionWrapperPatcher] = type(
        wrap_name,
        (FunctionWrapperPatcher,),
        {
            "__module__": __name__,
            "name": wrap_name,
            "target_path": method,
            "wrapper": wrapper,
        },
    )
    return sync_patcher, async_patcher, instance_patcher


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

_cc_create_sync, _cc_create_async, _wrap_chat_create = _make_method_patchers(
    name_prefix="openai.chat.completions.create",
    target_module="openai.resources.chat.completions",
    sync_class="Completions",
    async_class="AsyncCompletions",
    method="create",
    wrapper=_chat_completion_create_wrapper,
    wrap_name="openai.wrap.chat.create",
)

_cc_parse_sync, _cc_parse_async, _wrap_chat_parse = _make_method_patchers(
    name_prefix="openai.chat.completions.parse",
    target_module="openai.resources.chat.completions",
    sync_class="Completions",
    async_class="AsyncCompletions",
    method="parse",
    wrapper=_chat_completion_parse_wrapper,
    wrap_name="openai.wrap.chat.parse",
)


class ChatCompletionsPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``openai.resources.chat.completions`` for tracing."""

    name = "openai.chat.completions"
    sub_patchers = (
        _cc_create_sync,
        _cc_create_async,
        _cc_parse_sync,
        _cc_parse_async,
    )


class _WrapChatCompletions(CompositeFunctionWrapperPatcher):
    name = "openai.wrap.chat"
    sub_patchers = (_wrap_chat_create, _wrap_chat_parse)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_emb_create_sync, _emb_create_async, _wrap_emb_create = _make_method_patchers(
    name_prefix="openai.embeddings.create",
    target_module="openai.resources.embeddings",
    sync_class="Embeddings",
    async_class="AsyncEmbeddings",
    method="create",
    wrapper=_embedding_create_wrapper,
    wrap_name="openai.wrap.embeddings.create",
)


class EmbeddingsPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``openai.resources.embeddings`` for tracing."""

    name = "openai.embeddings"
    sub_patchers = (
        _emb_create_sync,
        _emb_create_async,
    )


class _WrapEmbeddings(CompositeFunctionWrapperPatcher):
    name = "openai.wrap.embeddings"
    sub_patchers = (_wrap_emb_create,)


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------

_mod_create_sync, _mod_create_async, _wrap_mod_create = _make_method_patchers(
    name_prefix="openai.moderations.create",
    target_module="openai.resources.moderations",
    sync_class="Moderations",
    async_class="AsyncModerations",
    method="create",
    wrapper=_moderation_create_wrapper,
    wrap_name="openai.wrap.moderations.create",
)


class ModerationsPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``openai.resources.moderations`` for tracing."""

    name = "openai.moderations"
    sub_patchers = (
        _mod_create_sync,
        _mod_create_async,
    )


class _WrapModerations(CompositeFunctionWrapperPatcher):
    name = "openai.wrap.moderations"
    sub_patchers = (_wrap_mod_create,)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

_resp_create_sync, _resp_create_async, _wrap_resp_create = _make_method_patchers(
    name_prefix="openai.responses.create",
    target_module="openai.resources.responses.responses",
    sync_class="Responses",
    async_class="AsyncResponses",
    method="create",
    wrapper=_responses_create_wrapper,
    wrap_name="openai.wrap.responses.create",
)

_resp_parse_sync, _resp_parse_async, _wrap_resp_parse = _make_method_patchers(
    name_prefix="openai.responses.parse",
    target_module="openai.resources.responses.responses",
    sync_class="Responses",
    async_class="AsyncResponses",
    method="parse",
    wrapper=_responses_parse_wrapper,
    wrap_name="openai.wrap.responses.parse",
)

_resp_raw_create_sync, _resp_raw_create_async, _wrap_resp_raw_create = _make_method_patchers(
    name_prefix="openai.responses.raw.create",
    target_module="openai.resources.responses.responses",
    sync_class="ResponsesWithRawResponse",
    async_class="AsyncResponsesWithRawResponse",
    method="create",
    wrapper=_responses_raw_create_wrapper,
    wrap_name="openai.wrap.responses.raw.create",
)

_resp_raw_parse_sync, _resp_raw_parse_async, _wrap_resp_raw_parse = _make_method_patchers(
    name_prefix="openai.responses.raw.parse",
    target_module="openai.resources.responses.responses",
    sync_class="ResponsesWithRawResponse",
    async_class="AsyncResponsesWithRawResponse",
    method="parse",
    wrapper=_responses_raw_parse_wrapper,
    wrap_name="openai.wrap.responses.raw.parse",
)


class ResponsesPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``openai.resources.responses`` for tracing."""

    name = "openai.responses"
    sub_patchers = (
        _resp_create_sync,
        _resp_create_async,
        _resp_parse_sync,
        _resp_parse_async,
        _resp_raw_create_sync,
        _resp_raw_create_async,
        _resp_raw_parse_sync,
        _resp_raw_parse_async,
    )


class _WrapResponses(CompositeFunctionWrapperPatcher):
    name = "openai.wrap.responses"
    sub_patchers = (_wrap_resp_create, _wrap_resp_parse)


class _WrapResponsesRaw(CompositeFunctionWrapperPatcher):
    name = "openai.wrap.responses.raw"
    sub_patchers = (_wrap_resp_raw_create, _wrap_resp_raw_parse)


# ---------------------------------------------------------------------------
# Resource mapping — single source of truth for wrap_openai.
#
# Each entry is (client_attr_path, instance_patcher).  ``wrap_openai``
# iterates this list to wrap a specific client instance.
#
# When adding a new resource, add it here AND add the corresponding
# module-level patchers to ``OpenAIIntegration.patchers`` in
# ``integration.py``.  The ``test_wrap_openai_and_setup_use_same_wrappers``
# test enforces that both paths cover the same wrapper functions.
# ---------------------------------------------------------------------------

_WRAP_TARGETS: tuple[tuple[str, type[CompositeFunctionWrapperPatcher]], ...] = (
    ("chat.completions", _WrapChatCompletions),
    ("embeddings", _WrapEmbeddings),
    ("moderations", _WrapModerations),
    ("responses", _WrapResponses),
    ("responses.with_raw_response", _WrapResponsesRaw),
    ("beta.chat.completions", _WrapChatCompletions),
)


# ---------------------------------------------------------------------------
# Public wrap_openai helper
# ---------------------------------------------------------------------------


def _is_class_method_wrapped(resource: Any, method_name: str) -> bool:
    """Return ``True`` if *method_name* on the **class** of *resource* is
    already a wrapt ``FunctionWrapper`` (i.e. patched by ``setup()``).

    This prevents double-tracing when both ``setup()`` and ``wrap_openai()``
    are active for the same client.
    """
    cls_attr = inspect.getattr_static(type(resource), method_name, None)
    return isinstance(cls_attr, FunctionWrapper)


def _delegates_to_wrapped_method(resource: Any, method_name: str) -> bool:
    """Return ``True`` when *resource.method_name* already delegates to an
    instrumented wrapt wrapper.

    OpenAI's ``with_raw_response`` helpers are regular functions created at
    object construction time. When the parent resource has already been
    wrapped, those helpers forward into the wrapped method via ``__wrapped__``.
    Patching them again would create duplicate spans.
    """
    method = getattr(resource, method_name, None)
    wrapped = getattr(method, "__wrapped__", None)
    return isinstance(wrapped, (FunctionWrapper, BoundFunctionWrapper))


def _wrap_resource(
    client: Any,
    attr_path: str,
    patcher: type[CompositeFunctionWrapperPatcher],
) -> None:
    """Walk *attr_path* from *client* and apply ``patcher.wrap_target()``.

    Skips wrapping when the class methods are already patched at the module
    level (by ``OpenAIIntegration.setup()``), avoiding double spans.
    The instance-level patchers handle their own idempotency via
    ``has_patch_marker``.
    """
    resource = client
    for part in attr_path.split("."):
        resource = getattr(resource, part, None)
        if resource is None:
            return
    # If any sub-patcher's target is already wrapped on the class, the module-
    # level patchers are active and we can skip instance-level wrapping.
    for sub in patcher.sub_patchers:
        attr = sub.target_path.rsplit(".", 1)[-1]
        if _is_class_method_wrapped(resource, attr):
            return
        if attr_path.endswith("with_raw_response") and _delegates_to_wrapped_method(resource, attr):
            return
    patcher.wrap_target(resource)


def wrap_openai(client: Any) -> Any:
    """Manually wrap an OpenAI client instance for tracing.

    Patches resource methods on *client* so that API calls produce Braintrust
    tracing spans.  Only the given instance is affected; other clients and
    the module-level classes are left untouched.

    Idempotent — each instance-level patcher sets its own marker via
    ``has_patch_marker`` so repeated calls are no-ops.

    Returns *client* for convenient chaining::

        client = wrap_openai(openai.OpenAI())
    """
    for attr_path, patcher in _WRAP_TARGETS:
        _wrap_resource(client, attr_path, patcher)
    return client
