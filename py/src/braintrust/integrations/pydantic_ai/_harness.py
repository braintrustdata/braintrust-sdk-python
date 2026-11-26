"""Optional Pydantic AI Harness capability instrumentation."""

import sys
from contextvars import ContextVar

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher
from braintrust.logger import Span
from braintrust.span_types import SpanTypeAttribute

from .tracing import start_span


async def _compact_messages(wrapped, _instance, args, kwargs):
    background = kwargs.get("background")
    if background == "used":
        return await wrapped(*args, **kwargs)
    messages = kwargs["messages"]
    before = len(messages)
    compacted = await wrapped(*args, **kwargs)
    # Harness returns the original list when compaction leaves the history unchanged.
    if compacted is not messages:
        metadata = {"strategy": kwargs["strategy"]}
        if kwargs.get("model_requested"):
            metadata["model_requested"] = True
        if background is not None:
            metadata["background"] = background
        start_span(
            name="harness_message_compaction",
            type=SpanTypeAttribute.TASK,
            input={"messages": before},
            output={"messages": len(compacted)},
            metadata=metadata,
        ).end()
    return compacted


def _record_memory_injection(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    _, main, subfiles, files_truncated, snapshot, appended, withheld = args
    main_chars = len(main.content) if main is not None else 0
    has_memory = bool(main_chars or subfiles or files_truncated)
    start_span(
        name="harness_memory_injection",
        type=SpanTypeAttribute.TASK,
        input={"main_chars": main_chars, "files": len(subfiles)},
        output={
            "appended": appended is not None,
            "injected_chars": len(appended or ""),
            "outcome": "budget_exhausted" if snapshot is None and has_memory else "ok",
        },
        metadata={
            "files_truncated": files_truncated,
            "main_truncated": main is not None and main.truncated,
            **({"main_withheld": withheld} if instance.scanner is not None else {}),
        },
    ).end()
    return result


def _trace_guardrail_block(wrapped, _instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    metadata = {"direction": kwargs["direction"]}
    tool_name = kwargs.get("tool_name")
    if tool_name is not None:
        metadata["tool_name"] = tool_name
    start_span(
        name="harness_guardrail_block",
        type=SpanTypeAttribute.TASK,
        output={"action": "block"},
        metadata=metadata,
    ).end()
    return result


_router_span = ContextVar[Span | None]("pydantic_ai_harness_router_span", default=None)
_ROUTER_METADATA_FIELDS = {
    "model_router.mode": "mode",
    "model_router.run_step": "run_step",
    "model_router.probability": "probability",
    "model_router.error.type": "error.type",
}


async def _select_model(wrapped, instance, args, kwargs):
    # Bootstrap and cached lookups do not make a routing decision.
    if instance._run_ctx is None or (instance.mode == "once" and instance._cached_choice is not None):
        return await wrapped(*args, **kwargs)
    with start_span(
        name="harness_model_selection",
        type=SpanTypeAttribute.TASK,
        input={"choices": list(instance.choices), "default": instance.default},
    ) as span:
        token = _router_span.set(span)
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _router_span.reset(token)


def _record_model_selection(wrapped, _instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    span = _router_span.get()
    if span is not None:
        _, attributes, choice, fallback_reason = args
        span.log(
            output={"choice": choice, "fallback_reason": fallback_reason},
            metadata={field: attributes[key] for key, field in _ROUTER_METADATA_FIELDS.items() if key in attributes},
        )
    return result


class _HarnessPatcher(FunctionWrapperPatcher):
    version_spec = ">=2.75"


class _SharedHarnessFunctionPatcher(_HarnessPatcher):
    @classmethod
    def patch(cls, module, version, *, target=None):
        root = cls.resolve_root(module, version, target=target)
        original = cls.resolve_target(module, version, target=target)
        if not super().patch(module, version, target=target):
            return False
        wrapped = cls.resolve_target(module, version, target=target)
        # Harness imports these helpers by name while initializing their feature packages.
        prefix = root.__package__ + "."
        for name, imported in sys.modules.copy().items():
            if name.startswith(prefix) and getattr(imported, cls.target_path, None) is original:
                setattr(imported, cls.target_path, wrapped)
        return True


class HarnessCompactionPatcher(_SharedHarnessFunctionPatcher):
    name = "pydantic_ai.harness.compaction"
    target_module = "pydantic_ai_harness.compaction._shared"
    target_path = "compact_with_span"
    wrapper = _compact_messages


class HarnessMemoryInjectionPatcher(_HarnessPatcher):
    name = "pydantic_ai.harness.memory_injection"
    target_module = "pydantic_ai_harness.memory._capability"
    target_path = "Memory._record_injection"
    wrapper = _record_memory_injection


class HarnessGuardrailBlockPatcher(_SharedHarnessFunctionPatcher):
    name = "pydantic_ai.harness.guardrail_block"
    target_module = "pydantic_ai_harness.guardrails._shared"
    target_path = "trace_block"
    wrapper = _trace_guardrail_block


class _HarnessModelRouterSelectPatcher(_HarnessPatcher):
    name = "pydantic_ai.harness.model_router.select"
    target_module = "pydantic_ai_harness.model_router._capability"
    target_path = "ModelRouter._select_model"
    wrapper = _select_model


class _HarnessModelRouterRecordPatcher(_HarnessPatcher):
    name = "pydantic_ai.harness.model_router.record"
    target_module = "pydantic_ai_harness.model_router._capability"
    target_path = "_record"
    wrapper = _record_model_selection


class HarnessModelRouterPatcher(CompositeFunctionWrapperPatcher):
    name = "pydantic_ai.harness.model_router"
    sub_patchers = (_HarnessModelRouterSelectPatcher, _HarnessModelRouterRecordPatcher)

    @classmethod
    def applies(cls, module, version, *, target=None):
        return all(patcher.applies(module, version, target=target) for patcher in cls.sub_patchers)
