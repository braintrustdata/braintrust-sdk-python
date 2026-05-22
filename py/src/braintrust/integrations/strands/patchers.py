"""Strands Agents patchers."""

from typing import Any

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _end_agent_span_wrapper,
    _end_event_loop_cycle_span_wrapper,
    _end_model_invoke_span_wrapper,
    _end_tool_call_span_wrapper,
    _start_agent_span_wrapper,
    _start_event_loop_cycle_span_wrapper,
    _start_model_invoke_span_wrapper,
    _start_tool_call_span_wrapper,
)


class _StartAgentSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.start_agent_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.start_agent_span"
    wrapper = _start_agent_span_wrapper


class _EndAgentSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.end_agent_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.end_agent_span"
    wrapper = _end_agent_span_wrapper


class _StartEventLoopCycleSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.start_event_loop_cycle_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.start_event_loop_cycle_span"
    wrapper = _start_event_loop_cycle_span_wrapper


class _EndEventLoopCycleSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.end_event_loop_cycle_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.end_event_loop_cycle_span"
    wrapper = _end_event_loop_cycle_span_wrapper


class _StartModelInvokeSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.start_model_invoke_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.start_model_invoke_span"
    wrapper = _start_model_invoke_span_wrapper


class _EndModelInvokeSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.end_model_invoke_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.end_model_invoke_span"
    wrapper = _end_model_invoke_span_wrapper


class _StartToolCallSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.start_tool_call_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.start_tool_call_span"
    wrapper = _start_tool_call_span_wrapper


class _EndToolCallSpanPatcher(FunctionWrapperPatcher):
    name = "strands.tracer.end_tool_call_span"
    target_module = "strands.telemetry.tracer"
    target_path = "Tracer.end_tool_call_span"
    wrapper = _end_tool_call_span_wrapper


class TracerPatcher(CompositeFunctionWrapperPatcher):
    """Patch Strands' native OTEL tracer lifecycle and mirror it to Braintrust."""

    name = "strands.tracer"
    sub_patchers = (
        _StartAgentSpanPatcher,
        _EndAgentSpanPatcher,
        _StartEventLoopCycleSpanPatcher,
        _EndEventLoopCycleSpanPatcher,
        _StartModelInvokeSpanPatcher,
        _EndModelInvokeSpanPatcher,
        _StartToolCallSpanPatcher,
        _EndToolCallSpanPatcher,
    )


def wrap_strands_tracer(Tracer: Any) -> Any:
    """Manually patch a Strands ``Tracer`` class for Braintrust tracing.

    Most users should call ``setup_strands()`` instead. Use this helper only
    when you need to instrument a specific imported/custom Strands ``Tracer``
    class directly, for example before constructing agents in an environment
    where automatic integration setup is not used.

    Example:
        ```python
        from braintrust.integrations.strands import wrap_strands_tracer
        from strands.telemetry.tracer import Tracer

        wrap_strands_tracer(Tracer)
        ```
    """
    return TracerPatcher.wrap_target(Tracer)
