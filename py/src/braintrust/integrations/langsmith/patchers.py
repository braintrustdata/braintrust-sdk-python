"""LangSmith patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _aevaluate_wrapper,
    _client_aevaluate_wrapper,
    _client_evaluate_wrapper,
    _evaluate_wrapper,
    _traceable_wrapper,
)


class TraceablePatcher(FunctionWrapperPatcher):
    """Patch ``langsmith.run_helpers.traceable``."""

    name = "langsmith.traceable"
    target_module = "langsmith.run_helpers"
    target_path = "traceable"
    wrapper = _traceable_wrapper


class _ClientEvaluatePatcher(FunctionWrapperPatcher):
    name = "langsmith.client.evaluate"
    target_module = "langsmith.client"
    target_path = "Client.evaluate"
    wrapper = _client_evaluate_wrapper


class _ClientAEvaluatePatcher(FunctionWrapperPatcher):
    name = "langsmith.client.aevaluate"
    target_module = "langsmith.client"
    target_path = "Client.aevaluate"
    wrapper = _client_aevaluate_wrapper


class ClientEvaluationPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``langsmith.Client.evaluate`` and ``langsmith.Client.aevaluate``."""

    name = "langsmith.client"
    sub_patchers = (_ClientEvaluatePatcher, _ClientAEvaluatePatcher)


class _EvaluatePatcher(FunctionWrapperPatcher):
    name = "langsmith.evaluate.sync"
    target_module = "langsmith.evaluation._runner"
    target_path = "evaluate"
    wrapper = _evaluate_wrapper


class _AEvaluatePatcher(FunctionWrapperPatcher):
    name = "langsmith.evaluate.async"
    target_module = "langsmith.evaluation._arunner"
    target_path = "aevaluate"
    wrapper = _aevaluate_wrapper


class ModuleEvaluationPatcher(CompositeFunctionWrapperPatcher):
    """Patch module-level ``langsmith.evaluate`` and ``langsmith.aevaluate``."""

    name = "langsmith.evaluate"
    sub_patchers = (_EvaluatePatcher, _AEvaluatePatcher)
