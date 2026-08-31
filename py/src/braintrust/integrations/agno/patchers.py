from typing import Any, ClassVar

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .eval_tracing import (
    _accuracy_aevaluate_answer_wrapper,
    _accuracy_arun_wrapper,
    _accuracy_evaluate_answer_wrapper,
    _accuracy_run_wrapper,
    _arun_case_wrapper,
    _arun_cases_wrapper,
    _judge_aevaluate_wrapper,
    _judge_arun_wrapper,
    _judge_async_post_check_wrapper,
    _judge_evaluate_wrapper,
    _judge_post_check_wrapper,
    _judge_run_wrapper,
    _performance_arun_wrapper,
    _performance_run_wrapper,
    _reliability_arun_wrapper,
    _reliability_run_wrapper,
)
from .tracing import (
    _agent_arun_private_wrapper,
    _agent_arun_public_wrapper,
    _agent_arun_stream_wrapper,
    _agent_run_private_wrapper,
    _agent_run_public_wrapper,
    _agent_run_stream_wrapper,
    _function_call_aexecute_wrapper,
    _function_call_execute_wrapper,
    _model_ainvoke_stream_wrapper,
    _model_ainvoke_wrapper,
    _model_aresponse_stream_wrapper,
    _model_aresponse_wrapper,
    _model_invoke_stream_wrapper,
    _model_invoke_wrapper,
    _model_response_stream_wrapper,
    _model_response_wrapper,
    _team_arun_private_wrapper,
    _team_arun_public_wrapper,
    _team_arun_stream_wrapper,
    _team_run_private_wrapper,
    _team_run_public_wrapper,
    _team_run_stream_wrapper,
    _workflow_aexecute_stream_wrapper,
    _workflow_aexecute_workflow_agent_wrapper,
    _workflow_aexecute_wrapper,
    _workflow_execute_stream_wrapper,
    _workflow_execute_workflow_agent_wrapper,
    _workflow_execute_wrapper,
    spans_suppressed,
)


class _AgnoFunctionWrapperPatcher(FunctionWrapperPatcher):
    """Base for every agno patcher: hands through untouched while spans are suppressed.

    ``PerformanceEval`` measures a function that is usually an agent run, so without
    this the wrappers would still build span payloads — and retain every stream chunk —
    for each of the 60 default iterations, charging instrumentation cost to the very
    measurement being taken and then discarding the result. Returning ``wrapped(...)``
    serves sync and async targets alike: an async target's coroutine is simply handed
    back to the caller that awaits it.
    """

    @classmethod
    def _wrapper(cls, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if spans_suppressed():
            return wrapped(*args, **kwargs)
        return cls.wrapper(wrapped, instance, args, kwargs)


# ---------------------------------------------------------------------------
# Agent patchers
# ---------------------------------------------------------------------------

# Private methods have higher priority (lower number) so they are tried first.
# The public fallback patchers override applies() to yield when the private
# variant exists.


class _AgentRunPrivatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.agent.run.private"
    target_module = "agno.agent"
    target_path = "Agent._run"
    wrapper = _agent_run_private_wrapper
    priority: ClassVar[int] = 50


class _AgentRunPublicPatcher(_AgnoFunctionWrapperPatcher):
    """Fallback: wrap ``Agent.run`` only when ``Agent._run`` does not exist."""

    name = "agno.agent.run.public"
    target_module = "agno.agent"
    target_path = "Agent.run"
    wrapper = _agent_run_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_AgentRunPrivatePatcher,)


class _AgentArunPrivatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.agent.arun.private"
    target_module = "agno.agent"
    target_path = "Agent._arun"
    wrapper = _agent_arun_private_wrapper
    priority: ClassVar[int] = 50


class _AgentRunStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.agent.run_stream"
    target_module = "agno.agent"
    target_path = "Agent._run_stream"
    wrapper = _agent_run_stream_wrapper


class _AgentArunStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.agent.arun_stream"
    target_module = "agno.agent"
    target_path = "Agent._arun_stream"
    wrapper = _agent_arun_stream_wrapper
    priority: ClassVar[int] = 50


class _AgentArunPublicPatcher(_AgnoFunctionWrapperPatcher):
    """Fallback: wrap ``Agent.arun`` only when neither ``_arun`` nor ``_arun_stream`` exist."""

    name = "agno.agent.arun.public"
    target_module = "agno.agent"
    target_path = "Agent.arun"
    wrapper = _agent_arun_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_AgentArunPrivatePatcher, _AgentArunStreamPatcher)


class AgentPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.agent.Agent`` for tracing."""

    name = "agno.agent"
    sub_patchers = (
        _AgentRunPrivatePatcher,
        _AgentRunPublicPatcher,
        _AgentArunPrivatePatcher,
        _AgentRunStreamPatcher,
        _AgentArunStreamPatcher,
        _AgentArunPublicPatcher,
    )


# ---------------------------------------------------------------------------
# Team patchers
# ---------------------------------------------------------------------------


class _TeamRunPrivatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.team.run.private"
    target_module = "agno.team"
    target_path = "Team._run"
    wrapper = _team_run_private_wrapper
    priority: ClassVar[int] = 50


class _TeamRunPublicPatcher(_AgnoFunctionWrapperPatcher):
    """Fallback: wrap ``Team.run`` only when ``Team._run`` does not exist."""

    name = "agno.team.run.public"
    target_module = "agno.team"
    target_path = "Team.run"
    wrapper = _team_run_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_TeamRunPrivatePatcher,)


class _TeamArunPrivatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.team.arun.private"
    target_module = "agno.team"
    target_path = "Team._arun"
    wrapper = _team_arun_private_wrapper
    priority: ClassVar[int] = 50


class _TeamRunStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.team.run_stream"
    target_module = "agno.team"
    target_path = "Team._run_stream"
    wrapper = _team_run_stream_wrapper


class _TeamArunStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.team.arun_stream"
    target_module = "agno.team"
    target_path = "Team._arun_stream"
    wrapper = _team_arun_stream_wrapper
    priority: ClassVar[int] = 50


class _TeamArunPublicPatcher(_AgnoFunctionWrapperPatcher):
    """Fallback: wrap ``Team.arun`` only when neither ``_arun`` nor ``_arun_stream`` exist."""

    name = "agno.team.arun.public"
    target_module = "agno.team"
    target_path = "Team.arun"
    wrapper = _team_arun_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_TeamArunPrivatePatcher, _TeamArunStreamPatcher)


class TeamPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.team.Team`` for tracing."""

    name = "agno.team"
    sub_patchers = (
        _TeamRunPrivatePatcher,
        _TeamRunPublicPatcher,
        _TeamArunPrivatePatcher,
        _TeamRunStreamPatcher,
        _TeamArunStreamPatcher,
        _TeamArunPublicPatcher,
    )


# ---------------------------------------------------------------------------
# Model patchers
# ---------------------------------------------------------------------------


class _ModelInvokePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.invoke"
    target_module = "agno.models.base"
    target_path = "Model.invoke"
    wrapper = _model_invoke_wrapper


class _ModelAinvokePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.ainvoke"
    target_module = "agno.models.base"
    target_path = "Model.ainvoke"
    wrapper = _model_ainvoke_wrapper


class _ModelInvokeStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.invoke_stream"
    target_module = "agno.models.base"
    target_path = "Model.invoke_stream"
    wrapper = _model_invoke_stream_wrapper


class _ModelAinvokeStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.ainvoke_stream"
    target_module = "agno.models.base"
    target_path = "Model.ainvoke_stream"
    wrapper = _model_ainvoke_stream_wrapper


class _ModelResponsePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.response"
    target_module = "agno.models.base"
    target_path = "Model.response"
    wrapper = _model_response_wrapper


class _ModelAresponsePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.aresponse"
    target_module = "agno.models.base"
    target_path = "Model.aresponse"
    wrapper = _model_aresponse_wrapper


class _ModelResponseStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.response_stream"
    target_module = "agno.models.base"
    target_path = "Model.response_stream"
    wrapper = _model_response_stream_wrapper


class _ModelAresponseStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.model.aresponse_stream"
    target_module = "agno.models.base"
    target_path = "Model.aresponse_stream"
    wrapper = _model_aresponse_stream_wrapper


class ModelPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.models.base.Model`` for tracing."""

    name = "agno.model"
    sub_patchers = (
        _ModelInvokePatcher,
        _ModelAinvokePatcher,
        _ModelInvokeStreamPatcher,
        _ModelAinvokeStreamPatcher,
        _ModelResponsePatcher,
        _ModelAresponsePatcher,
        _ModelResponseStreamPatcher,
        _ModelAresponseStreamPatcher,
    )


# ---------------------------------------------------------------------------
# FunctionCall patchers
# ---------------------------------------------------------------------------


class _FunctionCallExecutePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.function_call.execute"
    target_module = "agno.tools.function"
    target_path = "FunctionCall.execute"
    wrapper = _function_call_execute_wrapper


class _FunctionCallAexecutePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.function_call.aexecute"
    target_module = "agno.tools.function"
    target_path = "FunctionCall.aexecute"
    wrapper = _function_call_aexecute_wrapper


class FunctionCallPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.tools.function.FunctionCall`` for tracing."""

    name = "agno.function_call"
    sub_patchers = (
        _FunctionCallExecutePatcher,
        _FunctionCallAexecutePatcher,
    )


# ---------------------------------------------------------------------------
# Workflow patchers (optional — requires fastapi)
# ---------------------------------------------------------------------------


class _WorkflowExecutePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.execute"
    target_module = "agno.workflow"
    target_path = "Workflow._execute"
    wrapper = _workflow_execute_wrapper


class _WorkflowExecuteStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.execute_stream"
    target_module = "agno.workflow"
    target_path = "Workflow._execute_stream"
    wrapper = _workflow_execute_stream_wrapper


class _WorkflowAexecutePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.aexecute"
    target_module = "agno.workflow"
    target_path = "Workflow._aexecute"
    wrapper = _workflow_aexecute_wrapper


class _WorkflowAexecuteStreamPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.aexecute_stream"
    target_module = "agno.workflow"
    target_path = "Workflow._aexecute_stream"
    wrapper = _workflow_aexecute_stream_wrapper


class _WorkflowExecuteWorkflowAgentPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.execute_workflow_agent"
    target_module = "agno.workflow"
    target_path = "Workflow._execute_workflow_agent"
    wrapper = _workflow_execute_workflow_agent_wrapper


class _WorkflowAexecuteWorkflowAgentPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.workflow.aexecute_workflow_agent"
    target_module = "agno.workflow"
    target_path = "Workflow._aexecute_workflow_agent"
    wrapper = _workflow_aexecute_workflow_agent_wrapper


class WorkflowPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.workflow.Workflow`` for tracing (optional — requires fastapi)."""

    name = "agno.workflow"
    sub_patchers = (
        _WorkflowExecutePatcher,
        _WorkflowExecuteStreamPatcher,
        _WorkflowAexecutePatcher,
        _WorkflowAexecuteStreamPatcher,
        _WorkflowExecuteWorkflowAgentPatcher,
        _WorkflowAexecuteWorkflowAgentPatcher,
    )


# ---------------------------------------------------------------------------
# Eval patchers (``agno.eval``)
# ---------------------------------------------------------------------------

# Every target lives in an ``agno.eval`` submodule that the eval package imports
# lazily, so each patcher names its own ``target_module``. Submodules that a given
# agno release does not ship (``agent_as_judge`` before 2.4, ``suite`` before 2.9)
# fail to import, ``resolve_root()`` returns None, and the patcher simply does not
# apply — no explicit version gate needed.


class _AccuracyEvalRunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.run"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.run"
    wrapper = _accuracy_run_wrapper


class _AccuracyEvalArunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.arun"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.arun"
    wrapper = _accuracy_arun_wrapper


class _AccuracyEvalRunWithOutputPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.run_with_output"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.run_with_output"
    wrapper = _accuracy_run_wrapper


class _AccuracyEvalArunWithOutputPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.arun_with_output"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.arun_with_output"
    wrapper = _accuracy_arun_wrapper


class _AccuracyEvalEvaluateAnswerPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.evaluate_answer"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.evaluate_answer"
    wrapper = _accuracy_evaluate_answer_wrapper


class _AccuracyEvalAevaluateAnswerPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.accuracy.aevaluate_answer"
    target_module = "agno.eval.accuracy"
    target_path = "AccuracyEval.aevaluate_answer"
    wrapper = _accuracy_aevaluate_answer_wrapper


class AccuracyEvalPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.eval.accuracy.AccuracyEval`` for tracing."""

    name = "agno.eval.accuracy"
    sub_patchers = (
        _AccuracyEvalRunPatcher,
        _AccuracyEvalArunPatcher,
        _AccuracyEvalRunWithOutputPatcher,
        _AccuracyEvalArunWithOutputPatcher,
        _AccuracyEvalEvaluateAnswerPatcher,
        _AccuracyEvalAevaluateAnswerPatcher,
    )


class _AgentAsJudgeEvalRunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.run"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval.run"
    wrapper = _judge_run_wrapper


class _AgentAsJudgeEvalArunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.arun"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval.arun"
    wrapper = _judge_arun_wrapper


class _AgentAsJudgeEvalEvaluatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.evaluate"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval._evaluate"
    wrapper = _judge_evaluate_wrapper


class _AgentAsJudgeEvalAevaluatePatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.aevaluate"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval._aevaluate"
    wrapper = _judge_aevaluate_wrapper


class _AgentAsJudgeEvalPostCheckPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.post_check"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval.post_check"
    wrapper = _judge_post_check_wrapper


class _AgentAsJudgeEvalAsyncPostCheckPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.agent_as_judge.async_post_check"
    target_module = "agno.eval.agent_as_judge"
    target_path = "AgentAsJudgeEval.async_post_check"
    wrapper = _judge_async_post_check_wrapper


class AgentAsJudgeEvalPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.eval.agent_as_judge.AgentAsJudgeEval`` for tracing (agno >= 2.4)."""

    name = "agno.eval.agent_as_judge"
    sub_patchers = (
        _AgentAsJudgeEvalRunPatcher,
        _AgentAsJudgeEvalArunPatcher,
        _AgentAsJudgeEvalEvaluatePatcher,
        _AgentAsJudgeEvalAevaluatePatcher,
        _AgentAsJudgeEvalPostCheckPatcher,
        _AgentAsJudgeEvalAsyncPostCheckPatcher,
    )


class _ReliabilityEvalRunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.reliability.run"
    target_module = "agno.eval.reliability"
    target_path = "ReliabilityEval.run"
    wrapper = _reliability_run_wrapper


class _ReliabilityEvalArunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.reliability.arun"
    target_module = "agno.eval.reliability"
    target_path = "ReliabilityEval.arun"
    wrapper = _reliability_arun_wrapper


class ReliabilityEvalPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.eval.reliability.ReliabilityEval`` for tracing."""

    name = "agno.eval.reliability"
    sub_patchers = (
        _ReliabilityEvalRunPatcher,
        _ReliabilityEvalArunPatcher,
    )


class _PerformanceEvalRunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.performance.run"
    target_module = "agno.eval.performance"
    target_path = "PerformanceEval.run"
    wrapper = _performance_run_wrapper


class _PerformanceEvalArunPatcher(_AgnoFunctionWrapperPatcher):
    name = "agno.eval.performance.arun"
    target_module = "agno.eval.performance"
    target_path = "PerformanceEval.arun"
    wrapper = _performance_arun_wrapper


class PerformanceEvalPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.eval.performance.PerformanceEval`` for tracing."""

    name = "agno.eval.performance"
    sub_patchers = (
        _PerformanceEvalRunPatcher,
        _PerformanceEvalArunPatcher,
    )


class _EvalSuiteRunCasesPatcher(_AgnoFunctionWrapperPatcher):
    """Patch the suite runner, not ``cli``/``run_cases``.

    ``cli``, ``acli`` and ``run_cases`` all resolve the next call through the module
    globals at call time, so wrapping ``arun_cases`` covers every entry point — including
    the common ``from agno.eval import cli`` layout, where the caller's name is bound
    before ``setup_agno()`` gets a chance to patch anything.
    """

    name = "agno.eval.suite.arun_cases"
    target_module = "agno.eval.suite"
    target_path = "arun_cases"
    wrapper = _arun_cases_wrapper


class _EvalSuiteRunCasePatcher(_AgnoFunctionWrapperPatcher):
    """Patch the private per-case runner: it is the only per-case seam.

    The public presentation hooks (``on_case_start``/``on_case_end``) are two separate
    callbacks, so a span cannot be held current across the case body, and ``cli()``
    passes its own renderer into them.

    ``applies()`` fails open (an unresolvable target simply does not apply), so if agno
    renames this, per-case rows disappear silently while ``arun_cases`` above keeps
    opening an experiment — an empty experiment with no error. The suite tests assert
    one row per case, and are what catches that.
    """

    name = "agno.eval.suite.arun_case"
    target_module = "agno.eval.suite"
    target_path = "_arun_case"
    wrapper = _arun_case_wrapper


class EvalSuitePatcher(CompositeFunctionWrapperPatcher):
    """Patch ``agno.eval.suite`` for tracing (agno >= 2.9)."""

    name = "agno.eval.suite"
    sub_patchers = (
        _EvalSuiteRunCasesPatcher,
        _EvalSuiteRunCasePatcher,
    )


# ---------------------------------------------------------------------------
# Public wrap_*() helpers — thin wrappers around patcher.wrap_target()
# ---------------------------------------------------------------------------


def wrap_agent(Agent: Any) -> Any:
    """Manually patch an Agent class for tracing."""
    return AgentPatcher.wrap_target(Agent)


def wrap_team(Team: Any) -> Any:
    """Manually patch a Team class for tracing."""
    return TeamPatcher.wrap_target(Team)


def wrap_model(Model: Any) -> Any:
    """Manually patch a Model class for tracing."""
    return ModelPatcher.wrap_target(Model)


def wrap_function_call(FunctionCall: Any) -> Any:
    """Manually patch a FunctionCall class for tracing."""
    return FunctionCallPatcher.wrap_target(FunctionCall)


def wrap_workflow(Workflow: Any) -> Any:
    """Manually patch a Workflow class for tracing."""
    return WorkflowPatcher.wrap_target(Workflow)


def wrap_accuracy_eval(AccuracyEval: Any) -> Any:
    """Manually patch an AccuracyEval class for tracing."""
    return AccuracyEvalPatcher.wrap_target(AccuracyEval)


def wrap_agent_as_judge_eval(AgentAsJudgeEval: Any) -> Any:
    """Manually patch an AgentAsJudgeEval class for tracing."""
    return AgentAsJudgeEvalPatcher.wrap_target(AgentAsJudgeEval)


def wrap_reliability_eval(ReliabilityEval: Any) -> Any:
    """Manually patch a ReliabilityEval class for tracing."""
    return ReliabilityEvalPatcher.wrap_target(ReliabilityEval)


def wrap_performance_eval(PerformanceEval: Any) -> Any:
    """Manually patch a PerformanceEval class for tracing."""
    return PerformanceEvalPatcher.wrap_target(PerformanceEval)


def wrap_eval_suite(suite_module: Any) -> Any:
    """Manually patch the ``agno.eval.suite`` module for tracing."""
    return EvalSuitePatcher.wrap_target(suite_module)
