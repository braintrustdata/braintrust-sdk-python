from typing import Any, ClassVar

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

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
)


# ---------------------------------------------------------------------------
# Agent patchers
# ---------------------------------------------------------------------------

# Private methods have higher priority (lower number) so they are tried first.
# The public fallback patchers override applies() to yield when the private
# variant exists.


class _AgentRunPrivatePatcher(FunctionWrapperPatcher):
    name = "agno.agent.run.private"
    target_module = "agno.agent"
    target_path = "Agent._run"
    wrapper = _agent_run_private_wrapper
    priority: ClassVar[int] = 50


class _AgentRunPublicPatcher(FunctionWrapperPatcher):
    """Fallback: wrap ``Agent.run`` only when ``Agent._run`` does not exist."""

    name = "agno.agent.run.public"
    target_module = "agno.agent"
    target_path = "Agent.run"
    wrapper = _agent_run_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_AgentRunPrivatePatcher,)


class _AgentArunPrivatePatcher(FunctionWrapperPatcher):
    name = "agno.agent.arun.private"
    target_module = "agno.agent"
    target_path = "Agent._arun"
    wrapper = _agent_arun_private_wrapper
    priority: ClassVar[int] = 50


class _AgentRunStreamPatcher(FunctionWrapperPatcher):
    name = "agno.agent.run_stream"
    target_module = "agno.agent"
    target_path = "Agent._run_stream"
    wrapper = _agent_run_stream_wrapper


class _AgentArunStreamPatcher(FunctionWrapperPatcher):
    name = "agno.agent.arun_stream"
    target_module = "agno.agent"
    target_path = "Agent._arun_stream"
    wrapper = _agent_arun_stream_wrapper
    priority: ClassVar[int] = 50


class _AgentArunPublicPatcher(FunctionWrapperPatcher):
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


class _TeamRunPrivatePatcher(FunctionWrapperPatcher):
    name = "agno.team.run.private"
    target_module = "agno.team"
    target_path = "Team._run"
    wrapper = _team_run_private_wrapper
    priority: ClassVar[int] = 50


class _TeamRunPublicPatcher(FunctionWrapperPatcher):
    """Fallback: wrap ``Team.run`` only when ``Team._run`` does not exist."""

    name = "agno.team.run.public"
    target_module = "agno.team"
    target_path = "Team.run"
    wrapper = _team_run_public_wrapper
    priority: ClassVar[int] = 100
    superseded_by = (_TeamRunPrivatePatcher,)


class _TeamArunPrivatePatcher(FunctionWrapperPatcher):
    name = "agno.team.arun.private"
    target_module = "agno.team"
    target_path = "Team._arun"
    wrapper = _team_arun_private_wrapper
    priority: ClassVar[int] = 50


class _TeamRunStreamPatcher(FunctionWrapperPatcher):
    name = "agno.team.run_stream"
    target_module = "agno.team"
    target_path = "Team._run_stream"
    wrapper = _team_run_stream_wrapper


class _TeamArunStreamPatcher(FunctionWrapperPatcher):
    name = "agno.team.arun_stream"
    target_module = "agno.team"
    target_path = "Team._arun_stream"
    wrapper = _team_arun_stream_wrapper
    priority: ClassVar[int] = 50


class _TeamArunPublicPatcher(FunctionWrapperPatcher):
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


class _ModelInvokePatcher(FunctionWrapperPatcher):
    name = "agno.model.invoke"
    target_module = "agno.models.base"
    target_path = "Model.invoke"
    wrapper = _model_invoke_wrapper


class _ModelAinvokePatcher(FunctionWrapperPatcher):
    name = "agno.model.ainvoke"
    target_module = "agno.models.base"
    target_path = "Model.ainvoke"
    wrapper = _model_ainvoke_wrapper


class _ModelInvokeStreamPatcher(FunctionWrapperPatcher):
    name = "agno.model.invoke_stream"
    target_module = "agno.models.base"
    target_path = "Model.invoke_stream"
    wrapper = _model_invoke_stream_wrapper


class _ModelAinvokeStreamPatcher(FunctionWrapperPatcher):
    name = "agno.model.ainvoke_stream"
    target_module = "agno.models.base"
    target_path = "Model.ainvoke_stream"
    wrapper = _model_ainvoke_stream_wrapper


class _ModelResponsePatcher(FunctionWrapperPatcher):
    name = "agno.model.response"
    target_module = "agno.models.base"
    target_path = "Model.response"
    wrapper = _model_response_wrapper


class _ModelAresponsePatcher(FunctionWrapperPatcher):
    name = "agno.model.aresponse"
    target_module = "agno.models.base"
    target_path = "Model.aresponse"
    wrapper = _model_aresponse_wrapper


class _ModelResponseStreamPatcher(FunctionWrapperPatcher):
    name = "agno.model.response_stream"
    target_module = "agno.models.base"
    target_path = "Model.response_stream"
    wrapper = _model_response_stream_wrapper


class _ModelAresponseStreamPatcher(FunctionWrapperPatcher):
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


class _FunctionCallExecutePatcher(FunctionWrapperPatcher):
    name = "agno.function_call.execute"
    target_module = "agno.tools.function"
    target_path = "FunctionCall.execute"
    wrapper = _function_call_execute_wrapper


class _FunctionCallAexecutePatcher(FunctionWrapperPatcher):
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


class _WorkflowExecutePatcher(FunctionWrapperPatcher):
    name = "agno.workflow.execute"
    target_module = "agno.workflow"
    target_path = "Workflow._execute"
    wrapper = _workflow_execute_wrapper


class _WorkflowExecuteStreamPatcher(FunctionWrapperPatcher):
    name = "agno.workflow.execute_stream"
    target_module = "agno.workflow"
    target_path = "Workflow._execute_stream"
    wrapper = _workflow_execute_stream_wrapper


class _WorkflowAexecutePatcher(FunctionWrapperPatcher):
    name = "agno.workflow.aexecute"
    target_module = "agno.workflow"
    target_path = "Workflow._aexecute"
    wrapper = _workflow_aexecute_wrapper


class _WorkflowAexecuteStreamPatcher(FunctionWrapperPatcher):
    name = "agno.workflow.aexecute_stream"
    target_module = "agno.workflow"
    target_path = "Workflow._aexecute_stream"
    wrapper = _workflow_aexecute_stream_wrapper


class _WorkflowExecuteWorkflowAgentPatcher(FunctionWrapperPatcher):
    name = "agno.workflow.execute_workflow_agent"
    target_module = "agno.workflow"
    target_path = "Workflow._execute_workflow_agent"
    wrapper = _workflow_execute_workflow_agent_wrapper


class _WorkflowAexecuteWorkflowAgentPatcher(FunctionWrapperPatcher):
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
