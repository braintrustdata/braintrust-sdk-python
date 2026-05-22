"""AutoGen patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_run_stream_wrapper,
    _agent_run_wrapper,
    _team_run_stream_wrapper,
    _team_run_wrapper,
    _tool_run_wrapper,
)


class ChatAgentRunPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.run"
    target_module = "autogen_agentchat.agents"
    target_path = "BaseChatAgent.run"
    wrapper = _agent_run_wrapper


class ChatAgentRunStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.run_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "BaseChatAgent.run_stream"
    wrapper = _agent_run_stream_wrapper


class ChatAgentPatcher(CompositeFunctionWrapperPatcher):
    name = "autogen.chat_agent"
    sub_patchers = (ChatAgentRunPatcher, ChatAgentRunStreamPatcher)


class TeamRunPatcher(FunctionWrapperPatcher):
    name = "autogen.team.run"
    target_module = "autogen_agentchat.teams"
    target_path = "BaseGroupChat.run"
    wrapper = _team_run_wrapper


class TeamRunStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.team.run_stream"
    target_module = "autogen_agentchat.teams"
    target_path = "BaseGroupChat.run_stream"
    wrapper = _team_run_stream_wrapper


class TeamPatcher(CompositeFunctionWrapperPatcher):
    name = "autogen.team"
    sub_patchers = (TeamRunPatcher, TeamRunStreamPatcher)


class FunctionToolRunPatcher(FunctionWrapperPatcher):
    name = "autogen.function_tool.run"
    target_module = "autogen_core.tools"
    target_path = "FunctionTool.run"
    wrapper = _tool_run_wrapper
