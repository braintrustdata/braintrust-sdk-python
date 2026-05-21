"""AutoGen patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_on_messages_stream_wrapper,
    _agent_on_messages_wrapper,
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


class BaseChatAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.base.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "BaseChatAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class AssistantAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.assistant.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "AssistantAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class CodeExecutorAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.code_executor.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "CodeExecutorAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class MessageFilterAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.message_filter.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "MessageFilterAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class SocietyOfMindAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.society_of_mind.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "SocietyOfMindAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class UserProxyAgentOnMessagesPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.user_proxy.on_messages"
    target_module = "autogen_agentchat.agents"
    target_path = "UserProxyAgent.on_messages"
    wrapper = _agent_on_messages_wrapper


class BaseChatAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.base.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "BaseChatAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class AssistantAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.assistant.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "AssistantAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class CodeExecutorAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.code_executor.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "CodeExecutorAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class MessageFilterAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.message_filter.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "MessageFilterAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class SocietyOfMindAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.society_of_mind.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "SocietyOfMindAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class UserProxyAgentOnMessagesStreamPatcher(FunctionWrapperPatcher):
    name = "autogen.chat_agent.user_proxy.on_messages_stream"
    target_module = "autogen_agentchat.agents"
    target_path = "UserProxyAgent.on_messages_stream"
    wrapper = _agent_on_messages_stream_wrapper


class ChatAgentPatcher(CompositeFunctionWrapperPatcher):
    name = "autogen.chat_agent"
    sub_patchers = (
        ChatAgentRunPatcher,
        ChatAgentRunStreamPatcher,
        BaseChatAgentOnMessagesPatcher,
        AssistantAgentOnMessagesPatcher,
        CodeExecutorAgentOnMessagesPatcher,
        MessageFilterAgentOnMessagesPatcher,
        SocietyOfMindAgentOnMessagesPatcher,
        UserProxyAgentOnMessagesPatcher,
        BaseChatAgentOnMessagesStreamPatcher,
        AssistantAgentOnMessagesStreamPatcher,
        CodeExecutorAgentOnMessagesStreamPatcher,
        MessageFilterAgentOnMessagesStreamPatcher,
        SocietyOfMindAgentOnMessagesStreamPatcher,
        UserProxyAgentOnMessagesStreamPatcher,
    )


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
