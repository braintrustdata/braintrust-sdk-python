"""Patch the shared Conversation entry points used by Agent.chat()."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_chat_wrapper,
    _agent_exit_wrapper,
    _chat_wrapper,
    _lifecycle_wrapper,
    _receive_steps_wrapper,
    _tool_execute_wrapper,
)


class _AgentChatPatcher(FunctionWrapperPatcher):
    name = "antigravity.agent.chat"
    target_path = "Agent.chat"
    wrapper = _agent_chat_wrapper


class _AgentExitPatcher(FunctionWrapperPatcher):
    name = "antigravity.agent.exit"
    target_path = "Agent.__aexit__"
    wrapper = _agent_exit_wrapper


class AgentPatcher(CompositeFunctionWrapperPatcher):
    name = "antigravity.agent"
    sub_patchers = (_AgentChatPatcher, _AgentExitPatcher)

    @classmethod
    def wrap_target(cls, target):
        for patcher in cls.sub_patchers:
            method = getattr(target, patcher.target_path.rsplit(".", 1)[-1], None)
            # setup() marks methods; manual wrapping otherwise marks the class.
            # Honor either entry point's marker before adding another wrapper.
            if not patcher.has_patch_marker(method):
                patcher.wrap_target(target)
        return target


class ChatPatcher(FunctionWrapperPatcher):
    name = "antigravity.conversation.chat"
    target_module = "google.antigravity.conversation.conversation"
    target_path = "Conversation.chat"
    wrapper = _chat_wrapper


class ReceiveStepsPatcher(FunctionWrapperPatcher):
    name = "antigravity.conversation.receive_steps"
    target_module = "google.antigravity.conversation.conversation"
    target_path = "Conversation.receive_steps"
    wrapper = _receive_steps_wrapper


class ToolExecutePatcher(FunctionWrapperPatcher):
    name = "antigravity.tool.execute"
    target_module = "google.antigravity.tools.tool_runner"
    target_path = "ToolRunner.execute"
    wrapper = _tool_execute_wrapper


class _CancelPatcher(FunctionWrapperPatcher):
    name = "antigravity.conversation.cancel"
    target_module = "google.antigravity.conversation.conversation"
    target_path = "Conversation.cancel"
    wrapper = _lifecycle_wrapper


class _DisconnectPatcher(FunctionWrapperPatcher):
    name = "antigravity.conversation.disconnect"
    target_module = "google.antigravity.conversation.conversation"
    target_path = "Conversation.disconnect"
    wrapper = _lifecycle_wrapper


class ConversationLifecyclePatcher(CompositeFunctionWrapperPatcher):
    name = "antigravity.conversation.lifecycle"
    sub_patchers = (_CancelPatcher, _DisconnectPatcher)
