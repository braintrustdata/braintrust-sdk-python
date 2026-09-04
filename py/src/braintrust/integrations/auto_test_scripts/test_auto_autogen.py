from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.teams import BaseGroupChat
from autogen_core.tools import FunctionTool
from braintrust.integrations.test_utils import run_auto_smoke


_PATCHED_MARKERS = {
    BaseChatAgent.run: "__braintrust_patched_autogen_chat_agent_run__",
    AssistantAgent.on_messages_stream: "__braintrust_patched_autogen_chat_agent_assistant_on_messages_stream__",
    BaseGroupChat.run: "__braintrust_patched_autogen_team_run__",
    FunctionTool.run: "__braintrust_patched_autogen_function_tool_run__",
}


def _is_patched() -> bool:
    return all(getattr(target, marker, False) for target, marker in _PATCHED_MARKERS.items())


run_auto_smoke("autogen", is_patched=_is_patched)

# Additional marker checks not covered by the shared runner.
assert getattr(BaseChatAgent.run_stream, "__braintrust_patched_autogen_chat_agent_run_stream__", False)
assert getattr(BaseChatAgent.on_messages, "__braintrust_patched_autogen_chat_agent_base_on_messages__", False)
assert getattr(
    BaseChatAgent.on_messages_stream, "__braintrust_patched_autogen_chat_agent_base_on_messages_stream__", False
)
assert getattr(AssistantAgent.on_messages, "__braintrust_patched_autogen_chat_agent_assistant_on_messages__", False)
assert getattr(BaseGroupChat.run_stream, "__braintrust_patched_autogen_team_run_stream__", False)

print("SUCCESS")
