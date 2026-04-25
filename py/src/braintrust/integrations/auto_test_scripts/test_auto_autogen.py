from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.teams import BaseGroupChat
from autogen_core.tools import FunctionTool
from braintrust.auto import auto_instrument


assert not getattr(BaseChatAgent.run, "__braintrust_patched_autogen_chat_agent_run__", False)
assert not getattr(BaseGroupChat.run, "__braintrust_patched_autogen_team_run__", False)
assert not getattr(FunctionTool.run, "__braintrust_patched_autogen_function_tool_run__", False)

results = auto_instrument()
assert results.get("autogen") == True
assert auto_instrument().get("autogen") == True

assert getattr(BaseChatAgent.run, "__braintrust_patched_autogen_chat_agent_run__", False)
assert getattr(BaseChatAgent.run_stream, "__braintrust_patched_autogen_chat_agent_run_stream__", False)
assert getattr(BaseGroupChat.run, "__braintrust_patched_autogen_team_run__", False)
assert getattr(BaseGroupChat.run_stream, "__braintrust_patched_autogen_team_run_stream__", False)
assert getattr(FunctionTool.run, "__braintrust_patched_autogen_function_tool_run__", False)

print("SUCCESS")
