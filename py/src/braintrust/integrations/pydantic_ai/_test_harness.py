from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel as PydanticTestModel


OFFLINE_ANSWER = "offline answer"


def make_harness_agent(name, model=None, *, capabilities=(), **kwargs):
    if model is None:
        model = PydanticTestModel(custom_output_text=OFFLINE_ANSWER)
    return Agent(model, name=name, capabilities=list(capabilities), **kwargs)


def call_tool_once(tool_name, args=None, *, tool_call_id="offline-call"):
    def respond(messages, _info):
        if any(
            isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart(OFFLINE_ANSWER)])
        return ModelResponse(parts=[ToolCallPart(tool_name, args or {}, tool_call_id=tool_call_id)])

    return FunctionModel(respond, model_name=f"function:{tool_name}")
