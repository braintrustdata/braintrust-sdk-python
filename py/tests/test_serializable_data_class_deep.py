"""Regression tests for type-guided union deserialization in ``from_dict_deep``.

A union member that is a ``SerializableDataClass`` must only be tried when the
incoming value is mapping-like. Otherwise a valid scalar member (for example
``PromptMessage.function_call: str | FunctionCall | None``) is handed to a
dataclass constructor that expects a mapping and raises.
"""

import pytest
from braintrust.prompt import FunctionCall, PromptChatBlock, PromptData, PromptMessage


@pytest.mark.parametrize(
    ("function_call", "expected"),
    [
        ("legacy_fn", "legacy_fn"),
        (None, None),
        ({"name": "lookup", "arguments": '{"q": 1}'}, FunctionCall(name="lookup", arguments='{"q": 1}')),
        (FunctionCall(name="lookup", arguments='{"q": 1}'), FunctionCall(name="lookup", arguments='{"q": 1}')),
    ],
    ids=["str", "none", "dict", "dataclass"],
)
def test_prompt_message_from_dict_deep_preserves_function_call(function_call, expected):
    message = PromptMessage(content="hi", role="user", function_call=function_call)

    restored = PromptMessage.from_dict_deep(message.as_dict())

    assert restored.function_call == expected
    assert restored == PromptMessage(content="hi", role="user", function_call=expected)


def test_prompt_message_from_dict_deep_accepts_raw_legacy_function_call():
    restored = PromptMessage.from_dict_deep({"content": "hi", "role": "user", "function_call": "legacy_fn"})

    assert restored.function_call == "legacy_fn"


def test_prompt_data_from_dict_deep_preserves_legacy_function_call_in_nested_message():
    prompt_data = PromptData(
        prompt=PromptChatBlock(messages=[PromptMessage(content="hi", role="user", function_call="legacy_fn")])
    )

    restored = PromptData.from_dict_deep(prompt_data.as_dict())

    assert restored == prompt_data
    assert restored.prompt.messages[0].function_call == "legacy_fn"
