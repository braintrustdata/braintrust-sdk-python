import unittest
from dataclasses import dataclass

from .prompt import FunctionCall, PromptChatBlock, PromptMessage
from .prompt import PromptData as RealPromptData
from .serializable_data_class import SerializableDataClass


@dataclass
class PromptData(SerializableDataClass):
    prompt: str | None = None
    options: dict | None = None


@dataclass
class PromptSchema(SerializableDataClass):
    id: str
    project_id: str
    _xact_id: str
    name: str
    slug: str
    description: str | None
    prompt_data: PromptData
    tags: list[str] | None


class TestSerializableDataClass(unittest.TestCase):
    def test_from_dict_deep_with_none_values(self):
        """Test that from_dict_deep correctly handles None values in nested objects."""
        test_dict = {
            "id": "456",
            "project_id": "123",
            "_xact_id": "789",
            "name": "test-prompt",
            "slug": "test-prompt",
            "description": None,
            "prompt_data": {"prompt": None, "options": None},
            "tags": None,
        }

        prompt = PromptSchema.from_dict_deep(test_dict)

        # Verify all fields were set correctly.
        self.assertEqual(prompt.id, "456")
        self.assertEqual(prompt.project_id, "123")
        self.assertEqual(prompt._xact_id, "789")
        self.assertEqual(prompt.name, "test-prompt")
        self.assertEqual(prompt.slug, "test-prompt")
        self.assertIsNone(prompt.description)
        self.assertIsNone(prompt.tags)

        # Verify nested object was created and its fields are None.
        self.assertIsInstance(prompt.prompt_data, PromptData)
        self.assertIsNone(prompt.prompt_data.prompt)
        self.assertIsNone(prompt.prompt_data.options)

        # Verify round-trip serialization works.
        round_trip = PromptSchema.from_dict_deep(prompt.as_dict())
        self.assertEqual(round_trip.as_dict(), test_dict)


class TestUnionDeserializationScalars(unittest.TestCase):
    def test_from_dict_deep_preserves_function_call_union_members(self):
        cases = [
            ("legacy_fn", "legacy_fn"),
            (None, None),
            (
                {"name": "lookup", "arguments": '{"q": 1}'},
                FunctionCall(name="lookup", arguments='{"q": 1}'),
            ),
            (
                FunctionCall(name="lookup", arguments='{"q": 1}'),
                FunctionCall(name="lookup", arguments='{"q": 1}'),
            ),
        ]
        for function_call, expected in cases:
            with self.subTest(function_call=function_call):
                message = PromptMessage(content="hi", role="user", function_call=function_call)
                restored = PromptMessage.from_dict_deep(message.as_dict())
                self.assertEqual(restored.function_call, expected)
                self.assertEqual(restored, PromptMessage(content="hi", role="user", function_call=expected))

    def test_from_dict_deep_accepts_raw_legacy_function_call(self):
        restored = PromptMessage.from_dict_deep({"content": "hi", "role": "user", "function_call": "legacy_fn"})
        self.assertEqual(restored.function_call, "legacy_fn")

    def test_prompt_data_preserves_legacy_function_call_in_nested_message(self):
        prompt_data = RealPromptData(
            prompt=PromptChatBlock(messages=[PromptMessage(content="hi", role="user", function_call="legacy_fn")])
        )
        restored = RealPromptData.from_dict_deep(prompt_data.as_dict())
        self.assertEqual(restored, prompt_data)
        self.assertEqual(restored.prompt.messages[0].function_call, "legacy_fn")


if __name__ == "__main__":
    unittest.main()
