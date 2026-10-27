import unittest
from dataclasses import dataclass

from .prompt import FunctionCall, PromptMessage
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
    def test_union_preserves_string_and_none_values(self):
        for function_call in ("{{function_call}}", None):
            with self.subTest(function_call=function_call):
                message = PromptMessage.from_dict_deep(
                    {"role": "assistant", "content": "", "function_call": function_call}
                )
                self.assertEqual(message.function_call, function_call)

    def test_union_deserializes_nested_dataclass(self):
        message = PromptMessage.from_dict_deep(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": "lookup", "arguments": "{}"},
            }
        )
        self.assertEqual(message.function_call, FunctionCall(name="lookup", arguments="{}"))

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


if __name__ == "__main__":
    unittest.main()
