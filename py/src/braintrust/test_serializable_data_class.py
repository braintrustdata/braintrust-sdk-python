import copy
import pickle
import unittest
from dataclasses import dataclass, field

from .serializable_data_class import _EXPLICITLY_SET_FIELDS_ATTR, _INIT_FIELDS_REMAINING_ATTR, SerializableDataClass


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


@dataclass
class Child(SerializableDataClass):
    value: str | None = None
    label: str = "child"


@dataclass
class Parent(SerializableDataClass):
    child: Child | None = None
    children: list[Child] | None = None
    metadata: dict | None = None


@dataclass
class WithFactory(SerializableDataClass):
    items: list[str] = field(default_factory=list)


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

    def test_as_dict_exclude_unset_omits_defaults(self):
        prompt_data = PromptData()

        self.assertEqual(prompt_data.as_dict(), {"prompt": None, "options": None})
        self.assertEqual(prompt_data.as_dict(exclude_unset=True), {})

    def test_as_dict_exclude_unset_keeps_explicit_none(self):
        keyword_prompt_data = PromptData(prompt=None)
        positional_prompt_data = PromptData(None)

        self.assertEqual(keyword_prompt_data.as_dict(exclude_unset=True), {"prompt": None})
        self.assertEqual(positional_prompt_data.as_dict(exclude_unset=True), {"prompt": None})

    def test_as_dict_exclude_unset_keeps_default_factory_values(self):
        default_factory = WithFactory()
        explicit_factory_value = WithFactory(items=[])

        self.assertEqual(default_factory.as_dict(), {"items": []})
        self.assertEqual(default_factory.as_dict(exclude_unset=True), {"items": []})
        self.assertEqual(explicit_factory_value.as_dict(exclude_unset=True), {"items": []})

    def test_as_dict_exclude_unset_tracks_assignments(self):
        prompt_data = PromptData()

        prompt_data.prompt = None

        self.assertEqual(prompt_data.as_dict(exclude_unset=True), {"prompt": None})

    def test_as_dict_exclude_unset_tracks_assignments_after_copy_or_pickle(self):
        reconstructors = (
            ("copy", copy.copy),
            ("deepcopy", copy.deepcopy),
            ("pickle", lambda value: pickle.loads(pickle.dumps(value))),
        )

        for name, reconstruct in reconstructors:
            with self.subTest(name=name):
                original = PromptData()
                prompt_data = reconstruct(original)

                self.assertFalse(hasattr(prompt_data, _INIT_FIELDS_REMAINING_ATTR))
                prompt_data.prompt = None

                self.assertEqual(original.as_dict(exclude_unset=True), {})
                self.assertEqual(prompt_data.as_dict(exclude_unset=True), {"prompt": None})

        for name, reconstruct in reconstructors:
            with self.subTest(name=f"{name}_explicit"):
                original = PromptData(prompt=None)
                prompt_data = reconstruct(original)

                self.assertFalse(hasattr(prompt_data, _INIT_FIELDS_REMAINING_ATTR))
                self.assertEqual(prompt_data.as_dict(exclude_unset=True), {"prompt": None})
                prompt_data.options = None

                self.assertEqual(original.as_dict(exclude_unset=True), {"prompt": None})
                self.assertEqual(prompt_data.as_dict(exclude_unset=True), {"prompt": None, "options": None})

    def test_as_dict_exclude_unset_recurses_into_nested_values(self):
        parent = Parent(
            child=Child(value=None),
            children=[Child(label="child")],
            metadata={"nested": Child(value="set")},
        )

        self.assertEqual(
            parent.as_dict(exclude_unset=True),
            {
                "child": {"value": None, "label": "child"},
                "children": [{"label": "child"}],
                "metadata": {"nested": {"value": "set", "label": "child"}},
            },
        )

    def test_from_dict_deep_tracks_explicit_none_without_marking_missing_defaults(self):
        test_dict = {
            "id": "456",
            "project_id": "123",
            "_xact_id": "789",
            "name": "test-prompt",
            "slug": "test-prompt",
            "description": None,
            "prompt_data": {"prompt": None},
            "tags": None,
        }

        prompt = PromptSchema.from_dict_deep(test_dict)

        self.assertEqual(prompt.as_dict(exclude_unset=True), test_dict)

    def test_as_json_supports_exclude_unset(self):
        prompt_data = PromptData(prompt=None)

        self.assertEqual(prompt_data.as_json(exclude_unset=True, sort_keys=True), '{"prompt": null}')

    def test_as_dict_exclude_unset_without_tracking_falls_back_to_full_serialization(self):
        prompt_data = PromptData()
        object.__delattr__(prompt_data, _EXPLICITLY_SET_FIELDS_ATTR)

        self.assertEqual(prompt_data.as_dict(exclude_unset=True), prompt_data.as_dict())


if __name__ == "__main__":
    unittest.main()
