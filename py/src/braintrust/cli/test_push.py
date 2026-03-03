"""Tests for push command serialization."""

import pytest

pydantic = pytest.importorskip("pydantic")

from ..framework2 import (
    global_,
    projects,
)
from .push import _collect_function_function_defs


class ToolInput(pydantic.BaseModel):
    value: int


@pytest.fixture(autouse=True)
def clear_global_state():
    global_.functions.clear()
    global_.prompts.clear()
    yield
    global_.functions.clear()
    global_.prompts.clear()


class TestPushMetadata:
    """Tests for metadata in push command serialization."""

    def test_collect_function_function_defs_includes_metadata(self, mock_project_ids):
        project = projects.create("test-project")
        metadata = {"version": "1.0", "author": "test"}

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=ToolInput,
            metadata=metadata,
        )
        global_.functions.append(tool)

        functions = []
        _collect_function_function_defs(mock_project_ids, functions, "bundle-123", "error")

        assert len(functions) == 1
        assert functions[0]["metadata"] == metadata
        assert functions[0]["name"] == "test-tool"

    def test_collect_function_function_defs_excludes_metadata_when_none(self, mock_project_ids):
        project = projects.create("test-project")

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=ToolInput,
        )
        global_.functions.append(tool)

        functions = []
        _collect_function_function_defs(mock_project_ids, functions, "bundle-123", "error")

        assert len(functions) == 1
        assert "metadata" not in functions[0]


class TestPushTags:
    """Tests for tags in push command serialization."""

    def test_collect_function_function_defs_includes_tags(self, mock_project_ids):
        project = projects.create("test-project")
        tags = ["production", "v1"]

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=ToolInput,
            tags=tags,
        )
        global_.functions.append(tool)

        functions = []
        _collect_function_function_defs(mock_project_ids, functions, "bundle-123", "error")

        assert len(functions) == 1
        assert functions[0]["tags"] == ["production", "v1"]

    def test_collect_function_function_defs_excludes_tags_when_none(self, mock_project_ids):
        project = projects.create("test-project")

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=ToolInput,
        )
        global_.functions.append(tool)

        functions = []
        _collect_function_function_defs(mock_project_ids, functions, "bundle-123", "error")

        assert len(functions) == 1
        assert "tags" not in functions[0]
