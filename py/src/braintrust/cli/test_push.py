"""Tests for push command serialization."""

import sys

import pytest


pydantic = pytest.importorskip("pydantic")

from ..framework2 import (
    global_,
    projects,
)
from .push import _collect_function_function_defs, _validate_python_bundle_source_paths


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


class TestValidatePythonBundleSourcePaths:
    def _assert_whitespace_in_filename_rejected(self, tmp_path, filename: str):
        source = tmp_path / filename
        source.write_text("VALUE = 1\n")

        with pytest.raises(ValueError, match="contains whitespace in path component"):
            _validate_python_bundle_source_paths([str(source)], str(tmp_path))

    def test_rejects_whitespace_in_filename(self, tmp_path):
        self._assert_whitespace_in_filename_rejected(tmp_path, "my tool.py")

    @pytest.mark.skipif(sys.platform == "win32", reason="leading-space filenames are not portable on Windows")
    def test_rejects_leading_whitespace_in_filename(self, tmp_path):
        self._assert_whitespace_in_filename_rejected(tmp_path, " tool.py")

    def test_rejects_whitespace_in_directory_name(self, tmp_path):
        subdir = tmp_path / "my tools"
        subdir.mkdir()
        source = subdir / "tool.py"
        source.write_text("VALUE = 1\n")

        with pytest.raises(ValueError, match="contains whitespace in path component"):
            _validate_python_bundle_source_paths([str(source)], str(tmp_path))

    @pytest.mark.skipif(sys.platform == "win32", reason="tab characters in filenames are not allowed on Windows")
    def test_rejects_tab_in_filename(self, tmp_path):
        self._assert_whitespace_in_filename_rejected(tmp_path, "my\ttool.py")

    def test_accepts_valid_path(self, tmp_path):
        source = tmp_path / "my_tool.py"
        source.write_text("VALUE = 1\n")
        _validate_python_bundle_source_paths([str(source)], str(tmp_path))
