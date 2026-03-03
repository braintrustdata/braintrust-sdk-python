"""Tests for framework2 module, specifically metadata and tags support."""

import importlib.util

import pytest

from .framework2 import projects

# Check if pydantic is available
HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


class TestCodeFunctionMetadata:
    """Tests for CodeFunction metadata support."""

    def test_code_function_with_metadata(self):
        project = projects.create("test-project")
        metadata = {"version": "1.0", "author": "test"}

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=None,
            metadata=metadata,
        )

        assert tool.metadata == metadata
        assert tool.name == "test-tool"
        assert tool.slug == "test-tool"

    def test_code_function_without_metadata(self):
        project = projects.create("test-project")

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=None,
        )

        assert tool.metadata is None


class TestCodePromptMetadata:
    """Tests for CodePrompt metadata support."""

    def test_code_prompt_with_metadata(self):
        project = projects.create("test-project")
        metadata = {"category": "greeting", "priority": "high"}

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
            metadata=metadata,
        )

        assert prompt.metadata == metadata
        assert prompt.name == "test-prompt"

    def test_code_prompt_without_metadata(self):
        project = projects.create("test-project")

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
        )

        assert prompt.metadata is None

    def test_code_prompt_to_function_definition_includes_metadata(self, mock_project_ids):
        project = projects.create("test-project")
        metadata = {"version": "2.0", "tag": "production"}

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
            metadata=metadata,
        )

        func_def = prompt.to_function_definition(None, mock_project_ids)

        assert func_def["metadata"] == metadata
        assert func_def["name"] == "test-prompt"
        assert func_def["project_id"] == "project-123"

    def test_code_prompt_to_function_definition_excludes_metadata_when_none(self, mock_project_ids):
        project = projects.create("test-project")

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
        )

        func_def = prompt.to_function_definition(None, mock_project_ids)

        assert "metadata" not in func_def


class TestScorerMetadata:
    """Tests for Scorer metadata support."""

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
    def test_code_scorer_with_metadata(self):
        from pydantic import BaseModel

        class ScorerInput(BaseModel):
            output: str
            expected: str

        project = projects.create("test-project")
        metadata = {"type": "accuracy", "version": "1.0"}

        def my_scorer(output: str, expected: str) -> float:
            return 1.0 if output == expected else 0.0

        scorer = project.scorers.create(
            handler=my_scorer,
            name="test-scorer",
            parameters=ScorerInput,
            metadata=metadata,
        )

        assert scorer.metadata == metadata
        assert scorer.name == "test-scorer"

    def test_llm_scorer_with_metadata(self):
        project = projects.create("test-project")
        metadata = {"type": "llm_classifier", "version": "2.0"}

        scorer = project.scorers.create(
            name="llm-scorer",
            prompt="Is this correct?",
            model="gpt-4",
            use_cot=True,
            choice_scores={"yes": 1.0, "no": 0.0},
            metadata=metadata,
        )

        assert scorer.metadata == metadata
        assert scorer.name == "llm-scorer"


class TestCodeFunctionTags:
    """Tests for CodeFunction tags support."""

    def test_code_function_with_tags(self):
        project = projects.create("test-project")
        tags = ["production", "v1"]

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=None,
            tags=tags,
        )

        assert tool.tags == tags

    def test_code_function_without_tags(self):
        project = projects.create("test-project")

        tool = project.tools.create(
            handler=lambda x: x,
            name="test-tool",
            parameters=None,
        )

        assert tool.tags is None


class TestCodePromptTags:
    """Tests for CodePrompt tags support."""

    def test_code_prompt_with_tags(self):
        project = projects.create("test-project")
        tags = ["greeting", "v2"]

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
            tags=tags,
        )

        assert prompt.tags == tags

    def test_code_prompt_without_tags(self):
        project = projects.create("test-project")

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
        )

        assert prompt.tags is None

    def test_code_prompt_to_function_definition_includes_tags(self, mock_project_ids):
        project = projects.create("test-project")
        tags = ["production", "scorer"]

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
            tags=tags,
        )

        func_def = prompt.to_function_definition(None, mock_project_ids)

        assert func_def["tags"] == ["production", "scorer"]

    def test_code_prompt_to_function_definition_excludes_tags_when_none(self, mock_project_ids):
        project = projects.create("test-project")

        prompt = project.prompts.create(
            name="test-prompt",
            prompt="Hello {{name}}",
            model="gpt-4",
        )

        func_def = prompt.to_function_definition(None, mock_project_ids)

        assert "tags" not in func_def


class TestScorerTags:
    """Tests for Scorer tags support."""

    @pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
    def test_code_scorer_with_tags(self):
        from pydantic import BaseModel

        class ScorerInput(BaseModel):
            output: str
            expected: str

        project = projects.create("test-project")
        tags = ["accuracy", "v1"]

        def my_scorer(output: str, expected: str) -> float:
            return 1.0 if output == expected else 0.0

        scorer = project.scorers.create(
            handler=my_scorer,
            name="test-scorer",
            parameters=ScorerInput,
            tags=tags,
        )

        assert scorer.tags == tags

    def test_llm_scorer_with_tags(self):
        project = projects.create("test-project")
        tags = ["classifier", "v2"]

        scorer = project.scorers.create(
            name="llm-scorer",
            prompt="Is this correct?",
            model="gpt-4",
            use_cot=True,
            choice_scores={"yes": 1.0, "no": 0.0},
            tags=tags,
        )

        assert scorer.tags == tags
