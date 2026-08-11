"""Patch supported local Hugging Face Transformers pipeline classes."""

from typing import TypeVar

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import (
    _feature_extraction_pipeline_call_wrapper,
    _question_answering_pipeline_call_wrapper,
    _summarization_pipeline_call_wrapper,
    _text2text_generation_pipeline_call_wrapper,
    _text_generation_pipeline_call_wrapper,
    _translation_pipeline_call_wrapper,
)


class SummarizationPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.summarization"
    target_module = "transformers.pipelines.text2text_generation"
    target_path = "SummarizationPipeline.__call__"
    wrapper = _summarization_pipeline_call_wrapper


class TranslationPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.translation"
    target_module = "transformers.pipelines.text2text_generation"
    target_path = "TranslationPipeline.__call__"
    wrapper = _translation_pipeline_call_wrapper


class Text2TextGenerationPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.text2text_generation"
    target_module = "transformers.pipelines.text2text_generation"
    target_path = "Text2TextGenerationPipeline.__call__"
    wrapper = _text2text_generation_pipeline_call_wrapper


class TextGenerationPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.text_generation"
    target_module = "transformers.pipelines.text_generation"
    target_path = "TextGenerationPipeline.__call__"
    wrapper = _text_generation_pipeline_call_wrapper


class FeatureExtractionPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.feature_extraction"
    target_module = "transformers.pipelines.feature_extraction"
    target_path = "FeatureExtractionPipeline.__call__"
    wrapper = _feature_extraction_pipeline_call_wrapper


class QuestionAnsweringPipelinePatcher(FunctionWrapperPatcher):
    name = "transformers.pipeline.question_answering"
    target_module = "transformers.pipelines.question_answering"
    target_path = "QuestionAnsweringPipeline.__call__"
    wrapper = _question_answering_pipeline_call_wrapper


# Keep the text2text subclasses next to their parent. Each class wrapper only
# traces its own task, so subclass calls to ``super().__call__`` pass through
# the parent wrapper without creating another span.
PIPELINE_PATCHERS = (
    SummarizationPipelinePatcher,
    TranslationPipelinePatcher,
    Text2TextGenerationPipelinePatcher,
    TextGenerationPipelinePatcher,
    FeatureExtractionPipelinePatcher,
    QuestionAnsweringPipelinePatcher,
)


_T = TypeVar("_T")


def wrap_transformers(target: _T) -> _T:
    """Manually instrument one supported pipeline instance or class.

    Unsupported objects are returned unchanged. Patching is idempotent and is
    applied to the pipeline class because Python resolves ``obj()`` through the
    class-level ``__call__`` special method.
    """
    target_class = target if isinstance(target, type) else type(target)
    for patcher in PIPELINE_PATCHERS:
        root = patcher.resolve_root(None, None)
        if root is None:
            continue
        class_name = patcher.target_path.split(".", 1)[0]
        pipeline_class = getattr(root, class_name, None)
        if isinstance(pipeline_class, type) and issubclass(target_class, pipeline_class):
            patcher.wrap_target(target_class)
            break
    return target
