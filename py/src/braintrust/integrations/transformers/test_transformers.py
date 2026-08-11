"""Tests for local Hugging Face Transformers pipeline instrumentation."""

# Keep the large Transformers/PyTorch dependencies isolated to their nox job.
# pylint: disable=import-error

import inspect

import pytest
from braintrust import logger
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.integrations.transformers import TransformersIntegration, setup_transformers, wrap_transformers
from braintrust.integrations.transformers.patchers import PIPELINE_PATCHERS
from braintrust.integrations.transformers.tracing import _input, _metadata
from braintrust.integrations.utils import _tensor_shape
from braintrust.test_helpers import init_test_logger


transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from transformers import pipeline, set_seed  # noqa: E402
from transformers.pipelines import FeatureExtractionPipeline, TextGenerationPipeline  # noqa: E402


# Transformers v5 removed the text2text, summarization, translation, and
# extractive-QA pipeline APIs. Keep the v5 matrix coverage for the surviving
# supported classes while the v4 floor exercises all six task families.
Text2TextGenerationPipeline = getattr(transformers.pipelines, "Text2TextGenerationPipeline", None)
SummarizationPipeline = getattr(transformers.pipelines, "SummarizationPipeline", None)
TranslationPipeline = getattr(transformers.pipelines, "TranslationPipeline", None)
QuestionAnsweringPipeline = getattr(transformers.pipelines, "QuestionAnsweringPipeline", None)


PROJECT_NAME = "test-transformers-sdk"
TEXT_GENERATION_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEXT2TEXT_MODEL = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
FEATURE_EXTRACTION_MODEL = "hf-internal-testing/tiny-random-BertModel"
QUESTION_ANSWERING_MODEL = "hf-internal-testing/tiny-random-BertForQuestionAnswering"
CLASSIFICATION_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"

PIPELINE_CLASSES = tuple(
    pipeline_class
    for pipeline_class in (
        TextGenerationPipeline,
        SummarizationPipeline,
        TranslationPipeline,
        Text2TextGenerationPipeline,
        FeatureExtractionPipeline,
        QuestionAnsweringPipeline,
    )
    if isinstance(pipeline_class, type)
)


@pytest.fixture(autouse=True)
def clean_pipeline_methods():
    """Restore globally patched pipeline classes after every test."""
    originals = [
        (pipeline_class, inspect.getattr_static(pipeline_class, "__call__")) for pipeline_class in PIPELINE_CLASSES
    ]
    try:
        yield
    finally:
        for pipeline_class, original in originals:
            setattr(pipeline_class, "__call__", original)
        for patcher in PIPELINE_PATCHERS:
            marker = patcher.patch_marker_attr()
            root = patcher.resolve_root(None, None)
            if root is not None and marker in vars(root):
                delattr(root, marker)
            for pipeline_class, original in originals:
                if marker in pipeline_class.__dict__:
                    delattr(pipeline_class, marker)
                if hasattr(original, marker):
                    delattr(original, marker)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as background_logger:
        background_logger.pop()
        yield background_logger


@pytest.fixture(scope="module")
def text_generation_pipeline():
    return pipeline("text-generation", model=TEXT_GENERATION_MODEL, device=-1)


@pytest.fixture(scope="module")
def text2text_pipeline():
    if Text2TextGenerationPipeline is None:
        pytest.skip("Transformers v5 removed text2text-generation pipelines")
    return pipeline("text2text-generation", model=TEXT2TEXT_MODEL, device=-1)


@pytest.fixture(scope="module")
def summarization_pipeline():
    if SummarizationPipeline is None:
        pytest.skip("Transformers v5 removed summarization pipelines")
    return pipeline("summarization", model=TEXT2TEXT_MODEL, device=-1)


@pytest.fixture(scope="module")
def translation_pipeline():
    if TranslationPipeline is None:
        pytest.skip("Transformers v5 removed translation pipelines")
    return pipeline("translation", model=TEXT2TEXT_MODEL, device=-1)


@pytest.fixture(scope="module")
def language_translation_pipeline():
    if TranslationPipeline is None:
        pytest.skip("Transformers v5 removed translation pipelines")
    return pipeline("translation_en_to_fr", model=TEXT2TEXT_MODEL, device=-1)


@pytest.fixture(scope="module")
def feature_extraction_pipeline():
    return pipeline("feature-extraction", model=FEATURE_EXTRACTION_MODEL, device=-1)


@pytest.fixture(scope="module")
def question_answering_pipeline():
    if QuestionAnsweringPipeline is None:
        pytest.skip("Transformers v5 removed question-answering pipelines")
    return pipeline("question-answering", model=QUESTION_ANSWERING_MODEL, device=-1)


@pytest.fixture(scope="module")
def classification_pipeline():
    return pipeline("text-classification", model=CLASSIFICATION_MODEL, device=-1)


@pytest.fixture(autouse=True)
def fixed_seed():
    set_seed(0)


def _only_span(memory_logger):
    spans = memory_logger.pop()
    assert len(spans) == 1
    return spans[0]


def _assert_common_span(span, task, pipeline_instance):
    assert span["span_attributes"]["type"] == "llm"
    assert span["span_attributes"]["name"] == f"huggingface.transformers.{task.replace('-', '_')}"
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "transformers-auto"
    assert span["metadata"]["provider"] == "huggingface"
    assert span["metadata"]["model"] == pipeline_instance.model.config._name_or_path
    assert span["metadata"]["task"] == task
    assert not {"tokens", "prompt_tokens", "completion_tokens"} & span.get("metrics", {}).keys()
    for choice in span.get("output", []):
        assert "finish_reason" not in choice


def test_integration_targets_only_supported_pipeline_classes():
    assert TransformersIntegration.min_version == "4.42.0"
    assert {patcher.target_path for patcher in PIPELINE_PATCHERS} == {
        "TextGenerationPipeline.__call__",
        "Text2TextGenerationPipeline.__call__",
        "SummarizationPipeline.__call__",
        "TranslationPipeline.__call__",
        "FeatureExtractionPipeline.__call__",
        "QuestionAnsweringPipeline.__call__",
    }
    assert all(patcher.target_path != "Pipeline.__call__" for patcher in PIPELINE_PATCHERS)


def test_setup_is_idempotent(text_generation_pipeline, memory_logger):
    assert setup_transformers() is True
    assert setup_transformers() is True

    result = text_generation_pipeline("Hello", do_sample=False, max_new_tokens=1)
    assert result
    _assert_common_span(_only_span(memory_logger), "text-generation", text_generation_pipeline)


def test_manual_wrap_is_idempotent_and_traces_string_input(text_generation_pipeline, memory_logger):
    assert wrap_transformers(text_generation_pipeline) is text_generation_pipeline
    assert wrap_transformers(text_generation_pipeline) is text_generation_pipeline

    result = text_generation_pipeline(
        "Hello",
        do_sample=False,
        max_new_tokens=2,
        top_k=7,
        repetition_penalty=1.0,
    )
    span = _only_span(memory_logger)

    _assert_common_span(span, "text-generation", text_generation_pipeline)
    assert span["input"] == [{"role": "user", "content": "Hello"}]
    assert [choice["index"] for choice in span["output"]] == list(range(len(result)))
    assert span["metadata"]["max_tokens"] == 2
    assert span["metadata"]["do_sample"] is False
    assert span["metadata"]["top_k"] == 7
    assert span["metadata"]["repetition_penalty"] == 1.0
    # The integration passes runtime values to Braintrust unchanged; the
    # normal logger serialization boundary renders torch values as strings.
    assert span["metadata"]["device"] == str(text_generation_pipeline.device)
    assert span["metadata"]["torch_dtype"] == str(text_generation_pipeline.model.dtype)


def test_text_generation_chat_input_passes_through(text_generation_pipeline, memory_logger):
    setup_transformers()
    chat = [{"role": "user", "content": "Hello"}]

    result = text_generation_pipeline(chat, do_sample=False, max_new_tokens=1)
    span = _only_span(memory_logger)

    assert result
    _assert_common_span(span, "text-generation", text_generation_pipeline)
    assert span["input"] == chat
    assert span["output"][0]["message"]["content"] == result[0]["generated_text"][-1]["content"]


def test_text_generation_batch_is_one_span(text_generation_pipeline, memory_logger):
    setup_transformers()
    inputs = ["First", "Second"]

    result = text_generation_pipeline(inputs, do_sample=False, max_new_tokens=1)
    span = _only_span(memory_logger)

    _assert_common_span(span, "text-generation", text_generation_pipeline)
    assert span["input"] == [
        [{"role": "user", "content": "First"}],
        [{"role": "user", "content": "Second"}],
    ]
    assert len(span["output"]) == sum(len(batch) for batch in result)
    assert [choice["index"] for choice in span["output"]] == list(range(len(span["output"])))


@pytest.mark.parametrize(
    ("fixture_name", "task", "result_field"),
    [
        ("text2text_pipeline", "text2text-generation", "generated_text"),
        ("summarization_pipeline", "summarization", "summary_text"),
        ("translation_pipeline", "translation", "translation_text"),
    ],
)
def test_text2text_task_families(request, memory_logger, fixture_name, task, result_field):
    pipeline_instance = request.getfixturevalue(fixture_name)
    setup_transformers()

    result = pipeline_instance("Hello world", do_sample=False, max_new_tokens=2)
    span = _only_span(memory_logger)

    _assert_common_span(span, task, pipeline_instance)
    assert span["input"] == [{"role": "user", "content": "Hello world"}]
    assert span["output"][0]["message"]["content"] == result[0][result_field]


def test_language_specific_translation_name(language_translation_pipeline, memory_logger):
    setup_transformers()

    result = language_translation_pipeline("Hello", do_sample=False, max_new_tokens=2)
    span = _only_span(memory_logger)

    _assert_common_span(span, "translation_en_to_fr", language_translation_pipeline)
    assert span["output"][0]["message"]["content"] == result[0]["translation_text"]


def test_question_answering_combines_context_and_question(question_answering_pipeline, memory_logger):
    setup_transformers()
    context = "Ada built it."
    question = "Who built it?"

    result = question_answering_pipeline(context=context, question=question)
    span = _only_span(memory_logger)

    _assert_common_span(span, "question-answering", question_answering_pipeline)
    assert span["input"] == [
        {
            "role": "user",
            "content": "Context:\nAda built it.\n\nQuestion:\nWho built it?",
        }
    ]
    assert span["output"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": result["answer"]},
        }
    ]


def test_feature_extraction_logs_only_shape(feature_extraction_pipeline, memory_logger):
    setup_transformers()
    input_text = "Embed this"

    result = feature_extraction_pipeline(input_text, return_tensors=True)
    span = _only_span(memory_logger)

    _assert_common_span(span, "feature-extraction", feature_extraction_pipeline)
    assert span["input"] == input_text
    assert span["output"] == {"shape": list(result.shape)}
    assert str(result.flatten()[0].item()) not in str(span["output"])


def test_collection_helpers_preserve_runtime_values(text_generation_pipeline):
    chat = [{"role": "user", "content": "Hello"}]
    stop = ["END"]

    assert _input("text-generation", (chat,), {}) is chat
    metadata = _metadata(text_generation_pipeline, {"stop": stop}, "text-generation")
    assert metadata["stop"] is stop
    assert metadata["device"] is text_generation_pipeline.device
    assert metadata["torch_dtype"] is text_generation_pipeline.model.dtype


def test_tensor_shape_uses_only_shape_metadata():
    assert _tensor_shape(torch.empty(())) == []
    assert _tensor_shape(torch.empty(4)) == [4]
    assert _tensor_shape(torch.empty(3, 4)) == [3, 4]
    assert _tensor_shape(torch.empty(2, 3, 4)) == [2, 3, 4]


def test_real_pipeline_error_is_logged_and_propagated(text_generation_pipeline, memory_logger):
    setup_transformers()

    with pytest.raises(Exception) as raised:
        text_generation_pipeline(object(), do_sample=False, max_new_tokens=1)

    assert raised.value
    span = _only_span(memory_logger)
    assert span.get("error")
    assert "output" not in span


def test_unsupported_pipeline_produces_no_span(classification_pipeline, memory_logger):
    setup_transformers()

    result = classification_pipeline("Hello")

    assert result
    assert memory_logger.pop() == []


def test_streamer_call_produces_no_span(text_generation_pipeline, memory_logger, capsys):
    setup_transformers()
    streamer = transformers.TextStreamer(text_generation_pipeline.tokenizer, skip_prompt=True)

    result = text_generation_pipeline("Hello", do_sample=False, max_new_tokens=1, streamer=streamer)

    assert result
    assert memory_logger.pop() == []
    capsys.readouterr()


def test_auto_instrument_transformers():
    verify_autoinstrument_script("test_auto_transformers.py", timeout=120)
