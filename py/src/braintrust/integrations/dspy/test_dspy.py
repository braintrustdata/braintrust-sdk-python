"""
Tests for DSPy integration with Braintrust.
"""

import base64
import inspect
import os
from types import SimpleNamespace

import dspy
import pytest
from braintrust import Attachment, logger
from braintrust.bt_json import bt_dumps
from braintrust.integrations.dspy import BraintrustDSpyCallback
from braintrust.integrations.dspy.tracing import _process_images
from braintrust.integrations.litellm import patch_litellm
from braintrust.integrations.test_utils import run_in_subprocess, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger
from pydantic import BaseModel, Field, field_serializer


PROJECT_NAME = "test-dspy-app"
MODEL = "openai/gpt-4o-mini"
# DSPy >= 3.4 defaults to engine="auto", which routes OpenAI calls through the
# vendored lm15 raw-socket transport that VCR cannot intercept. Pin LiteLLM so
# the request is recorded.
LM_KWARGS = {"engine": "litellm"} if "engine" in inspect.signature(dspy.LM.__init__).parameters else {}


def test_dspy_image_conversion_preserves_other_model_serializers():
    class Page(BaseModel):
        image: dspy.Image
        number: int
        secret: str = Field(exclude=True)

        @field_serializer("number")
        def serialize_number(self, number):
            return f"page-{number}"

    image = dspy.Image(url="data:image/png;base64,aW1hZ2U=")
    page = Page(image=image, number=2, secret="private")
    converted = _process_images(page)
    assert isinstance(converted["image"]["image_url"]["url"], Attachment)
    assert converted["number"] == "page-2"
    assert "secret" not in converted
    assert page.image is image


@pytest.mark.parametrize("url", ["https://example.com/page.png", "data:image/png;base64,not base64!"])
def test_dspy_image_conversion_preserves_unmaterialized_images(url):
    image = dspy.Image(url=url)
    assert _process_images(image) is image
    assert _process_images({"image": image, "url": url}) == {"image": image, "url": url}


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_dspy_images_materialize_attachments(memory_logger):
    """Images stay out of inline logs throughout a real DSPy/LiteLLM call."""
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image_url = f"data:image/png;base64,{image_base64}"

    class Page(BaseModel):
        number: int
        image: dspy.Image

    # DSPy 2.6 supports direct image fields, but does not expand images nested
    # in Pydantic fields into message content parts.
    nested_images = os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "2.6.0"
    describe = dspy.Signature(
        {
            "image": (dspy.Image, dspy.InputField()),
            "page": (Page if nested_images else dspy.Image, dspy.InputField()),
            "answer": (str, dspy.OutputField()),
        },
        instructions="Describe the colors in these two page images.",
    )

    class DescribePages(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(describe)

        def forward(self, image, page):
            result = self.predict(image=image, page=page if nested_images else page.image)
            return dspy.Prediction(answer=result.answer, image=image, page=page)

    patch_litellm()
    lm = dspy.LM(MODEL, cache=False, **LM_KWARGS)
    image = dspy.Image(url=image_url)
    page = Page(number=2, image=dspy.Image(url=image_url))
    with dspy.context(lm=lm, callbacks=[BraintrustDSpyCallback()]):
        result = DescribePages()(image=image, page=page)

    assert result.answer
    assert result.image is image
    assert result.page is page
    assert image.url == page.image.url == image_url
    spans = {span["span_attributes"]["name"]: span for span in memory_logger.pop()}

    for name in ("dspy.module.DescribePages", "dspy.module.Predict"):
        span = spans[name]
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["module_class"].endswith(name.removeprefix("dspy.module."))

    lm_span = spans["dspy.lm"]
    assert lm_span["context"]["span_origin"]["instrumentation"]["name"] == "dspy-auto"
    assert lm_span["span_attributes"].get("type") != "llm"
    assert lm_span["metadata"]["model"] == MODEL
    assert lm_span["metadata"]["provider"] == "openai"
    assert lm_span["output"]
    metrics = lm_span["metrics"]
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    assert metrics["tokens"] == metrics["prompt_tokens"] + metrics["completion_tokens"]

    for name in ("dspy.adapter.format", "dspy.adapter.parse"):
        span = spans[name]
        assert span["span_attributes"]["type"] == "task"
        assert span["metadata"]["adapter_class"].endswith("ChatAdapter")
        assert "signature" in span["input"]
    assert "demos" in spans["dspy.adapter.format"]["input"]
    assert "completion" in spans["dspy.adapter.parse"]["input"]
    assert isinstance(spans["dspy.adapter.parse"]["output"], dict)
    span_ids = {span["span_id"] for span in spans.values()}
    for name in ("dspy.lm", "dspy.adapter.format", "dspy.adapter.parse"):
        assert spans[name]["span_parents"][0] in span_ids

    def assert_image(value):
        attachment = value["image_url"]["url"]
        assert isinstance(attachment, Attachment)
        assert attachment.data == base64.b64decode(image_base64)
        assert attachment.reference["content_type"] == "image/png"

    for name in ("dspy.module.DescribePages", "dspy.module.Predict"):
        inputs = spans[name]["input"]
        inputs = inputs.get("kwargs", inputs)
        assert_image(inputs["image"])
        if nested_images or name == "dspy.module.DescribePages":
            assert_image(inputs["page"]["image"])
            assert inputs["page"]["number"] == 2
        else:
            assert_image(inputs["page"])
    assert_image(spans["dspy.module.DescribePages"]["output"]["image"])
    assert_image(spans["dspy.module.DescribePages"]["output"]["page"]["image"])
    assert_image(spans["dspy.adapter.format"]["input"]["inputs"]["image"])
    logged_page = spans["dspy.adapter.format"]["input"]["inputs"]["page"]
    assert_image(logged_page["image"] if nested_images else logged_page)
    for messages in (
        spans["dspy.adapter.format"]["output"],
        spans["dspy.lm"]["input"]["messages"],
        spans["Completion"]["input"],
    ):
        image_parts = [
            part
            for message in messages
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        assert len(image_parts) == 2
        for part in image_parts:
            assert_image(part)
    serialized = bt_dumps(spans)
    assert image_base64 not in serialized
    assert "CUSTOM-TYPE" not in serialized


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def test_dspy_callback_ignores_usage_from_previous_lm_call(memory_logger):
    """A failed call must not inherit usage from an older history entry."""
    instance = SimpleNamespace(
        model=MODEL,
        history=[
            {
                "prompt": "previous request",
                "outputs": ["previous response"],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            }
        ],
    )
    callback = BraintrustDSpyCallback()
    callback.on_lm_start("failed-call", instance, {"prompt": "new request"})
    callback.on_lm_end("failed-call", None, RuntimeError("provider failed"))

    span = next(span for span in memory_logger.pop() if span["span_attributes"]["name"] == "dspy.lm")
    assert not {"prompt_tokens", "completion_tokens", "tokens"} & span.get("metrics", {}).keys()


def test_dspy_callback_correlates_concurrent_lm_usage(memory_logger):
    """Concurrent calls on one LM must read their own new history entries."""
    instance = SimpleNamespace(model=MODEL, history=[])
    callback = BraintrustDSpyCallback()
    callback.on_lm_start("call-a", instance, {"prompt": "same request"})
    callback.on_lm_start("call-b", instance, {"prompt": "same request"})

    outputs_a = ["same response"]
    outputs_b = ["same response"]
    entry_a = {
        "prompt": "same request",
        "outputs": outputs_a,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    entry_b = {
        "prompt": "same request",
        "outputs": outputs_b,
        "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
    }
    # Both provider calls can append history before either end callback runs.
    instance.history.extend([entry_a, entry_b])
    callback.on_lm_end("call-a", outputs_a)
    callback.on_lm_end("call-b", outputs_b)

    spans = [span for span in memory_logger.pop() if span["span_attributes"]["name"] == "dspy.lm"]
    assert [span["metrics"]["tokens"] for span in spans] == [12, 35]


def test_dspy_adapter_callbacks(memory_logger):
    """Adapter format/parse callbacks should log spans without an LM call."""
    assert not memory_logger.pop()

    dspy.configure(callbacks=[BraintrustDSpyCallback()])

    signature = dspy.make_signature("question -> answer")
    adapter = dspy.ChatAdapter()
    formatted = adapter.format(
        signature,
        demos=[{"question": "1+1", "answer": "2"}],
        inputs={"question": "2+2"},
    )
    parsed = adapter.parse(signature, "[[ ## answer ## ]]\n4")

    assert formatted
    assert parsed == {"answer": "4"}

    spans = memory_logger.pop()
    assert len(spans) == 2

    spans_by_name = {span["span_attributes"]["name"]: span for span in spans}
    format_span = spans_by_name["dspy.adapter.format"]
    parse_span = spans_by_name["dspy.adapter.parse"]

    assert format_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert format_span["input"]["inputs"] == {"question": "2+2"}
    assert format_span["output"] == formatted

    assert parse_span["metadata"]["adapter_class"].endswith("ChatAdapter")
    assert parse_span["input"]["completion"] == "[[ ## answer ## ]]\n4"
    assert parse_span["output"] == parsed


class TestPatchDSPy:
    """Tests for patch_dspy()."""

    def test_patch_dspy_adds_exactly_one_callback(self):
        """After patch_dspy(), dspy.configure() should add exactly one BraintrustDSpyCallback."""
        result = run_in_subprocess("""
            from braintrust.integrations.dspy import patch_dspy, BraintrustDSpyCallback
            assert patch_dspy(), "patch_dspy() should return True"
            assert patch_dspy(), "second patch_dspy() should be a no-op that still returns True"

            import dspy
            from dspy.dsp.utils.settings import settings
            from dspy.utils.callback import BaseCallback

            def bt_callbacks():
                return [cb for cb in settings.callbacks if isinstance(cb, BraintrustDSpyCallback)]

            # Configure without explicitly adding callback (no double-wrap)
            dspy.configure(lm=None)
            assert len(bt_callbacks()) == 1, f"Expected one BraintrustDSpyCallback in {settings.callbacks}"

            # User-provided callbacks are preserved
            class MyCallback(BaseCallback):
                pass

            my_callback = MyCallback()
            dspy.configure(lm=None, callbacks=[my_callback])
            assert any(cb is my_callback for cb in settings.callbacks), "User callback should be preserved"
            assert len(bt_callbacks()) == 1, f"Expected one BraintrustDSpyCallback in {settings.callbacks}"

            # An explicit BraintrustDSpyCallback is not duplicated
            bt_callback = BraintrustDSpyCallback()
            dspy.configure(lm=None, callbacks=[bt_callback])
            assert bt_callbacks() == [bt_callback], f"Expected only the user's callback in {settings.callbacks}"
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_legacy_wrapper_import_still_works(self):
        """The old braintrust.wrappers.dspy import path should still work."""
        from braintrust.wrappers.dspy import BraintrustDSpyCallback as LegacyBraintrustDSpyCallback
        from braintrust.wrappers.dspy import patch_dspy

        assert LegacyBraintrustDSpyCallback is BraintrustDSpyCallback
        assert callable(patch_dspy)


class TestAutoInstrumentDSPy:
    """Tests for auto_instrument() with DSPy."""

    def test_auto_instrument_dspy(self):
        """Test auto_instrument patches DSPy, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_dspy.py")
