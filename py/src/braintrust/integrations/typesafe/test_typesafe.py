import os
from pathlib import Path

import pytest
import typesafe_sdk
from braintrust import Attachment, logger
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.integrations.typesafe import setup_typesafe, wrap_typesafe
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score, TypeSafeClient, TypeSafeError


PROJECT_NAME = "test-typesafe-sdk"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _assert_span(span, *, question_ids):
    assert span["span_attributes"]["name"] == "typesafe.systemOne"
    assert span["span_attributes"]["type"] == SpanTypeAttribute.DECISION
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "typesafe"
    assert span["metadata"]["provider"] == "typesafe"
    assert span["metadata"]["model"].startswith("jev-")
    assert [question["id"] for question in span["input"]["questions"]] == question_ids
    assert [answer["id"] for answer in span["output"]["answers"]] == question_ids
    assert span["metrics"]["tokens"] == (span["metrics"]["prompt_tokens"] + span["metrics"]["completion_tokens"])
    assert_metrics_are_valid(span["metrics"])


@pytest.mark.vcr
def test_wrap_typesafe_system_one_sync(memory_logger):
    with (
        wrap_typesafe(TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"])) as client,
        TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as unwrapped_client,
    ):
        response = client.system_one(
            state="The package arrived intact and on time.",
            questions={
                "positive": {
                    "type": "noul",
                    "instructions": "Is this feedback positive?",
                    "not_allowlisted": "secret",
                }
            },
        )
        spans = memory_logger.pop()

        with pytest.raises(TypeSafeError, match="At least one question is required") as raised:
            client.system_one(state="hello", questions={})
        error_spans = memory_logger.pop()

        with pytest.raises(TypeSafeError, match="At least one question is required"):
            unwrapped_client.system_one(state="hello", questions={})
        unwrapped_spans = memory_logger.pop()

    assert 0 <= response.nouls["positive"].noul <= 1
    assert len(spans) == 1
    _assert_span(spans[0], question_ids=["positive"])
    assert spans[0]["input"] == {
        "state": "The package arrived intact and on time.",
        "questions": [
            {
                "id": "positive",
                "type": "noul",
                "instructions": "Is this feedback positive?",
            }
        ],
    }
    assert "not_allowlisted" not in spans[0]["input"]["questions"][0]
    assert spans[0]["output"]["answers"][0]["type"] == "noul"
    assert len(error_spans) == 1
    assert "At least one question is required" in error_spans[0]["error"]
    assert raised.value.__class__.__module__.startswith("typesafe_sdk")
    assert unwrapped_spans == []


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_setup_typesafe_system_one_async(memory_logger):
    async with (
        wrap_typesafe(AsyncTypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"])) as client,
        AsyncTypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as unwrapped_client,
    ):
        response = await client.system_one(
            state={"message": "I was charged twice. Please refund the duplicate charge today."},
            questions={
                "category": Choice(
                    instructions="Which team should handle this?",
                    criteria={"billing": None, "technical": None, "other": None},
                ),
                "urgency": Score(
                    instructions="How urgent is this request?",
                    criteria=["routine", "soon", "urgent"],
                ),
                "duplicate_charge": Noul(instructions={"question": "Does the customer report a duplicate charge?"}),
            },
        )
        spans = memory_logger.pop()

        with pytest.raises(TypeSafeError, match="At least one question is required"):
            await unwrapped_client.system_one(state="hello", questions={})
        assert memory_logger.pop() == []

        assert setup_typesafe()
        assert setup_typesafe()

        with pytest.raises(TypeSafeError, match="At least one question is required"):
            await unwrapped_client.system_one(state="hello", questions={})
        setup_spans = memory_logger.pop()

    assert set(response.answers) == {"category", "urgency", "duplicate_charge"}
    assert len(spans) == 1
    _assert_span(spans[0], question_ids=["category", "urgency", "duplicate_charge"])
    assert [question["type"] for question in spans[0]["input"]["questions"]] == [
        "choice",
        "score",
        "noul",
    ]
    assert len(setup_spans) == 1
    assert "At least one question is required" in setup_spans[0]["error"]


def test_auto_instrument_typesafe_subprocess():
    verify_autoinstrument_script("test_auto_typesafe.py")


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.vcr
async def test_typesafe_boxes_and_images(memory_logger, is_async):
    if not hasattr(typesafe_sdk, "Box"):
        pytest.skip("Box and Image require TypeSafe SDK >=0.7.5")

    # TypeSafe SDK <0.7 uses msgspec and does not install pydantic.
    from pydantic import BaseModel

    class CustomImage(typesafe_sdk.Image):
        pass

    class CustomAnswers(BaseModel):
        bounds: typesafe_sdk.BoxAnswer
        present: typesafe_sdk.NoulAnswer
        category: typesafe_sdk.ChoiceAnswer
        size: typesafe_sdk.ScoreAnswer

    class CustomResponse(BaseModel):
        model: str
        usage: typesafe_sdk.Usage
        answers: CustomAnswers

    image_file = Path(__file__).parents[2] / "fixtures" / "decision-box.png"
    image = CustomImage(image_file)
    state = {"photos/~": [image], "description": "A test image"}
    questions = {
        "bounds": typesafe_sdk.Box(instructions="The main object in `photos/~[0]`"),
        "present": Noul(instructions="Is there a main object in the image?"),
        "category": Choice(instructions="What is in the image?", criteria={"object": None, "text": None}),
        "size": Score(instructions="How much of the image is occupied?", criteria=["little", "most"]),
    }
    assert setup_typesafe()
    client_type = AsyncTypeSafeClient if is_async else TypeSafeClient
    client = client_type(api_key=os.environ["TYPESAFE_API_KEY"], compression=False)
    kwargs = dict(
        model="jev-1.14.0",
        state=state,
        questions=questions,
        extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
    )
    if is_async:
        kwargs["response_model"] = CustomResponse
        async with client:
            response = await client.system_one(**kwargs)
        assert isinstance(response, CustomResponse)
        assert isinstance(response.answers, CustomAnswers)
        response_answers = {identifier: getattr(response.answers, identifier) for identifier in questions}
    else:
        with client:
            response = client.system_one(**kwargs)
        response_answers = response.answers

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    _assert_span(span, question_ids=list(questions))
    assert span["metadata"]["model"] == "jev-1.14.0"
    attachment = span["input"]["state"]["photos/~"][0]
    assert isinstance(attachment, Attachment)
    assert attachment.data == image_file.read_bytes()
    assert attachment.reference["content_type"] == "image/png"
    assert state["photos/~"][0] is image
    assert span["input"]["questions"][0] == {
        "id": "bounds",
        "type": "box",
        "instructions": questions["bounds"].instructions,
    }
    answers = {answer["id"]: answer for answer in span["output"]["answers"]}
    assert answers["bounds"] == {
        "id": "bounds",
        "type": "box",
        "box": response_answers["bounds"].box.model_dump(),
        "confidence": response_answers["bounds"].confidence,
        "image_path": "/state/photos~1~0/0",
    }
    assert answers["present"]["noul"] == response_answers["present"].noul
    assert answers["size"]["score"] == response_answers["size"].score
    assert "scores" not in span
    assert "classifications" not in span


@pytest.mark.vcr
def test_typesafe_nested_image_assets(memory_logger):
    if not hasattr(typesafe_sdk, "Box"):
        pytest.skip("Box and Image require TypeSafe SDK >=0.7.5")

    image_file = Path(__file__).parents[2] / "fixtures" / "decision-box.png"
    image = typesafe_sdk.Image(image_file)
    state = {"photos": ["image", "inspect image_2"]}
    questions = {
        # IDs are text, even when they happen to match an asset reference.
        "image_2": typesafe_sdk.Box(instructions={"image": image, "target": "The black square"}),
        "match": Choice(instructions="Choose the matching image", criteria={"square": image, "none": None}),
    }
    assert setup_typesafe()
    with TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], compression=False) as client:
        response = client.system_one(
            model="jev-1.14.0",
            state=state,
            questions=questions,
            assets={"image": image, "image_2": image},
            extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
        )

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    _assert_span(span, question_ids=list(questions))
    captured = span["input"]
    photos = captured["state"]["photos"]
    assert photos[1] == ["inspect ", photos[0]]
    images = [
        photos[0],
        photos[1][1],
        captured["questions"][0]["instructions"]["image"],
        captured["questions"][1]["criteria"]["square"],
    ]
    assert all(isinstance(item, Attachment) and item.data == image_file.read_bytes() for item in images)
    answers = {answer["id"]: answer for answer in span["output"]["answers"]}
    assert answers["image_2"]["box"] == response.answers["image_2"].box.model_dump()
    # Multiple image occurrences have no structured selector in TypeSafe;
    # instruction text must not be used to guess their association.
    assert "image_path" not in answers["image_2"]
    assert state == {"photos": ["image", "inspect image_2"]}
    assert questions["image_2"].instructions == {"image": image, "target": "The black square"}
