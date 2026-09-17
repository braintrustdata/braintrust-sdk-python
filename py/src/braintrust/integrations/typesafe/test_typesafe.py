import os

import pytest
from braintrust import logger
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
    assert span["span_attributes"]["type"] == SpanTypeAttribute.QUESTION
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
