import base64
from pathlib import Path

import pytest
from braintrust import Attachment
from braintrust.integrations.litellm.tracing import _parse_metrics_from_usage as parse_litellm_usage
from braintrust.integrations.openai.tracing import _decisions_input, _decisions_output
from braintrust.integrations.openai.tracing import _parse_metrics_from_usage as parse_openai_usage
from braintrust.integrations.question import _question_span_input
from braintrust.integrations.typesafe.tracing import _image_payload
from braintrust.test_helpers import assert_dict_matches


DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_INPUT = "The customer says: I love this product!"
DECISIONS_QUESTIONS = [
    {"type": "predicate", "name": "positive", "instructions": "Is the sentiment positive?"},
    {
        "type": "choice",
        "name": "sentiment",
        "instructions": "What is the sentiment?",
        "choices": [{"value": "positive", "description": "Happy feedback"}, {"value": "negative"}],
    },
    {
        "type": "score",
        "name": "enthusiasm",
        "instructions": "How enthusiastic is the customer?",
        "levels": [{"label": "low"}, {"label": "high", "description": "Very enthusiastic"}],
    },
    {"type": "predicate", "instructions": "Does the customer like the product?"},
]


def decisions_image_input():
    image = (Path(__file__).parent.parent / "fixtures" / "test-image.png").read_bytes()
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": DECISIONS_INPUT},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(image).decode(),
                    "detail": "low",
                },
            ],
        }
    ]


def assert_decisions_image_span(spans, response, messages):
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["type"] == "decision"
    image = span["input"]["state"][0]["content"][1]
    attachment = image["image_url"]["url"]
    assert isinstance(attachment, Attachment)
    original = messages[0]["content"][1]
    assert original["type"] == "input_image"
    assert original["detail"] == "low"
    assert attachment.data == base64.b64decode(original["image_url"].split(",", 1)[1])
    assert attachment.reference["content_type"] == "image/png"
    assert attachment.reference["filename"] == "image.png"
    assert span["input"]["state"] == [
        {
            **messages[0],
            "content": [
                messages[0]["content"][0],
                {"type": "image_url", "image_url": {"url": attachment, "detail": "low"}},
            ],
        }
    ]
    assert span["output"]["answers"] == [
        {"id": "positive", "type": "boolean", "probability": response.answers[0].probability}
    ]


def assert_decisions_span(spans, response, *, name, model, instrumentation, start=None, end=None):
    assert len(spans) == 1
    span = spans[0]
    assert_dict_matches(span["span_attributes"], {"name": name, "type": "decision"})
    assert span["input"] == {
        "state": DECISIONS_INPUT,
        "questions": [
            {"id": "positive", "type": "boolean", "instructions": "Is the sentiment positive?"},
            {
                "id": "sentiment",
                "type": "choice",
                "instructions": "What is the sentiment?",
                "criteria": {"positive": "Happy feedback", "negative": None},
            },
            {
                "id": "enthusiasm",
                "type": "score",
                "instructions": "How enthusiastic is the customer?",
                "criteria": [{"label": "low"}, {"label": "high", "description": "Very enthusiastic"}],
            },
            {"id": "question_3", "type": "boolean", "instructions": "Does the customer like the product?"},
        ],
    }
    assert [answer["id"] for answer in span["output"]["answers"]] == [
        "positive",
        "sentiment",
        "enthusiasm",
        "question_3",
    ]
    assert span["output"]["answers"][0] == {
        "id": "positive",
        "type": "boolean",
        "probability": response.answers[0].probability,
    }
    assert span["output"]["answers"][1]["choice"] == response.answers[1].choice
    assert span["output"]["answers"][1]["probabilities"] == {
        item.value: item.probability for item in response.answers[1].probabilities
    }
    assert span["output"]["answers"][1]["confidence"] == response.answers[1].confidence
    assert span["output"]["answers"][2]["score"] == response.answers[2].score
    assert span["output"]["answers"][2]["confidence"] == response.answers[2].confidence
    assert span["output"]["answers"][2]["probabilities"] == {
        str(item.value): item.probability for item in response.answers[2].probabilities
    }
    assert span["output"]["answers"][2]["legend"] == {
        str(item.value): item.label for item in response.answers[2].probabilities
    }
    assert span["output"]["answers"][3] == {
        "id": "question_3",
        "type": "boolean",
        "probability": response.answers[3].probability,
    }
    assert span["metadata"] == {"model": model, "provider": "openai"}
    assert span["context"]["span_origin"]["instrumentation"]["name"] == instrumentation
    metrics = span["metrics"]
    assert metrics["start"] <= metrics["end"]
    if start is not None and end is not None:
        assert start <= metrics["start"] <= metrics["end"] <= end
    assert all(metrics[key] >= 0 for key in ("tokens", "prompt_tokens", "completion_tokens"))
    assert metrics["tokens"] == response.usage.total_tokens
    assert metrics["prompt_tokens"] == response.usage.input_tokens
    assert metrics["completion_tokens"] == response.usage.output_tokens
    assert metrics["prompt_cached_tokens"] == response.usage.input_tokens_details.cached_tokens
    assert metrics["prompt_cache_creation_tokens"] == response.usage.input_tokens_details.cache_write_tokens
    assert metrics["completion_reasoning_tokens"] == response.usage.output_tokens_details.reasoning_tokens
    assert "prompt_cache_write_tokens" not in metrics


def test_decisions_ids_are_unique_and_match_answers():
    questions = [
        {"type": "predicate", "name": "question_1", "instructions": "First"},
        {"type": "predicate", "instructions": "Second"},
        {"type": "predicate", "name": "question_1", "instructions": "Third"},
    ]
    span_input = _decisions_input("state", questions)
    span_output = _decisions_output([{"type": "predicate", "probability": 0.5}] * 3, questions)
    ids = [question["id"] for question in span_input["questions"]]
    assert ids == ["question_1", "question_1_", "question_2"]
    assert [answer["id"] for answer in span_output["answers"]] == ids


def test_decisions_preserves_typed_choices():
    questions = [
        {
            "type": "choice",
            "name": "typed",
            "instructions": "Choose",
            "choices": [{"value": True}, {"value": "true"}],
        }
    ]
    span_input = _decisions_input("state", questions)
    span_output = _decisions_output(
        [
            {
                "type": "choice",
                "choice": True,
                "probabilities": [{"value": True, "probability": 0.8}, {"value": "true", "probability": 0.2}],
            }
        ],
        questions,
    )
    assert span_input["questions"][0]["criteria"] == {"boolean:true": None, "true": None}
    assert span_output["answers"][0] == {
        "id": "typed",
        "type": "choice",
        "choice": "boolean:true",
        "probabilities": {"boolean:true": 0.8, "true": 0.2},
    }


def test_decisions_refusals_and_allowlists():
    questions = [{"type": "predicate", "name": "refused", "instructions": "Question", "secret": "omit"}]
    span_input = _decisions_input({"message": "state"}, questions)
    span_output = _decisions_output([{"type": "refusal", "name": "refused", "secret": "omit"}], questions)
    assert span_input == {
        "state": {"message": "state"},
        "questions": [{"id": "refused", "type": "boolean", "instructions": "Question"}],
    }
    assert span_output == {"answers": [{"id": "refused", "type": "refusal"}]}


def test_decisions_system_one_maps_match_typesafe_structure():
    questions = {"present": {"type": "noul", "instructions": "Is the target present?", "secret": "omit"}}
    answers = {"present": {"type": "noul", "noul": 0.01, "secret": "omit"}}
    box = {"top": 0, "bottom": 1, "left": 0.123456789, "right": 0.9}
    for identifier, optional_fields in (
        ("zero", {"confidence": 0}),
        ("null", {"confidence": None}),
        ("absent", {}),
    ):
        questions[identifier] = {"type": "box", "instructions": {"target": "label"}}
        answers[identifier] = {"type": "box", "box": box, "image_path": "/state/photos/1", **optional_fields}
    # Map keys link answers to questions, independently of return order.
    answers = dict(reversed(answers.items()))
    assert _decisions_input("state", questions) == {
        "state": "state",
        "questions": [
            {"id": "present", "type": "noul", "instructions": "Is the target present?"},
            *({"id": identifier, **questions[identifier]} for identifier in ("zero", "null", "absent")),
        ],
    }
    assert _decisions_output(answers, questions) == {
        "answers": [
            *({"id": identifier, **answers[identifier]} for identifier in ("absent", "null", "zero")),
            {"id": "present", "type": "noul", "noul": 0.01},
        ]
    }


@pytest.mark.parametrize("name", ["present", None], ids=["named", "unnamed"])
def test_decisions_named_answers_match_questions_when_reordered(name):
    questions = [
        {"type": "box", "name": "bounds", "instructions": "The target"},
        {"type": "predicate", "instructions": "Is it present?", **({"name": name} if name else {})},
    ]
    box = {"top": -0.1, "bottom": 0, "left": 0.8, "right": 0.2}
    assert _decisions_output(
        [{"type": "predicate", "name": name, "probability": 0.01}, {"type": "box", "name": "bounds", "box": box}],
        questions,
    ) == {
        "answers": [
            {"id": name or "question_1", "type": "boolean", "probability": 0.01},
            {"id": "bounds", "type": "box", "box": box},
        ]
    }


def test_decisions_nested_media_and_existing_references():
    image = decisions_image_input()[0]["content"][1]["image_url"]
    reference = {"type": "external_attachment", "url": "https://example.com/image.png", "content_type": "image/png"}
    questions = {"bounds": {"type": "box", "instructions": {"image": image, "text": "target"}}}
    span_input = _decisions_input({"photos": [reference, image], "remote": "https://example.com/a.png"}, questions)
    assert span_input["state"]["photos"][0] == reference
    assert isinstance(span_input["state"]["photos"][1], Attachment)
    assert isinstance(span_input["questions"][0]["instructions"]["image"], Attachment)
    assert span_input["state"]["remote"] == "https://example.com/a.png"
    assert questions["bounds"]["instructions"]["image"] == image


def test_decisions_media_transform_handles_serialized_images():
    data_url = decisions_image_input()[0]["content"][1]["image_url"]
    image = {"type": "image", "format": "png", "data_base64": data_url.split(",", 1)[1]}
    questions = {"bounds": {"type": "box", "instructions": "The target"}}
    span_input = _question_span_input({"photo": image}, questions, media_transform=_image_payload)
    attachment = span_input["state"]["photo"]
    assert isinstance(attachment, Attachment)
    assert attachment.data == base64.b64decode(image["data_base64"])
    box = {"top": 0, "bottom": 1, "left": 0, "right": 1}
    assert _decisions_output({"bounds": {"type": "box", "box": box}}, questions, span_input) == {
        "answers": [{"id": "bounds", "type": "box", "box": box, "image_path": "/state/photo"}]
    }
    assert image["data_base64"] == data_url.split(",", 1)[1]


@pytest.mark.parametrize("openai_image", [False, True], ids=["data-url", "litellm-input-image"])
def test_decisions_media_failure_keeps_entire_payload(monkeypatch, openai_image):
    image = decisions_image_input()[0]["content"][1]["image_url"]
    from braintrust.integrations import question
    from braintrust.integrations.litellm.tracing import _decisions_media
    from braintrust.integrations.openai import tracing as openai_tracing

    materialize = question._materialize_attachment
    calls = 0

    def fail_second(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("conversion failed")
        return materialize(value)

    monkeypatch.setattr(question, "_materialize_attachment", fail_second)
    monkeypatch.setattr(openai_tracing, "_materialize_attachment", fail_second)
    state = {"image": {"type": "input_image", "image_url": image} if openai_image else image}
    questions = {"bounds": {"type": "box", "instructions": {"image": image}}}
    span_input = (
        _decisions_input(state, questions, media_transform=_decisions_media)
        if openai_image
        else _question_span_input(state, questions)
    )
    assert calls == 2
    assert span_input == {
        "state": state,
        "questions": [{"id": "bounds", **questions["bounds"]}],
    }


@pytest.mark.parametrize("parse_usage", (parse_openai_usage, parse_litellm_usage), ids=("openai", "litellm"))
def test_decisions_usage_with_nonzero_token_details(parse_usage):
    # The live Decisions recordings report zero for these optional details.
    # Exercise nonzero values directly through the local usage parser too.
    metrics = parse_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 32, "cache_write_tokens": 16},
            "output_tokens_details": {"reasoning_tokens": 6},
        }
    )
    assert metrics == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "tokens": 120,
        "prompt_cached_tokens": 32,
        "prompt_cache_creation_tokens": 16,
        "completion_reasoning_tokens": 6,
    }
