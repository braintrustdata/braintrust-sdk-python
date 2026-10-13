import pytest
from braintrust.integrations.litellm.tracing import _parse_metrics_from_usage as parse_litellm_usage
from braintrust.integrations.openai.tracing import _decisions_input, _decisions_output
from braintrust.integrations.openai.tracing import _parse_metrics_from_usage as parse_openai_usage
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


def assert_decisions_span(spans, response, *, name, model, instrumentation, start=None, end=None):
    assert len(spans) == 1
    span = spans[0]
    assert_dict_matches(span["span_attributes"], {"name": name, "type": "question"})
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
    questions = {"positive": {"type": "noul", "instructions": "Is this positive?", "secret": "omit"}}
    answers = {"positive": {"type": "noul", "noul": 0.9, "secret": "omit"}}
    assert _decisions_input("state", questions) == {
        "state": "state",
        "questions": [{"id": "positive", "type": "noul", "instructions": "Is this positive?"}],
    }
    assert _decisions_output(answers, questions) == {"answers": [{"id": "positive", "type": "noul", "noul": 0.9}]}


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
