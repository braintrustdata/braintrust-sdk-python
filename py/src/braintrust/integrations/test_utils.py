from braintrust.integrations.utils import (
    _camel_to_snake,
    _is_supported_metric_value,
    _merge_timing_and_usage_metrics,
    _parse_openai_usage_metrics,
    _prettify_response_params,
)


class NotGiven:
    pass


def test_camel_to_snake():
    assert _camel_to_snake("promptTokens") == "prompt_tokens"
    assert _camel_to_snake("TotalTokens") == "total_tokens"
    assert _camel_to_snake("already_snake") == "already_snake"


def test_is_supported_metric_value_excludes_booleans():
    assert _is_supported_metric_value(1)
    assert _is_supported_metric_value(1.5)
    assert not _is_supported_metric_value(True)
    assert not _is_supported_metric_value(False)
    assert not _is_supported_metric_value("1")


def test_parse_openai_usage_metrics_handles_nested_token_details():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "input_tokens_details": {"cached_tokens": 4},
        "is_byok": True,
    }

    metrics = _parse_openai_usage_metrics(
        usage,
        token_name_map={
            "prompt_tokens": "prompt_tokens",
            "completion_tokens": "completion_tokens",
            "total_tokens": "tokens",
        },
        token_prefix_map={"input": "prompt"},
    )

    assert metrics == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "tokens": 30,
        "prompt_cached_tokens": 4,
    }


def test_prettify_response_params_filters_not_given_without_mutating_input():
    original = {
        "model": "gpt-5",
        "response_format": object(),
        "optional": NotGiven(),
    }

    prettified = _prettify_response_params(original, drop_not_given=True)

    assert prettified == {
        "model": "gpt-5",
        "response_format": original["response_format"],
    }
    assert "optional" in original


def test_merge_timing_and_usage_metrics(monkeypatch):
    monkeypatch.setattr("braintrust.integrations.utils.time.time", lambda: 15.0)

    metrics = _merge_timing_and_usage_metrics(
        10.0,
        {"usage": 1},
        lambda usage: {"tokens": usage["usage"]},
        12.0,
    )

    assert metrics == {
        "start": 10.0,
        "end": 15.0,
        "duration": 5.0,
        "time_to_first_token": 2.0,
        "tokens": 1,
    }
