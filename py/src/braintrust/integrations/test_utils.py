import unittest.mock

import pytest
from braintrust import Attachment
from braintrust.integrations.utils import (
    _attachment_filename_for_mime_type,
    _attachment_from_base64_data,
    _attachment_from_bytes,
    _attachment_from_file_input,
    _camel_to_snake,
    _convert_data_url_to_attachment,
    _image_url_payload,
    _is_supported_metric_value,
    _log_and_end_span,
    _log_error_and_end_span,
    _merge_timing_and_usage_metrics,
    _parse_openai_usage_metrics,
    _prettify_response_params,
    _serialize_response_format,
    _timing_metrics,
    _try_to_dict,
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


def test_try_to_dict_uses_pydantic_model_dump_for_basemodel_instances():
    pydantic = pytest.importorskip("pydantic")

    class Usage(pydantic.BaseModel):
        tokens: int
        cached_tokens: int

    result = _try_to_dict(Usage(tokens=3, cached_tokens=1))

    assert result == {"tokens": 3, "cached_tokens": 1}


def test_try_to_dict_uses_to_dict_when_available():
    class ToDictOnly:
        __slots__ = ("_payload",)

        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

    result = _try_to_dict(ToDictOnly({"tokens": 3}))

    assert result == {"tokens": 3}


def test_try_to_dict_falls_back_from_model_dump_python_to_bare_model_dump():
    class BareModelDumpOnly:
        def model_dump(self, mode=None):
            if mode == "python":
                raise TypeError("mode not supported")
            return {"tokens": 3}

    result = _try_to_dict(BareModelDumpOnly())

    assert result == {"tokens": 3}


def test_try_to_dict_continues_past_non_dict_converter_results():
    class MixedConverters:
        def model_dump(self, mode=None):
            return [mode]

        def to_dict(self):
            return {"tokens": 3}

    result = _try_to_dict(MixedConverters())

    assert result == {"tokens": 3}


def test_try_to_dict_falls_back_to_vars_for_plain_objects():
    class PlainObject:
        def __init__(self):
            self.foo = "bar"
            self.count = 2

    result = _try_to_dict(PlainObject())

    assert result == {"foo": "bar", "count": 2}


def test_try_to_dict_returns_original_when_no_conversion_is_possible():
    obj = object()

    result = _try_to_dict(obj)

    assert result is obj


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


def test_attachment_filename_for_mime_type_handles_suffixes_and_prefixes():
    assert _attachment_filename_for_mime_type("image/png", prefix="image") == "image.png"
    assert _attachment_filename_for_mime_type("application/pdf", prefix="document") == "document.pdf"
    assert _attachment_filename_for_mime_type("image/svg+xml", prefix="file") == "file.svg"


def test_attachment_from_bytes_uses_default_filename():
    attachment = _attachment_from_bytes(b"hello", "image/png")

    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "image/png"
    assert attachment.reference["filename"] == "file.png"


def test_attachment_from_base64_data_accepts_data_urls_and_custom_filenames():
    attachment = _attachment_from_base64_data(
        "data:image/png;base64,aGVsbG8=",
        "image/png",
        filename="generated_image_0.png",
    )

    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "image/png"
    assert attachment.reference["filename"] == "generated_image_0.png"


def test_attachment_from_base64_data_returns_none_for_invalid_payloads():
    assert _attachment_from_base64_data("aGVsbG8=!", "image/png") is None


def test_convert_data_url_to_attachment_converts_valid_base64():
    data_url = "data:image/png;base64,aGVsbG8="

    attachment = _convert_data_url_to_attachment(data_url)

    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "image/png"
    assert attachment.reference["filename"] == "image.png"


def test_image_url_payload_wraps_attachment_and_string_urls():
    attachment = _attachment_from_bytes(b"hello", "image/png")

    assert _image_url_payload(attachment) == {"image_url": {"url": attachment}}
    assert _image_url_payload("https://example.com/image.png") == {
        "image_url": {"url": "https://example.com/image.png"}
    }


def test_attachment_from_file_input_handles_common_input_shapes(tmp_path):
    file_path = tmp_path / "example.png"
    file_path.write_bytes(b"abc")

    class FileLike:
        def __init__(self):
            self.name = str(file_path)
            self._position = 0

        def read(self):
            self._position = 3
            return b"abc"

        def tell(self):
            return 0

        def seek(self, position):
            self._position = position

    path_attachment = _attachment_from_file_input(file_path)
    bytes_attachment = _attachment_from_file_input(b"abc", filename="example.png")
    tuple_attachment = _attachment_from_file_input((str(file_path), b"abc", "image/png"))
    file_attachment = _attachment_from_file_input(FileLike())

    for attachment in (path_attachment, bytes_attachment, tuple_attachment, file_attachment):
        assert isinstance(attachment, Attachment)
        assert attachment.reference["filename"] == "example.png"
        assert attachment.reference["content_type"] == "image/png"


def test_attachment_from_file_input_preserves_file_position(tmp_path):
    file_path = tmp_path / "example.png"
    file_path.write_bytes(b"abc")

    with file_path.open("rb") as file_obj:
        assert file_obj.tell() == 0
        attachment = _attachment_from_file_input(file_obj)
        assert isinstance(attachment, Attachment)
        assert file_obj.tell() == 0


def test_convert_data_url_to_attachment_preserves_invalid_base64():
    data_url = "data:image/png;base64,aGVsbG8=!"

    converted = _convert_data_url_to_attachment(data_url)

    assert converted == data_url


def test_convert_data_url_to_attachment_uses_file_prefix_for_non_image_mime_types():
    data_url = "data:application/pdf;base64,aGVsbG8="

    attachment = _convert_data_url_to_attachment(data_url)

    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "application/pdf"
    assert attachment.reference["filename"] == "file.pdf"


def test_convert_data_url_to_attachment_preserves_non_data_urls():
    value = "https://example.com/image.png"

    converted = _convert_data_url_to_attachment(value)

    assert converted == value


def test_serialize_response_format_with_pydantic_basemodel_subclass():
    pydantic = pytest.importorskip("pydantic")

    class ResponseFormat(pydantic.BaseModel):
        answer: str

    serialized = _serialize_response_format(ResponseFormat)

    assert serialized["type"] == "json_schema"
    assert serialized["json_schema"]["name"] == "ResponseFormat"
    assert serialized["json_schema"]["schema"]["properties"]["answer"]["title"] == "Answer"


def test_timing_metrics_includes_time_to_first_token_when_present():
    assert _timing_metrics(10.0, 15.0, 12.0) == {
        "start": 10.0,
        "end": 15.0,
        "duration": 5.0,
        "time_to_first_token": 2.0,
    }


def test_timing_metrics_omits_time_to_first_token_when_absent():
    assert _timing_metrics(10.0, 15.0) == {
        "start": 10.0,
        "end": 15.0,
        "duration": 5.0,
    }


def test_log_and_end_span_logs_populated_event_then_ends():
    span = unittest.mock.Mock()

    _log_and_end_span(
        span,
        output={"answer": "4"},
        metrics={"tokens": 2},
        metadata={"provider": "test"},
    )

    span.log.assert_called_once_with(
        output={"answer": "4"},
        metrics={"tokens": 2},
        metadata={"provider": "test"},
    )
    span.end.assert_called_once_with()


def test_log_and_end_span_skips_log_for_empty_event():
    span = unittest.mock.Mock()

    _log_and_end_span(span)

    span.log.assert_not_called()
    span.end.assert_called_once_with()


def test_log_error_and_end_span_logs_error_then_ends():
    span = unittest.mock.Mock()
    error = RuntimeError("boom")

    _log_error_and_end_span(span, error)

    span.log.assert_called_once_with(error=error)
    span.end.assert_called_once_with()


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
