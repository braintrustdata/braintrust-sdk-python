import os
import subprocess
import sys
import textwrap
import unittest.mock
from contextlib import contextmanager
from pathlib import Path

import pytest
import vcr
from braintrust import Attachment, logger
from braintrust.conftest import get_vcr_config
from braintrust.test_helpers import init_test_logger


# Source directory paths (resolved to handle installed vs source locations).
# When running inside a subprocess spawned by verify_autoinstrument_script,
# __file__ may resolve to the installed site-packages location where
# non-Python files (cassettes, scripts) are absent.  The env-var override
# lets the parent process hand down the *source-tree* integrations path.
_INTEGRATIONS_DIR = Path(os.environ.get("BRAINTRUST_INTEGRATIONS_DIR", Path(__file__).resolve().parent))
AUTO_TEST_SCRIPTS_DIR = _INTEGRATIONS_DIR / "auto_test_scripts"
from braintrust.integrations.utils import (
    _attachment_filename_for_mime_type,
    _camel_to_snake,
    _extract_audio_output,
    _infer_audio_mime_type,
    _is_supported_metric_value,
    _log_and_end_span,
    _log_error_and_end_span,
    _materialize_attachment,
    _merge_timing_and_usage_metrics,
    _parse_openai_usage_metrics,
    _prettify_response_params,
    _ResolvedAttachment,
    _serialize_response_format,
    _timing_metrics,
    _try_to_dict,
)


@contextmanager
def autoinstrument_test_context(
    cassette_name: str,
    *,
    integration: str | None = None,
    use_vcr: bool = True,
    cassettes_dir: Path | None = None,
):
    """Context manager for auto_instrument tests.

    Sets up the shared memory_logger context and, by default, VCR.

    Use ``integration`` to automatically resolve cassettes from
    ``integrations/<name>/cassettes/``::

        with autoinstrument_test_context("test_auto_openai", integration="openai") as ml:
            ...

    Use ``cassettes_dir`` to override with an explicit path (takes
    precedence over ``integration``).

    Use ``use_vcr=False`` for tests that replay provider traffic through a
    non-VCR mechanism, such as the Claude Agent SDK subprocess cassette
    transport.
    """
    if cassettes_dir is None and integration is not None:
        cassettes_dir = _INTEGRATIONS_DIR / integration / "cassettes"
    if cassettes_dir is None and use_vcr:
        raise ValueError(
            "Either integration or cassettes_dir is required – e.g. integration='openai' or cassettes_dir=Path(...)"
        )

    cassette_path = cassettes_dir / f"{cassette_name}.yaml" if cassettes_dir else None

    init_test_logger("test-auto-instrument")

    with logger._internal_with_memory_background_logger() as memory_logger:
        memory_logger.pop()  # Clear any prior spans

        if not use_vcr:
            yield memory_logger
            return

        my_vcr = vcr.VCR(**get_vcr_config())
        with my_vcr.use_cassette(str(cassette_path)):
            yield memory_logger


def run_in_subprocess(code: str, timeout: int = 30, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run Python code in a fresh subprocess."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


def verify_autoinstrument_script(script_name: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a test script from the integrations auto_test_scripts directory.

    Raises AssertionError if the script exits with non-zero code.
    """
    script_path = AUTO_TEST_SCRIPTS_DIR / script_name
    env = os.environ.copy()
    # Hand source-tree integrations dir to the subprocess so that
    # cassettes and auto_test_scripts resolve correctly even when
    # braintrust is installed from a wheel (which excludes .yaml files).
    env["BRAINTRUST_INTEGRATIONS_DIR"] = str(_INTEGRATIONS_DIR)
    env["BRAINTRUST_CLAUDE_AGENT_SDK_CASSETTES_DIR"] = str(_INTEGRATIONS_DIR / "claude_agent_sdk" / "cassettes")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    assert result.returncode == 0, f"Script {script_name} failed:\n{result.stderr}"
    return result


def assert_metrics_are_valid(metrics, start=None, end=None):
    assert metrics
    # assert 0 < metrics["time_to_first_token"]
    assert 0 < metrics["tokens"]
    assert 0 < metrics["prompt_tokens"]
    assert 0 < metrics["completion_tokens"]
    # we use <= because windows timestamps are not very precise and
    # we use VCR which skips HTTP requests.
    if start and end:
        assert start <= metrics["start"] <= metrics["end"] <= end
    else:
        assert metrics["start"] <= metrics["end"]


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


def test_attachment_filename_for_mime_type_prefers_known_extensions():
    assert _attachment_filename_for_mime_type("image/svg+xml", prefix="image") == "image.svg"
    assert (
        _attachment_filename_for_mime_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        == "file.xlsx"
    )


def test_materialize_attachment_from_bytes_uses_default_filename():
    resolved = _materialize_attachment(b"hello", mime_type="image/png")

    assert isinstance(resolved, _ResolvedAttachment)
    assert isinstance(resolved.attachment, Attachment)
    assert resolved.mime_type == "image/png"
    assert resolved.filename == "image.png"
    assert resolved.attachment.reference["content_type"] == "image/png"
    assert resolved.attachment.reference["filename"] == "image.png"


def test_materialize_attachment_from_bytes_accepts_custom_prefix():
    resolved = _materialize_attachment(b"hello", mime_type="application/pdf", prefix="document")

    assert isinstance(resolved, _ResolvedAttachment)
    assert resolved.mime_type == "application/pdf"
    assert resolved.filename == "document.pdf"


def test_materialize_attachment_prefix_handles_mime_suffixes():
    resolved = _materialize_attachment(b"<svg />", mime_type="image/svg+xml", prefix="image")

    assert isinstance(resolved, _ResolvedAttachment)
    assert resolved.mime_type == "image/svg+xml"
    assert resolved.filename == "image.svg"


def test_materialize_attachment_from_base64_accepts_data_urls_and_custom_filenames():
    resolved = _materialize_attachment(
        "data:image/png;base64,aGVsbG8=",
        mime_type="image/png",
        filename="generated_image_0.png",
    )

    assert isinstance(resolved, _ResolvedAttachment)
    assert isinstance(resolved.attachment, Attachment)
    assert resolved.attachment.reference["content_type"] == "image/png"
    assert resolved.attachment.reference["filename"] == "generated_image_0.png"


def test_materialize_attachment_returns_none_for_invalid_base64_payloads():
    assert _materialize_attachment("aGVsbG8=!", mime_type="image/png") is None


def test_materialize_attachment_converts_valid_data_url():
    data_url = "data:image/png;base64,aGVsbG8="

    resolved = _materialize_attachment(data_url, label="image")

    assert isinstance(resolved, _ResolvedAttachment)
    assert isinstance(resolved.attachment, Attachment)
    assert resolved.attachment.reference["content_type"] == "image/png"
    assert resolved.attachment.reference["filename"] == "image.png"


def test_resolved_attachment_multimodal_part_payload_uses_image_url_for_images():
    resolved = _materialize_attachment(b"hello", mime_type="image/png")

    assert resolved is not None
    assert resolved.multimodal_part_payload == {"image_url": {"url": resolved.attachment}}


def test_resolved_attachment_multimodal_part_payload_uses_file_parts_for_non_images():
    resolved = _materialize_attachment(b"hello", mime_type="application/pdf", filename="document.pdf")

    assert resolved is not None
    assert resolved.multimodal_part_payload == {
        "file": {
            "file_data": resolved.attachment,
            "filename": "document.pdf",
        }
    }


def test_materialize_attachment_handles_common_input_shapes(tmp_path):
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

    path_attachment = _materialize_attachment(file_path)
    bytes_attachment = _materialize_attachment(b"abc", filename="example.png")
    tuple_attachment = _materialize_attachment((str(file_path), b"abc", "image/png"))
    file_attachment = _materialize_attachment(FileLike())

    for resolved in (path_attachment, bytes_attachment, tuple_attachment, file_attachment):
        assert isinstance(resolved, _ResolvedAttachment)
        assert resolved.attachment.reference["filename"] == "example.png"
        assert resolved.attachment.reference["content_type"] == "image/png"


def test_materialize_attachment_preserves_file_position(tmp_path):
    file_path = tmp_path / "example.png"
    file_path.write_bytes(b"abc")

    with file_path.open("rb") as file_obj:
        assert file_obj.tell() == 0
        resolved = _materialize_attachment(file_obj)
        assert isinstance(resolved, _ResolvedAttachment)
        assert file_obj.tell() == 0


def test_materialize_attachment_preserves_invalid_base64_strings_without_mime_type():
    assert _materialize_attachment("data:image/png;base64,aGVsbG8=!") is None


def test_materialize_attachment_uses_file_prefix_for_non_image_mime_types():
    resolved = _materialize_attachment("data:application/pdf;base64,aGVsbG8=")

    assert isinstance(resolved, _ResolvedAttachment)
    assert resolved.attachment.reference["content_type"] == "application/pdf"
    assert resolved.attachment.reference["filename"] == "file.pdf"


def test_materialize_attachment_preserves_existing_attachment_filename_over_prefix():
    attachment = Attachment(data=b"hello", filename="existing.pdf", content_type="application/pdf")

    resolved = _materialize_attachment(attachment, prefix="document")

    assert isinstance(resolved, _ResolvedAttachment)
    assert resolved.attachment.reference["filename"] == "existing.pdf"


def test_materialize_attachment_returns_none_for_non_data_url_strings():
    assert _materialize_attachment("https://example.com/image.png") is None


def test_infer_audio_mime_type_prefers_response_headers():
    raw_response = unittest.mock.Mock(headers={"content-type": "audio/mpeg; charset=binary"})
    response = unittest.mock.Mock(response=raw_response)

    assert _infer_audio_mime_type(response, response_format="wav") == "audio/mpeg"


def test_extract_audio_output_materializes_attachment_from_binary_response():
    raw_response = unittest.mock.Mock(headers={"content-type": "audio/mpeg"})
    response = unittest.mock.Mock(content=b"audio-bytes", response=raw_response)

    output = _extract_audio_output(response, prefix="generated_speech")

    assert output["type"] == "audio"
    assert output["mime_type"] == "audio/mpeg"
    assert output["audio_size_bytes"] == len(b"audio-bytes")
    attachment = output["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "audio/mpeg"
    assert attachment.reference["filename"] == "generated_speech.mp3"


def test_extract_audio_output_supports_mapping_with_raw_response_only():
    raw_response = unittest.mock.Mock(headers={"content-type": "audio/wav"}, content=b"wave")

    output = _extract_audio_output({"response": raw_response}, prefix="generated_speech")

    assert output["type"] == "audio"
    assert output["mime_type"] == "audio/wav"
    assert output["audio_size_bytes"] == len(b"wave")
    attachment = output["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == "audio/wav"
    assert attachment.reference["filename"] == "generated_speech.wav"


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
