import json

import pytest
from braintrust.conftest import get_vcr_config, sanitize_generated_images


def test_sanitize_generated_images_preserves_metadata_and_scrubs_headers():
    response = {
        "body": {"string": json.dumps({"data": [{"b64_json": "large-image"}], "usage": {"output_tokens": 42}})},
        "headers": {"Content-Length": ["1234"], "Set-Cookie": ["secret"]},
    }
    result = sanitize_generated_images(response)
    body = json.loads(result["body"]["string"])
    assert body["data"][0]["b64_json"].startswith("iVBORw0KGgo")
    assert body["usage"] == {"output_tokens": 42}
    assert result["headers"] == {"Content-Length": [str(len(result["body"]["string"].encode()))]}


def test_vcr_config_scrubs_sensitive_provider_headers():
    config = get_vcr_config()

    assert {"cookie", "openai-organization", "openai-project"} <= set(config["filter_headers"])

    response = {
        "headers": {
            "Content-Type": ["application/json"],
            "OpenAI-Organization": ["org-sensitive"],
            "OpenAI-Project": ["proj-sensitive"],
            "Set-Cookie": ["session=sensitive"],
        }
    }
    scrubbed = config["before_record_response"](response)

    assert scrubbed["headers"] == {"Content-Type": ["application/json"]}


@pytest.mark.vcr(cassette_name="test_vcr_config_scrubs_sensitive_provider_headers")
def test_vcr_marker_cassette_name_overrides_default_cassette(vcr_cassette_name):
    assert vcr_cassette_name == "test_vcr_config_scrubs_sensitive_provider_headers"
