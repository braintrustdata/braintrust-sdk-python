from braintrust.conftest import get_vcr_config


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
