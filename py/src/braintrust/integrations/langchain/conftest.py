import os

import pytest


@pytest.fixture(scope="module")
def vcr_config():
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"

    return {
        "filter_headers": [
            "authorization",
            "x-goog-api-key",
            "x-api-key",
            "api-key",
            "openai-api-key",
        ],
        "record_mode": record_mode,
        "match_on": ["uri", "method"],
        "decode_compressed_response": True,
    }
