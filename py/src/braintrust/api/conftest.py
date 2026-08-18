import os

import pytest


@pytest.fixture
def api_key() -> str:
    return os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")
