import pytest
from braintrust.conftest import get_vcr_config


@pytest.fixture(scope="module")
def vcr_config():
    return {
        **get_vcr_config(),
        "match_on": ["uri", "method"],
    }
