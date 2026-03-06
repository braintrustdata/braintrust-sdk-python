from pathlib import Path

import pytest
from braintrust import logger
from braintrust.test_helpers import init_test_logger

from ._test_agno_helpers import PROJECT_NAME


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(Path(__file__).parent.parent / "cassettes"),
    }
