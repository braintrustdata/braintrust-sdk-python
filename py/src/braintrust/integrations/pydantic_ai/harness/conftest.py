import pytest
from braintrust import setup_pydantic_ai
from braintrust.test_helpers import init_test_logger
from braintrust.test_helpers import with_memory_logger as with_memory_logger


@pytest.fixture
def harness_spans(with_memory_logger):
    pytest.importorskip("pydantic_ai_harness", exc_type=ModuleNotFoundError)
    project = "test-pydantic-ai-harness"
    assert setup_pydantic_ai(project_name=project)
    init_test_logger(project)
    return with_memory_logger
