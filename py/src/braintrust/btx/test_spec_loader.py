from uuid import UUID

import pytest

from .spec_loader import load_specs


@pytest.mark.parametrize("live", [False, True], ids=["replay", "live"])
def test_cache_variables_are_shared_by_request_and_expectation(tmp_path, monkeypatch, live):
    monkeypatch.setenv("BTX_CLIENT", "openai")
    provider = tmp_path / "openai"
    provider.mkdir()
    (provider / "cache.yaml").write_text(
        """name: cache
type: llm_span_test
provider: openai
endpoint: /v1/responses
variables:
  client: !gen test_runner_client
  nonce: !gen vcr_nonce
requests:
  - input: "{{client}} {{nonce}}"
expected_brainstore_spans:
  - input: "{{client}} {{nonce}}"
"""
    )

    first = load_specs(tmp_path, live=live)[0]
    second = load_specs(tmp_path, live=live)[0]
    first_input = first.requests[0]["input"]
    assert first.expected_brainstore_spans[0]["input"] == first_input
    assert first_input.startswith("python-openai ")
    if live:
        assert str(UUID(first_input.split(" ")[1])) == first_input.split(" ")[1]
        assert second.requests[0]["input"] != first_input
    else:
        assert first_input == second.requests[0]["input"] == "python-openai vcr-mode"
