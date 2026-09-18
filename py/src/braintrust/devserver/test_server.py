from unittest.mock import MagicMock

import pytest
from braintrust.test_helpers import has_devserver_installed


def test_ui_scorer_forwards_org_name_to_function_invoke():
    if not has_devserver_installed():
        pytest.skip("Devserver dependencies not installed (requires .[cli])")

    from braintrust.devserver.server import make_scorer

    response = MagicMock()
    response.json.return_value = {"score": 1}
    connection = MagicMock()
    connection.post.return_value = response
    state = MagicMock()
    state.org_name = "test-org"
    state.proxy_conn.return_value = connection
    state.current_span.get.return_value.export.return_value = {}

    scorer = make_scorer(state, "ui-scorer", {"function_id": "function-id"}, "project-id")

    assert scorer("input", "output", "expected", {}) == {"score": 1}
    assert connection.post.call_args.args == ("function/invoke",)
    assert connection.post.call_args.kwargs["headers"] == {
        "Accept": "application/json",
        "x-bt-org-name": "test-org",
        "x-bt-project-id": "project-id",
    }
