"""Regression tests for ProjectIdCache keying by full project identity."""

from unittest.mock import MagicMock, patch

from braintrust.framework2 import ProjectIdCache


def _mock_state_with_distinct_ids():
    mock_state = MagicMock()
    mock_state.org_name = "test-org"

    def post_project(body):
        group = body.get("project_group_name")
        return {"id": f"id-{group}", "name": body["name"]}

    mock_state.api_client.return_value.projects.post_project.side_effect = post_project
    return mock_state


def test_get_by_name_does_not_collide_across_project_groups():
    mock_state = _mock_state_with_distinct_ids()
    cache = ProjectIdCache()

    with patch("braintrust.logger._state", mock_state):
        group_a_id = cache.get_by_name("shared", "group-a")
        group_b_id = cache.get_by_name("shared", "group-b")

    assert group_a_id == "id-group-a"
    assert group_b_id == "id-group-b"
    post = mock_state.api_client.return_value.projects.post_project
    assert post.call_count == 2
    assert post.call_args_list[0].kwargs["body"]["project_group_name"] == "group-a"
    assert post.call_args_list[1].kwargs["body"]["project_group_name"] == "group-b"


def test_get_by_name_reuses_cached_id_within_same_group():
    mock_state = _mock_state_with_distinct_ids()
    cache = ProjectIdCache()

    with patch("braintrust.logger._state", mock_state):
        first = cache.get_by_name("shared", "group-a")
        cached = cache.get_by_name("shared", "group-a")

    assert first == cached == "id-group-a"
    assert mock_state.api_client.return_value.projects.post_project.call_count == 1


def test_get_by_name_distinguishes_ungrouped_from_grouped():
    mock_state = _mock_state_with_distinct_ids()
    cache = ProjectIdCache()

    with patch("braintrust.logger._state", mock_state):
        ungrouped_id = cache.get_by_name("shared")
        grouped_id = cache.get_by_name("shared", "group-a")

    assert ungrouped_id == "id-None"
    assert grouped_id == "id-group-a"
    assert mock_state.api_client.return_value.projects.post_project.call_count == 2
