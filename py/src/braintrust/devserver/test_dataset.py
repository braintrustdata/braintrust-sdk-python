from unittest.mock import AsyncMock, Mock

import pytest

from braintrust.devserver.dataset import get_dataset


@pytest.mark.asyncio
async def test_get_dataset_named_reference_forwards_version_and_environment(monkeypatch):
    state = Mock()
    dataset = Mock()
    init_dataset = Mock(return_value=dataset)
    monkeypatch.setattr("braintrust.devserver.dataset.init_dataset", init_dataset)

    result = await get_dataset(
        state,
        {
            "project_name": "project-name",
            "dataset_name": "dataset-name",
            "dataset_version": "version-1",
            "dataset_environment": "production",
            "_internal_btql": {"limit": 10},
        },
    )

    assert result is dataset
    init_dataset.assert_called_once_with(
        state=state,
        project="project-name",
        name="dataset-name",
        version="version-1",
        environment="production",
        _internal_btql={"limit": 10},
    )


@pytest.mark.asyncio
async def test_get_dataset_id_reference_forwards_version_and_environment(monkeypatch):
    state = Mock()
    dataset = Mock()
    get_dataset_by_id = AsyncMock(return_value={"project_id": "project-id", "dataset": "dataset-name"})
    init_dataset = Mock(return_value=dataset)
    monkeypatch.setattr("braintrust.devserver.dataset.get_dataset_by_id", get_dataset_by_id)
    monkeypatch.setattr("braintrust.devserver.dataset.init_dataset", init_dataset)

    result = await get_dataset(
        state,
        {
            "dataset_id": "dataset-id",
            "dataset_version": "version-1",
            "dataset_environment": "production",
            "_internal_btql": {"limit": 10},
        },
    )

    assert result is dataset
    get_dataset_by_id.assert_awaited_once_with(state, "dataset-id")
    init_dataset.assert_called_once_with(
        state=state,
        project_id="project-id",
        name="dataset-name",
        version="version-1",
        environment="production",
        _internal_btql={"limit": 10},
    )


@pytest.mark.asyncio
async def test_get_dataset_returns_inline_data_unchanged():
    state = Mock()
    data = [{"input": "hello", "expected": "world"}]

    assert await get_dataset(state, {"data": data}) is data
