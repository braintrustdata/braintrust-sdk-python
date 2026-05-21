from collections.abc import Mapping
from typing import Any

from braintrust.logger import Dataset, Experiment, Logger


class PydanticV2Metadata:
    def model_dump(self, *, exclude_none: bool = False) -> Mapping[str, Any]:
        assert exclude_none
        return {"user_id": "user-1"}


class PydanticV1Metadata:
    def dict(self, *, exclude_none: bool = False) -> Mapping[str, Any]:
        assert exclude_none
        return {"user_id": "user-1"}


def accepts_logger_metadata(logger: Logger) -> None:
    mapping_metadata: Mapping[str, Any] = {"user_id": "user-1"}

    logger.log(metadata=mapping_metadata)
    logger.log(metadata=PydanticV2Metadata())
    logger.log(metadata=PydanticV1Metadata())

    logger.log_feedback(id="event-id", metadata=mapping_metadata)
    logger.log_feedback(id="event-id", metadata=PydanticV2Metadata())
    logger.log_feedback(id="event-id", metadata=PydanticV1Metadata())


def accepts_experiment_metadata(experiment: Experiment) -> None:
    mapping_metadata: Mapping[str, Any] = {"user_id": "user-1"}

    experiment.log(metadata=mapping_metadata)
    experiment.log(metadata=PydanticV2Metadata())
    experiment.log(metadata=PydanticV1Metadata())

    experiment.log_feedback(id="event-id", metadata=mapping_metadata)
    experiment.log_feedback(id="event-id", metadata=PydanticV2Metadata())
    experiment.log_feedback(id="event-id", metadata=PydanticV1Metadata())


def accepts_dataset_metadata(dataset: Dataset) -> None:
    mapping_metadata: Mapping[str, Any] = {"user_id": "user-1"}

    dataset.insert(metadata=mapping_metadata)
    dataset.insert(metadata=PydanticV2Metadata())
    dataset.insert(metadata=PydanticV1Metadata())

    dataset.update(id="record-id", metadata=mapping_metadata)
    dataset.update(id="record-id", metadata=PydanticV2Metadata())
    dataset.update(id="record-id", metadata=PydanticV1Metadata())
