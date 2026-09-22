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
    dict_metadata: dict[str, Any] = {"user_id": "user-1"}

    logger.log(metadata=mapping_metadata)
    logger.log(metadata=PydanticV2Metadata())
    logger.log(metadata=PydanticV1Metadata())

    logger.emit_log(body="message", level="info", metadata=dict_metadata)
    logger.trace("message", metadata=dict_metadata)
    logger.debug("message", metadata=dict_metadata)
    logger.info("message", metadata=dict_metadata)
    logger.warn("message", metadata=dict_metadata)
    logger.error("message", metadata=dict_metadata)
    logger.fatal("message", metadata=dict_metadata)

    logger.emit_log("User {user_id}", "info", metadata=dict_metadata, user_id="user-1")
    logger.trace("User {user_id}", metadata=dict_metadata, user_id="user-1")
    logger.debug("User {user_id}", metadata=dict_metadata, user_id="user-1")
    logger.info("User {user_id}", metadata=dict_metadata, user_id="user-1")
    logger.warn("User {user_id}", metadata=dict_metadata, user_id="user-1")
    logger.error("User {user_id}", metadata=dict_metadata, user_id="user-1")
    logger.fatal("User {user_id}", metadata=dict_metadata, user_id="user-1")

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
