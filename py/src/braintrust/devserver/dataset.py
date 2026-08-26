from typing import Any

from braintrust import init_dataset
from braintrust._generated_types import RunEvalData, RunEvalData1, RunEvalData2
from braintrust.logger import BraintrustState


# NOTE: To make this performant, we'll have to make these functions work with async i/o
async def get_dataset(state: BraintrustState, data: RunEvalData | RunEvalData1 | RunEvalData2 | dict[str, Any]) -> Any:
    """
    Get dataset from various data sources.

    Handles:
    - Dataset reference by project_name/dataset_name
    - Dataset reference by dataset_id
    - Inline data array
    """
    # Handle dict-based data (common case)
    if isinstance(data, dict):
        if "project_name" in data and "dataset_name" in data:
            # Dataset reference by name
            return init_dataset(
                state=state,
                project=data["project_name"],
                name=data["dataset_name"],
                **({"version": data["dataset_version"]} if "dataset_version" in data else {}),
                **({"environment": data["dataset_environment"]} if "dataset_environment" in data else {}),
                # _internal_btql is optional
                **({"_internal_btql": data["_internal_btql"]} if "_internal_btql" in data else {}),
            )
        elif "dataset_id" in data:
            # Dataset reference by ID
            return init_dataset(
                state=state,
                dataset_id=data["dataset_id"],
                **({"version": data["dataset_version"]} if "dataset_version" in data else {}),
                **({"environment": data["dataset_environment"]} if "dataset_environment" in data else {}),
                # _internal_btql is optional
                **({"_internal_btql": data["_internal_btql"]} if "_internal_btql" in data else {}),
            )
        elif "data" in data:
            # Inline data
            return data["data"]

    # If it's not a dict, assume it's inline data
    return data
