TRANSACTION_ID_FIELD = "_xact_id"
OBJECT_DELETE_FIELD = "_object_delete"

IS_MERGE_FIELD = "_is_merge"

AUDIT_SOURCE_FIELD = "_audit_source"
AUDIT_METADATA_FIELD = "_audit_metadata"
VALID_SOURCES = ["app", "api", "external"]

# Keys that identify which object (experiment, dataset, project logs, etc.) a row belongs to.
OBJECT_ID_KEYS = (
    "experiment_id",
    "dataset_id",
    "prompt_session_id",
    "project_id",
    "log_id",
    "function_data",
)
