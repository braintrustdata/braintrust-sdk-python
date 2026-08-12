"""Dataset API service."""

from ._service import ResourceAPI


class DatasetsAPI(ResourceAPI):
    """Synchronous dataset operations.

    Endpoint methods are added as dataset call sites migrate to the API client.
    """
