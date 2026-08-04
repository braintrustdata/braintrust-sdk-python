"""Query API service."""

from ._service import ResourceAPI


class QueriesAPI(ResourceAPI):
    """Synchronous query operations.

    Endpoint methods are added as query call sites migrate to the API client.
    """
