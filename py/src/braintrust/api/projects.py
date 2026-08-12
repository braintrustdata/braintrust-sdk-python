"""Project API service."""

from ._service import ResourceAPI


class ProjectsAPI(ResourceAPI):
    """Synchronous project operations.

    Endpoint methods are added as project call sites migrate to the API client.
    """
