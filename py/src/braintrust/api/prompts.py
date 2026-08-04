"""Prompt API service."""

from ._service import ResourceAPI


class PromptsAPI(ResourceAPI):
    """Synchronous prompt operations.

    Endpoint methods are added as prompt call sites migrate to the API client.
    """
