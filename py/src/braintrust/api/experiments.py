"""Experiment API service."""

from ._service import ResourceAPI


class ExperimentsAPI(ResourceAPI):
    """Synchronous experiment operations.

    Endpoint methods are added as experiment call sites migrate to the API client.
    """
