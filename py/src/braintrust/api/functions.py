"""Function metadata API service."""

from ._service import ResourceAPI


class FunctionsAPI(ResourceAPI):
    """Synchronous function metadata operations.

    Invocation remains a specialized client and is not implemented here.
    """
