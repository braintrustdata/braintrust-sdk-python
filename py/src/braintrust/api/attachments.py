"""Attachment metadata API service."""

from ._service import ResourceAPI


class AttachmentsAPI(ResourceAPI):
    """Synchronous attachment metadata operations.

    Signed object-storage traffic remains outside the routed transport.
    """
