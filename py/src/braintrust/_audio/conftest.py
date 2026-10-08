"""Audio tests retain attachments locally instead of contacting Braintrust storage."""

import pytest
from braintrust.logger import Attachment


@pytest.fixture(autouse=True)
def local_attachment_uploads(monkeypatch):
    # Override in lifecycle tests to hold/fail uploads. Keep real encoding and payloads.
    monkeypatch.setattr(Attachment, "upload", lambda self: {"upload_status": "done"})
